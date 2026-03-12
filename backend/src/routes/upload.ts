import { FastifyInstance, FastifyRequest } from 'fastify';
import '@fastify/multipart';
import { createHash } from 'crypto';
import { createWriteStream, unlinkSync, existsSync } from 'fs';
import { join } from 'path';
import { promisify } from 'util';
import { pipeline } from 'stream';
import { supabase } from '../supabase.js';
import { updateProfileTrustScore, trackDeepfakeAlert } from '../lib/trust.js';
import { runAIScript, getPythonCommand } from '../lib/ai-runner.js';

const pump = promisify(pipeline);

export async function uploadRoutes(fastify: FastifyInstance) {
  // Ensure storage buckets exist
  const ensureBuckets = async () => {
    try {
      const { data: buckets } = await supabase.storage.listBuckets();
      const bucketNames = buckets?.map((b: any) => b.name) || [];
      if (!bucketNames.includes('posts')) {
        await supabase.storage.createBucket('posts', { public: true });
      }
    } catch (e) {
      console.warn('Storage bucket check/creation failed - check permissions');
    }
  };
  ensureBuckets();

  /**
   * POST /api/verify-upload
   * 1. Receive file
   * 2. Verify with AI (CPU/Inline)
   * 3. Log to verification_logs (SQL)
   * 4. IF REAL: Upload to Storage & Create Post
   * 5. IF FAKE: Block & Delete
   */
  fastify.post('/verify-upload', async (request, reply) => {
    const startTime = Date.now();
    const tempDir = join(process.cwd(), 'tmp', 'uploads');
    let tempPath = '';
    let mediaHash = '';
    let userId = '';
    // State to track verdicts
    let deepfakeVerdict = 'REJECTED';
    let fakeNewsVerdict = 'SKIPPED';
    let finalVerdict = 'FAKE';
    let finalReason = 'Verification failed';
    let finalScore = 0;

    try {
      // --- 1. AUTH & SETUP ---
      const authHeader = request.headers.authorization;
      if (!authHeader) return reply.code(401).send({ verified: false, reason: 'Unauthorized' });
      const token = authHeader.replace('Bearer ', '');
      const { data: { user }, error: authError } = await supabase.auth.getUser(token);
      if (authError || !user) return reply.code(401).send({ verified: false, reason: 'Invalid token' });
      userId = user.id;

      const data = await request.file();
      if (!data) return reply.code(400).send({ verified: false, reason: 'No file provided' });

      // Caption
      let caption = '';
      if (data.fields && (data.fields as any).caption) {
        caption = (data.fields as any).caption.value;
      }

      // Upload Source (Camera vs Gallery)
      let uploadSource = 'GALLERY';
      if (data.fields && (data.fields as any).uploadSource) {
        uploadSource = (data.fields as any).uploadSource.value === 'CAMERA' ? 'CAMERA' : 'GALLERY';
      }

      // Device metadata for camera captures
      let deviceMetadata: any = null;
      if (data.fields && (data.fields as any).deviceMetadata) {
        try {
          deviceMetadata = JSON.parse((data.fields as any).deviceMetadata.value);
        } catch (e) { /* ignore parse errors */ }
      }

      // Save Temp
      const fs = await import('fs/promises');
      await fs.mkdir(tempDir, { recursive: true });
      const filename = `${Date.now()}-${data.filename}`;
      tempPath = join(tempDir, filename);
      await pump(data.file, createWriteStream(tempPath));

      // Hash
      const hash = createHash('sha256');
      const fileBuffer = await fs.readFile(tempPath);
      hash.update(fileBuffer);
      mediaHash = hash.digest('hex');

      // --- 1b. ENSURE PROFILE EXISTS ---
      // We need a profile record to track trust status even for blocked uploads
      const { data: existingProfile } = await supabase.from('profiles').select('id').eq('id', userId).single();
      if (!existingProfile) {
        await supabase.from('profiles').insert({
          id: userId,
          username: user.email?.split('@')[0] || `user_${userId.slice(0, 5)}`,
          trust_status: 'NEW_USER',
          real_percentage: 100
        });
      }

      // --- 1c. HASH DUPLICATE CHECK ---
      const { data: previousRejection } = await supabase
        .from('verification_logs')
        .select('id, final_verdict')
        .eq('media_hash', mediaHash)
        .in('final_verdict', ['FAKE', 'REJECTED'])
        .limit(1)
        .maybeSingle();

      if (previousRejection) {
        await logVerification(userId, mediaHash, 'REJECTED', 'SKIPPED', 'FAKE', 1.0, 1.0, 'Previously rejected content (hash match)');
        await updateProfileTrustScore(userId);
        if (existsSync(tempPath)) unlinkSync(tempPath);
        return reply.code(400).send({
          verified: false,
          reason: 'This content was previously rejected. Uploading known-fake content impacts your trust score.'
        });
      }

      // --- 2. DEEPFAKE DETECTION ---
      // Determine if image or video to run appropriate engine
      const isVideo = data.mimetype.startsWith('video');
      const engineFile = isVideo ? 'training/reel_inference.py' : 'main.py';
      const aiEnginePath = join(process.cwd(), '..', 'ai_service', engineFile);
      
      let mediaResult: any;
      try {
        mediaResult = await runAIVerification(aiEnginePath, tempPath);
      } catch (e: any) {
        await logVerification(userId, mediaHash, 'REJECTED', 'SKIPPED', 'REJECTED', 1.0, 1.0, `Engine Error: ${e.message}`);
        await updateProfileTrustScore(userId);
        if (existsSync(tempPath)) unlinkSync(tempPath);
        return reply.code(400).send({ verified: false, reason: e.message || 'Deepfake service error' });
      }

      const modelScore = mediaResult.model_score ?? 0;
      finalScore = mediaResult.final_score ?? 0;
      const modelName = mediaResult.model ?? 'efficientnet-b0';
      const modelVersion = '1.0';

      // Extract component scores for breakdown
      const scoreBreakdown = {
        model_score: mediaResult.model_score ?? 0,
        artifact_score: mediaResult.artifact_score ?? 0,
        temporal_score: mediaResult.temporal_score ?? 0,
        metadata_score: mediaResult.metadata_score ?? 0,
        compression_score: mediaResult.compression_score ?? 0
      };

      if (mediaResult.verdict === 'APPROVED') {
        deepfakeVerdict = 'APPROVED';
      } else if (mediaResult.verdict === 'UNDER_REVIEW') {
        deepfakeVerdict = 'UNDER_REVIEW';
        const sigs = mediaResult.signals || [];
        const reasons = Array.isArray(sigs) ? sigs : [];
        finalReason = reasons.join(', ') || 'Content flagged for review';
      } else {
        deepfakeVerdict = 'REJECTED';
        const sigs = mediaResult.signals || [];
        const reasons = Array.isArray(sigs) ? sigs : [];
        finalReason = reasons.join(', ') || 'Synthetic content detected';
      }

      // --- 3. FAKE NEWS DETECTION (If Media Passed or Under Review) ---
      if (deepfakeVerdict === 'APPROVED' || deepfakeVerdict === 'UNDER_REVIEW') {
        const contextEnginePath = join(process.cwd(), '..', 'ai_service', 'context_verify.py');
        let contextResult;
        try {
          contextResult = await runContextVerification(contextEnginePath, caption, tempPath);
          if (contextResult.verdict === 'ALLOW') {
            fakeNewsVerdict = 'APPROVED';
          } else {
            fakeNewsVerdict = 'REJECTED';
            finalReason = contextResult.verdict === 'BLOCK_FAKE'
              ? 'Misleading factual claim detected'
              : 'Unverified content blocked';
            if (contextResult.reasons) finalReason += `: ${contextResult.reasons.join(', ')}`;
          }
        } catch (e: any) {
          await logVerification(userId, mediaHash, 'APPROVED', 'REJECTED', 'REJECTED', 1.0, 1.0, `Context Engine Error: ${e.message}`);
          await updateProfileTrustScore(userId);
          if (existsSync(tempPath)) unlinkSync(tempPath);
          return reply.code(400).send({ verified: false, reason: 'Context service error' });
        }
      }

      // --- 4. COMPUTE FINAL VERDICT ---
      if (deepfakeVerdict === 'REJECTED' || fakeNewsVerdict === 'REJECTED') {
        finalVerdict = 'FAKE';
      } else if (deepfakeVerdict === 'UNDER_REVIEW') {
        finalVerdict = 'UNDER_REVIEW';
      } else if (deepfakeVerdict === 'APPROVED' && (fakeNewsVerdict === 'APPROVED' || fakeNewsVerdict === 'SKIPPED')) {
        finalVerdict = 'REAL';
      } else {
        finalVerdict = 'FAKE';
      }

      // --- 4b. DETERMINE AUTHENTICITY LABEL ---
      let authenticityLabel = 'VERIFIED_REAL';
      if (finalVerdict === 'FAKE') {
        authenticityLabel = fakeNewsVerdict === 'REJECTED' ? 'REJECTED_MISLEADING' : 'REJECTED_SYNTHETIC';
      } else if (finalVerdict === 'UNDER_REVIEW') {
        authenticityLabel = 'PENDING_REVIEW';
      } else if (uploadSource === 'CAMERA' && finalScore < 0.15) {
        authenticityLabel = 'CAMERA_ORIGINAL';
      } else if (finalScore < 0.10) {
        authenticityLabel = 'CAMERA_ORIGINAL';
      }

      // --- 5. LOG ONCE ---
      const { data: logEntry } = await logVerification(
        userId,
        mediaHash,
        deepfakeVerdict,
        fakeNewsVerdict,
        finalVerdict,
        modelScore,
        finalScore,
        finalReason,
        data.mimetype.startsWith('video') ? 'video' : 'image',
        modelName,
        modelVersion,
        authenticityLabel,
        scoreBreakdown,
        uploadSource,
        deviceMetadata
      );

      // --- 6. UPDATE PROFILE CACHE ---
      await updateProfileTrustScore(userId);

      // --- 7. ACTION ---
      if (finalVerdict === 'UNDER_REVIEW') {
        // Upload content but mark as pending review — not visible in public feed
        const storagePath = `${userId}/${Date.now()}_${data.filename}`;
        const { data: { publicUrl } } = supabase.storage.from('posts').getPublicUrl(storagePath);
        await supabase.storage.from('posts').upload(storagePath, fileBuffer, { contentType: data.mimetype });

        const { data: newPost } = await supabase.from('posts').insert({
          user_id: userId,
          media_url: publicUrl,
          media_type: data.mimetype.startsWith('video') ? 'video' : 'image',
          caption: caption,
          verification_log_id: logEntry?.id,
          media_hash_check: mediaHash,
          authenticity_label: authenticityLabel,
          upload_source: uploadSource,
          visibility: 'UNDER_REVIEW',
          verification_status: 'UNDER_REVIEW'
        }).select('id').single();

        if (newPost) {
          await generateContentProof(newPost.id, mediaHash, userId);
        }

        await supabase.from('notifications').insert({
          user_id: userId,
          type: 'VERIFICATION_PASSED',
          title: 'Content Under Review',
          message: 'Your upload scored in the borderline range and is pending manual review. It will be visible once approved.'
        });

        if (existsSync(tempPath)) unlinkSync(tempPath);
        return {
          verified: true,
          underReview: true,
          fakeNews: false,
          score: finalScore,
          mediaUrl: publicUrl,
          authenticityLabel,
          scoreBreakdown,
          logId: logEntry?.id
        };
      }

      if (finalVerdict === 'REAL') {
        const storagePath = `${userId}/${Date.now()}_${data.filename}`;
        const { data: { publicUrl } } = supabase.storage.from('posts').getPublicUrl(storagePath);
        await supabase.storage.from('posts').upload(storagePath, fileBuffer, { contentType: data.mimetype });

        const { data: newPost } = await supabase.from('posts').insert({
          user_id: userId,
          media_url: publicUrl,
          media_type: data.mimetype.startsWith('video') ? 'video' : 'image',
          caption: caption,
          verification_log_id: logEntry?.id,
          media_hash_check: mediaHash,
          authenticity_label: authenticityLabel,
          upload_source: uploadSource
        }).select('id').single();

        // Generate content proof (blockchain-like hash chain)
        if (newPost) {
          await generateContentProof(newPost.id, mediaHash, userId);
        }

        // Send notification to user
        await supabase.from('notifications').insert({
          user_id: userId,
          type: 'VERIFICATION_PASSED',
          title: 'Content Verified',
          message: `Your ${uploadSource === 'CAMERA' ? 'camera capture' : 'upload'} passed verification with ${authenticityLabel.replace(/_/g, ' ').toLowerCase()} status.`,
          related_post_id: newPost?.id
        });

        if (existsSync(tempPath)) unlinkSync(tempPath);
        return {
          verified: true,
          fakeNews: false,
          score: finalScore,
          mediaUrl: publicUrl,
          authenticityLabel,
          scoreBreakdown,
          logId: logEntry?.id
        };
      } else {
        // Notify user of failed verification
        await supabase.from('notifications').insert({
          user_id: userId,
          type: 'VERIFICATION_FAILED',
          title: 'Upload Blocked',
          message: finalReason || 'Your content did not pass verification.'
        });

        // Track deepfake pattern for alerts
        await trackDeepfakeAlert(mediaHash, finalReason);

        if (existsSync(tempPath)) unlinkSync(tempPath);
        return reply.code(400).send({
          verified: false,
          fakeNews: fakeNewsVerdict === 'REJECTED',
          reason: finalReason,
          score: finalScore,
          authenticityLabel,
          scoreBreakdown,
          logId: logEntry?.id
        });
      }

    } catch (error: any) {
      console.error('[GLOBAL ERROR]', error);
      if (userId && mediaHash) {
        await logVerification(userId, mediaHash, 'REJECTED', 'SKIPPED', 'REJECTED', 1.0, 1.0, `System Error: ${error.message}`);
        await updateProfileTrustScore(userId);
      }
      if (tempPath && existsSync(tempPath)) unlinkSync(tempPath);
      return reply.code(500).send({ verified: false, reason: 'Internal Server Error' });
    }
  });
}

async function runAIVerification(scriptPath: string, filePath: string): Promise<any> {
  return runAIScript(scriptPath, [filePath], 300000); // 5 minutes (allows first run HF download)
}

async function runContextVerification(scriptPath: string, caption: string, filePath: string): Promise<any> {
  return runAIScript(scriptPath, [caption, filePath], 120000); // 2 minutes
}

async function logVerification(
  userId: string,
  mediaHash: string,
  deepfakeVerdict: string,
  fakeNewsVerdict: string,
  finalVerdict: string,
  modelScore: number,
  finalScore: number,
  reason: string | null,
  mediaType: string = 'image',
  modelName: string = 'efficientnet-b0',
  modelVersion: string = '1.0',
  authenticityLabel: string = 'UNKNOWN',
  scoreBreakdown: Record<string, number> | null = null,
  uploadSource: string = 'GALLERY',
  deviceMetadata: any = null
) {
  const insertData: any = {
    user_id: userId,
    media_hash: mediaHash,
    media_type: mediaType,
    deepfake_verdict: deepfakeVerdict,
    fake_news_verdict: fakeNewsVerdict,
    final_verdict: finalVerdict,
    verdict: finalVerdict,
    score: finalScore,
    reason,
    model_name: modelName,
    model_version: modelVersion,
    model_score: modelScore,
    final_score: finalScore,
    authenticity_label: authenticityLabel,
    score_breakdown: scoreBreakdown,
    upload_source: uploadSource,
    device_metadata: deviceMetadata
  };

  return await supabase.from('verification_logs').insert(insertData).select().single();
}

async function generateContentProof(postId: string, mediaHash: string, userId: string) {
  try {
    // Get last proof in chain
    const { data: lastProof } = await supabase
      .from('content_proofs')
      .select('proof_hash, proof_chain_index')
      .order('proof_chain_index', { ascending: false })
      .limit(1)
      .single();

    const previousHash = lastProof?.proof_hash || '0000000000000000';
    const chainIndex = (lastProof?.proof_chain_index || 0) + 1;

    // Create metadata hash
    const metadataString = `${postId}:${userId}:${Date.now()}`;
    const metadataHash = createHash('sha256').update(metadataString).digest('hex');

    // Create proof hash (chain: previous + media + metadata)
    const proofString = `${previousHash}:${mediaHash}:${metadataHash}`;
    const proofHash = createHash('sha256').update(proofString).digest('hex');

    await supabase.from('content_proofs').insert({
      post_id: postId,
      media_hash: mediaHash,
      metadata_hash: metadataHash,
      proof_hash: proofHash,
      previous_proof_hash: previousHash,
      proof_chain_index: chainIndex
    });

    // Update post with proof hash
    await supabase.from('posts')
      .update({ content_hash_proof: proofHash })
      .eq('id', postId);
  } catch (e) {
    console.warn('[PROOF] Content proof generation failed:', e);
  }
}

