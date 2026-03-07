import { FastifyInstance, FastifyRequest } from 'fastify';
import '@fastify/multipart';
import { spawn } from 'child_process';
import { createHash } from 'crypto';
import { createReadStream, createWriteStream, unlinkSync, existsSync } from 'fs';
import { join } from 'path';
import { promisify } from 'util';
import { pipeline } from 'stream';
import { supabase } from '../supabase.js';

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
      const aiEnginePath = join(process.cwd(), '..', 'ai_service', 'main.py');
      let mediaResult: any;
      try {
        mediaResult = await runAIVerification(aiEnginePath, tempPath);
      } catch (e: any) {
        await logVerification(userId, mediaHash, 'REJECTED', 'SKIPPED', 'REJECTED', 1.0, 1.0, `Engine Error: ${e.message}`);
        await updateProfileTrustScore(userId);
        if (existsSync(tempPath)) unlinkSync(tempPath);
        return reply.code(400).send({ verified: false, reason: 'Deepfake service error' });
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
      } else {
        deepfakeVerdict = 'REJECTED';
        const sigs = mediaResult.signals || [];
        const reasons = Array.isArray(sigs) ? sigs : [];
        finalReason = reasons.join(', ') || 'Synthetic content detected';
      }

      // --- 3. FAKE NEWS DETECTION (If Media Passed) ---
      if (deepfakeVerdict === 'APPROVED') {
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
      finalVerdict = (deepfakeVerdict === 'APPROVED' && fakeNewsVerdict === 'APPROVED') ? 'REAL' : 'FAKE';

      // --- 4b. DETERMINE AUTHENTICITY LABEL ---
      let authenticityLabel = 'VERIFIED_REAL';
      if (finalVerdict === 'FAKE') {
        authenticityLabel = fakeNewsVerdict === 'REJECTED' ? 'REJECTED_MISLEADING' : 'REJECTED_SYNTHETIC';
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
  const pythonCmd = await getPythonCommand();
  return new Promise((resolve, reject) => {
    const python = spawn(pythonCmd, [scriptPath, filePath]);
    let stdout = '';
    let stderr = '';
    python.stdout.on('data', (d) => stdout += d.toString());
    python.stderr.on('data', (d) => stderr += d.toString());
    python.on('close', (code) => {
      if (code !== 0) return reject(new Error(`AI Error ${code}: ${stderr}`));
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) resolve(JSON.parse(jsonMatch[0]));
        else reject(new Error('Invalid AI output'));
      } catch (e) { reject(e); }
    });
    setTimeout(() => { python.kill(); reject(new Error('AI Timeout')); }, 15000);
  });
}

async function runContextVerification(scriptPath: string, caption: string, filePath: string): Promise<any> {
  const pythonCmd = await getPythonCommand();
  return new Promise((resolve, reject) => {
    const python = spawn(pythonCmd, [scriptPath, caption, filePath]);
    let stdout = '';
    let stderr = '';
    python.stdout.on('data', (d) => stdout += d.toString());
    python.stderr.on('data', (d) => stderr += d.toString());
    python.on('close', (code) => {
      if (code !== 0) return reject(new Error(`Context Error ${code}: ${stderr}`));
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) resolve(JSON.parse(jsonMatch[0]));
        else reject(new Error('Invalid Context output'));
      } catch (e) { reject(e); }
    });
    setTimeout(() => { python.kill(); reject(new Error('Context Timeout')); }, 5000);
  });
}

async function updateProfileTrustScore(userId: string) {
  try {
    // Fetch profile for account age
    const { data: profile } = await supabase
      .from('profiles')
      .select('created_at')
      .eq('id', userId)
      .single();

    // Fetch verification counts
    const [{ count: total }, { count: real }] = await Promise.all([
      supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', userId),
      supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', userId).in('final_verdict', ['REAL', 'APPROVED'])
    ]);

    const totalUploads = total || 0;
    const verifiedUploads = real || 0;
    const rejectedUploads = totalUploads - verifiedUploads;
    const realPercentage = totalUploads > 0 ? Math.round((verifiedUploads / totalUploads) * 100) : 100;
    const fakePercentage = totalUploads > 0 ? Math.round((rejectedUploads / totalUploads) * 100) : 0;

    // COMPONENT 1: Verification Rate (0-40 points)
    const verificationRate = totalUploads > 0 ? (verifiedUploads / totalUploads) : 0.5;
    const verificationScore = Math.round(verificationRate * 40);

    // COMPONENT 2: Account Age (0-20 points) — maxes at 90 days
    const accountAgeDays = profile?.created_at
      ? (Date.now() - new Date(profile.created_at).getTime()) / (1000 * 86400)
      : 0;
    const ageScore = Math.min(20, Math.round((accountAgeDays / 90) * 20));

    // COMPONENT 3: Activity Volume (0-20 points) — maxes at 50 uploads
    const volumeScore = Math.min(20, Math.round((totalUploads / 50) * 20));

    // COMPONENT 4: Community Reputation (0-20 points)
    const { count: followers } = await supabase
      .from('follows')
      .select('*', { count: 'exact', head: true })
      .eq('following_id', userId);
    const followerScore = Math.min(10, Math.round(((followers || 0) / 100) * 10));

    const { data: userPosts } = await supabase
      .from('posts')
      .select('like_count')
      .eq('user_id', userId);
    const totalLikeCount = userPosts?.reduce((sum: number, p: any) => sum + (p.like_count || 0), 0) || 0;
    const engagementScore = Math.min(10, Math.round((totalLikeCount / 500) * 10));
    const communityScore = followerScore + engagementScore;

    // FINAL TRUST SCORE (0-100)
    const trustScore = Math.min(100, verificationScore + ageScore + volumeScore + communityScore);

    // Derive status from score
    let status = 'TRUSTED';
    if (totalUploads === 0) status = 'NEW_USER';
    else if (trustScore < 25) status = 'UNDER_REVIEW';
    else if (trustScore < 50) status = 'AT_RISK';

    await supabase.from('profiles').update({
      trust_status: status,
      trust_score: trustScore,
      trust_score_updated_at: new Date().toISOString(),
      real_percentage: realPercentage,
      fake_percentage: fakePercentage,
      total_attempts: totalUploads,
      real_count: verifiedUploads,
      fake_count: rejectedUploads
    }).eq('id', userId);

    // Insert trust score history snapshot
    await supabase.from('trust_score_history').insert({
      user_id: userId,
      trust_score: trustScore
    });

    console.log(`[TRUST-CACHE] User ${userId}: ${status} (Score: ${trustScore}, ${realPercentage}% Real)`);
  } catch (e) { console.warn(`[TRUST-CACHE] Failed for ${userId}`, e); }
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

async function trackDeepfakeAlert(mediaHash: string, reason: string) {
  try {
    // Check if this hash was already flagged
    const { data: existing } = await supabase
      .from('deepfake_alerts')
      .select('id, detection_count')
      .eq('media_hash', mediaHash)
      .single();

    if (existing) {
      await supabase.from('deepfake_alerts')
        .update({
          detection_count: existing.detection_count + 1,
          severity: existing.detection_count >= 5 ? 'CRITICAL' : existing.detection_count >= 3 ? 'HIGH' : 'MEDIUM',
          updated_at: new Date().toISOString()
        })
        .eq('id', existing.id);
    } else {
      await supabase.from('deepfake_alerts').insert({
        title: 'Deepfake Content Detected',
        description: reason,
        media_hash: mediaHash,
        severity: 'LOW'
      });
    }
  } catch (e) {
    console.warn('[ALERT] Deepfake alert tracking failed:', e);
  }
}

async function getPythonCommand(): Promise<string> {
  if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;
  const { execSync } = await import('child_process');
  try { execSync('python3 --version', { stdio: 'ignore' }); return 'python3'; }
  catch (e) {
    try { execSync('python --version', { stdio: 'ignore' }); return 'python'; }
    catch (e2) { throw new Error('Python not found'); }
  }
}
