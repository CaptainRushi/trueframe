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

      // --- 2. DEEPFAKE DETECTION ---
      const aiEnginePath = join(process.cwd(), '..', 'ai_service', 'main.py');
      let mediaResult;
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
        modelVersion
      );

      // --- 6. UPDATE PROFILE CACHE ---
      await updateProfileTrustScore(userId);

      // --- 7. ACTION ---
      if (finalVerdict === 'REAL') {
        const storagePath = `${userId}/${Date.now()}_${data.filename}`;
        const { data: { publicUrl } } = supabase.storage.from('posts').getPublicUrl(storagePath);
        await supabase.storage.from('posts').upload(storagePath, fileBuffer, { contentType: data.mimetype });

        await supabase.from('posts').insert({
          user_id: userId,
          media_url: publicUrl,
          media_type: data.mimetype.startsWith('video') ? 'video' : 'image',
          caption: caption,
          verification_log_id: logEntry?.id,
          media_hash_check: mediaHash
        });

        if (existsSync(tempPath)) unlinkSync(tempPath);
        return { verified: true, fakeNews: false, score: finalScore, mediaUrl: publicUrl };
      } else {
        if (existsSync(tempPath)) unlinkSync(tempPath);
        return reply.code(400).send({
          verified: false,
          fakeNews: fakeNewsVerdict === 'REJECTED',
          reason: finalReason,
          score: finalScore
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
    const [{ count: total }, { count: real }] = await Promise.all([
      supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', userId),
      supabase.from('verification_logs').select('*', { count: 'exact', head: true }).eq('user_id', userId).in('final_verdict', ['REAL', 'APPROVED'])
    ]);

    const totalUploads = total || 0;
    const verifiedUploads = real || 0;
    const rejectedUploads = totalUploads - verifiedUploads;
    const realPercentage = totalUploads > 0 ? Math.round((verifiedUploads / totalUploads) * 100) : 100;
    const fakePercentage = totalUploads > 0 ? Math.round((rejectedUploads / totalUploads) * 100) : 0;

    let status = 'TRUSTED';
    if (totalUploads === 0) status = 'NEW_USER';
    else if (fakePercentage > 30) status = 'RESTRICTED';
    else if (fakePercentage > 10) status = 'AT_RISK';

    await supabase.from('profiles').update({
      trust_status: status,
      real_percentage: realPercentage,
      fake_percentage: fakePercentage,
      total_attempts: totalUploads,
      real_count: verifiedUploads,
      fake_count: rejectedUploads
    }).eq('id', userId);
    console.log(`[TRUST-CACHE] User ${userId}: ${status} (${realPercentage}% Real, ${fakePercentage}% Fake)`);
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
  modelVersion: string = '1.0'
) {
  // Map finalScore to the legacy 'score' column for compatibility
  const insertData: any = {
    user_id: userId,
    media_hash: mediaHash,
    media_type: mediaType,
    deepfake_verdict: deepfakeVerdict,
    fake_news_verdict: fakeNewsVerdict,
    final_verdict: finalVerdict,
    verdict: finalVerdict, // compatibility
    score: finalScore,     // compatibility
    reason,
    model_name: modelName,
    model_version: modelVersion,
    model_score: modelScore,
    final_score: finalScore
  };

  return await supabase.from('verification_logs').insert(insertData).select().single();
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
