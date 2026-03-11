import { FastifyInstance } from 'fastify';
import '@fastify/multipart';
import { spawn } from 'child_process';
import { createWriteStream, existsSync, unlinkSync } from 'fs';
import { join } from 'path';
import { promisify } from 'util';
import { pipeline } from 'stream';
import { supabase } from '../supabase.js';

const pump = promisify(pipeline);

export async function verificationRoutes(fastify: FastifyInstance) {

  /**
   * GET /api/verification/status
   * Check current identity verification status
   */
  fastify.get('/status', async (request, reply) => {
    const authHeader = request.headers.authorization;
    if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

    try {
      const token = authHeader.replace('Bearer ', '');
      const { data: { user } } = await supabase.auth.getUser(token);
      if (!user) return reply.code(401).send({ error: 'Invalid token' });

      const { data: profile } = await supabase
        .from('profiles')
        .select('identity_verified, identity_verified_at, identity_verification_method')
        .eq('id', user.id)
        .single();

      if (!profile) return reply.code(404).send({ error: 'Profile not found' });

      // Check for active/pending verification sessions
      const { data: latestSession } = await supabase
        .from('identity_verifications')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(1)
        .maybeSingle();

      // Check if reverification is needed (90-day expiry)
      let needsReverification = false;
      if (profile.identity_verified && profile.identity_verified_at) {
        const verifiedAt = new Date(profile.identity_verified_at);
        const daysSince = (Date.now() - verifiedAt.getTime()) / (1000 * 86400);
        needsReverification = daysSince > 90;
      }

      return {
        isVerified: profile.identity_verified || false,
        verifiedAt: profile.identity_verified_at || null,
        method: profile.identity_verification_method || null,
        needsReverification,
        latestSession: latestSession ? {
          id: latestSession.id,
          status: latestSession.status,
          verdict: latestSession.verdict,
          createdAt: latestSession.created_at
        } : null
      };
    } catch (error: any) {
      fastify.log.error(error);
      return reply.code(500).send({ error: 'Failed to check verification status' });
    }
  });

  /**
   * POST /api/verification/selfie
   * Submit a selfie for identity verification
   * Requires in-app camera capture (uploadSource: CAMERA)
   */
  fastify.post('/selfie', async (request, reply) => {
    const tempDir = join(process.cwd(), 'tmp', 'verification');
    let tempPath = '';

    try {
      // Auth
      const authHeader = request.headers.authorization;
      if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });
      const token = authHeader.replace('Bearer ', '');
      const { data: { user }, error: authError } = await supabase.auth.getUser(token);
      if (authError || !user) return reply.code(401).send({ error: 'Invalid token' });

      // Get file
      const data = await request.file();
      if (!data) return reply.code(400).send({ error: 'No file provided' });

      // Enforce camera-only capture
      let uploadSource = 'GALLERY';
      if (data.fields && (data.fields as any).uploadSource) {
        uploadSource = (data.fields as any).uploadSource.value;
      }

      if (uploadSource !== 'CAMERA') {
        return reply.code(400).send({
          error: 'Identity verification requires a live camera capture. Gallery uploads are not allowed.'
        });
      }

      // Device metadata
      let deviceMetadata: any = null;
      if (data.fields && (data.fields as any).deviceMetadata) {
        try {
          deviceMetadata = JSON.parse((data.fields as any).deviceMetadata.value);
        } catch (e) { /* ignore */ }
      }

      // Save temp file
      const fs = await import('fs/promises');
      await fs.mkdir(tempDir, { recursive: true });
      const filename = `selfie_${user.id}_${Date.now()}.jpg`;
      tempPath = join(tempDir, filename);
      await pump(data.file, createWriteStream(tempPath));

      // Check for existing pending verification (prevent spam)
      const { data: recent } = await supabase
        .from('identity_verifications')
        .select('id, created_at')
        .eq('user_id', user.id)
        .eq('status', 'PENDING')
        .gte('created_at', new Date(Date.now() - 5 * 60 * 1000).toISOString())
        .maybeSingle();

      if (recent) {
        if (existsSync(tempPath)) unlinkSync(tempPath);
        return reply.code(429).send({ error: 'Verification already in progress. Please wait.' });
      }

      // Create verification session
      const { data: session, error: sessionError } = await supabase
        .from('identity_verifications')
        .insert({
          user_id: user.id,
          status: 'PROCESSING',
          device_metadata: deviceMetadata
        })
        .select()
        .single();

      if (sessionError) throw sessionError;

      // Run AI selfie verification
      const selfieEnginePath = join(process.cwd(), '..', 'ai_service', 'selfie_verify.py');
      let aiResult: any;

      try {
        aiResult = await runSelfieVerification(selfieEnginePath, tempPath);
      } catch (e: any) {
        // Update session as failed
        await supabase.from('identity_verifications').update({
          status: 'FAILED',
          verdict: 'ERROR',
          reason: `AI Engine Error: ${e.message}`
        }).eq('id', session.id);

        if (existsSync(tempPath)) unlinkSync(tempPath);
        return reply.code(400).send({ error: 'Verification service error. Please try again.' });
      }

      // Process AI result
      const verdict = aiResult.verdict; // VERIFIED, REVIEW, REJECTED
      const finalScore = aiResult.final_score ?? 1.0;
      const signals = aiResult.signals || [];

      let reason = '';
      if (verdict === 'REJECTED') {
        reason = mapSignalsToReason(signals);
      } else if (verdict === 'REVIEW') {
        reason = 'Borderline confidence — requires manual review';
      }

      // Update verification session
      await supabase.from('identity_verifications').update({
        status: verdict === 'VERIFIED' ? 'COMPLETED' : verdict === 'REVIEW' ? 'PENDING_REVIEW' : 'COMPLETED',
        verdict,
        final_score: finalScore,
        liveness_score: aiResult.liveness_score ?? 0,
        spoof_score: aiResult.spoof_score ?? 0,
        ai_face_score: aiResult.ai_face_score ?? 0,
        deepfake_score: aiResult.deepfake_score ?? 0,
        signals,
        reason,
        processing_ms: aiResult.processing_ms ?? 0
      }).eq('id', session.id);

      // If verified, update profile
      if (verdict === 'VERIFIED') {
        await supabase.from('profiles').update({
          identity_verified: true,
          identity_verified_at: new Date().toISOString(),
          identity_verification_method: 'SELFIE_AI'
        }).eq('id', user.id);

        // Send notification
        await supabase.from('notifications').insert({
          user_id: user.id,
          type: 'SYSTEM',
          title: 'Identity Verified',
          message: 'Your identity has been verified! Your profile now displays the Verified Human badge.'
        });
      } else if (verdict === 'REJECTED') {
        await supabase.from('notifications').insert({
          user_id: user.id,
          type: 'SYSTEM',
          title: 'Verification Failed',
          message: reason || 'Your selfie did not pass identity verification. Please try again with better lighting.'
        });
      }

      // Cleanup temp file (never store raw biometric data)
      if (existsSync(tempPath)) unlinkSync(tempPath);

      return {
        verified: verdict === 'VERIFIED',
        verdict,
        score: finalScore,
        reason: verdict !== 'VERIFIED' ? reason : undefined,
        scoreBreakdown: {
          liveness: aiResult.liveness_score ?? 0,
          spoof: aiResult.spoof_score ?? 0,
          aiFace: aiResult.ai_face_score ?? 0,
          deepfake: aiResult.deepfake_score ?? 0
        },
        sessionId: session.id
      };

    } catch (error: any) {
      fastify.log.error(error);
      if (tempPath && existsSync(tempPath)) unlinkSync(tempPath);
      return reply.code(500).send({ error: 'Verification failed due to a system error' });
    }
  });

  /**
   * DELETE /api/verification/data
   * Delete all verification data (GDPR compliance)
   */
  fastify.delete('/data', async (request, reply) => {
    const authHeader = request.headers.authorization;
    if (!authHeader) return reply.code(401).send({ error: 'Unauthorized' });

    try {
      const token = authHeader.replace('Bearer ', '');
      const { data: { user } } = await supabase.auth.getUser(token);
      if (!user) return reply.code(401).send({ error: 'Invalid token' });

      // Delete verification records
      await supabase.from('identity_verifications').delete().eq('user_id', user.id);

      // Remove verification status from profile
      await supabase.from('profiles').update({
        identity_verified: false,
        identity_verified_at: null,
        identity_verification_method: null
      }).eq('id', user.id);

      return { success: true, message: 'All verification data has been deleted.' };
    } catch (error: any) {
      fastify.log.error(error);
      return reply.code(500).send({ error: 'Failed to delete verification data' });
    }
  });
}

async function runSelfieVerification(scriptPath: string, filePath: string): Promise<any> {
  const pythonCmd = await getPythonCommand();
  return new Promise((resolve, reject) => {
    const python = spawn(pythonCmd, [scriptPath, filePath]);
    let stdout = '';
    let stderr = '';
    python.stdout.on('data', (d) => stdout += d.toString());
    python.stderr.on('data', (d) => stderr += d.toString());
    python.on('close', (code) => {
      if (code !== 0) return reject(new Error(`Selfie AI Error ${code}: ${stderr}`));
      try {
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (jsonMatch) resolve(JSON.parse(jsonMatch[0]));
        else reject(new Error('Invalid selfie AI output'));
      } catch (e) { reject(e); }
    });
    setTimeout(() => { python.kill(); reject(new Error('Selfie AI Timeout')); }, 120000);
  });
}

function mapSignalsToReason(signals: string[]): string {
  const reasonMap: Record<string, string> = {
    'no_face_detected': 'No face detected in the selfie. Please ensure your face is clearly visible.',
    'face_too_small': 'Face is too far from the camera. Please move closer.',
    'low_texture_variance': 'Image appears to be a photo of a screen or printed image.',
    'moire_pattern_detected': 'Screen display patterns detected. Please use a live camera.',
    'screen_reflection_artifacts': 'Screen reflections detected in the image.',
    'unnatural_facial_symmetry': 'Face appears artificially generated.',
    'gan_frequency_fingerprint': 'AI-generated face patterns detected.',
    'ai_smooth_skin_texture': 'Artificially smooth skin texture detected.',
    'blue_shift_screen_detected': 'Screen color profile detected.',
    'low_edge_density_print': 'Printed photo characteristics detected.',
    'multi_frame_liveness_confirmed': 'Liveness confirmed across multiple frames.'
  };

  for (const sig of signals) {
    if (reasonMap[sig]) return reasonMap[sig];
  }

  return 'Verification failed due to suspicious image characteristics.';
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
