import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Camera,
  CheckCircle2,
  XCircle,
  Loader2,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Fingerprint,
  Eye,
  Scan,
  Brain,
  Lock,
  ArrowRight,
  RefreshCw,
  Trash2
} from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { BACKEND_URL } from '@/lib/api';

type VerifyStep = 'intro' | 'camera' | 'processing' | 'result';
type Verdict = 'VERIFIED' | 'REVIEW' | 'REJECTED' | null;

interface VerificationStatus {
  isVerified: boolean;
  verifiedAt: string | null;
  needsReverification: boolean;
  method: string | null;
}

interface VerificationResult {
  verified: boolean;
  verdict: string;
  score: number;
  reason?: string;
  scoreBreakdown: {
    liveness: number;
    spoof: number;
    aiFace: number;
    deepfake: number;
  };
}

export default function VerifyIdentity() {
  const navigate = useNavigate();
  const [step, setStep] = useState<VerifyStep>('intro');
  const [status, setStatus] = useState<VerificationStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [verdict, setVerdict] = useState<Verdict>(null);
  const [processingStage, setProcessingStage] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Fetch current verification status
  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) { navigate('/login'); return; }

      const res = await fetch(`${BACKEND_URL}/api/verification/status`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e) {
      console.error('Failed to fetch verification status', e);
    } finally {
      setLoading(false);
    }
  };

  // Camera management
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      setCameraStream(stream);
      setStep('camera');
    } catch (e) {
      console.error('Camera access denied', e);
      alert('Camera access is required for identity verification. Please allow camera access and try again.');
    }
  };

  const cameraVideoRef = useCallback((el: HTMLVideoElement | null) => {
    (videoRef as React.MutableRefObject<HTMLVideoElement | null>).current = el;
    if (el && cameraStream) {
      el.srcObject = cameraStream;
    }
  }, [cameraStream]);

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
  };

  useEffect(() => {
    return () => { stopCamera(); };
  }, []);

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Mirror the capture (selfie mode)
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `selfie_verify_${Date.now()}.jpg`, { type: 'image/jpeg' });
        setCapturedImage(file);
        setPreviewUrl(URL.createObjectURL(blob));
        stopCamera();
      }
    }, 'image/jpeg', 0.95);
  };

  const retakePhoto = () => {
    setCapturedImage(null);
    setPreviewUrl('');
    startCamera();
  };

  // Submit verification
  const submitVerification = async () => {
    if (!capturedImage) return;

    setStep('processing');
    setProcessingStage(0);

    // Animate processing stages
    const stages = [0, 1, 2, 3, 4];
    for (const s of stages) {
      await new Promise(r => setTimeout(r, 800));
      setProcessingStage(s);
    }

    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error('Not authenticated');

      const formData = new FormData();
      formData.append('file', capturedImage);
      formData.append('uploadSource', 'CAMERA');
      formData.append('deviceMetadata', JSON.stringify({
        userAgent: navigator.userAgent,
        timestamp: new Date().toISOString(),
        source: 'TrueFrame Identity Verification',
        screenResolution: `${screen.width}x${screen.height}`
      }));

      const res = await fetch(`${BACKEND_URL}/api/verification/selfie`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${session.access_token}` },
        body: formData
      });

      const data = await res.json();

      if (res.ok) {
        setResult(data);
        setVerdict(data.verdict as Verdict);
      } else {
        setResult(null);
        setVerdict('REJECTED');
      }
    } catch (e: any) {
      console.error('Verification failed', e);
      setVerdict('REJECTED');
    } finally {
      setStep('result');
    }
  };

  const deleteVerificationData = async () => {
    if (!confirm('This will remove your Verified Human badge and delete all verification data. Continue?')) return;

    try {
      setDeleting(true);
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/verification/data`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        await fetchStatus();
        setStep('intro');
        setResult(null);
        setVerdict(null);
      }
    } catch (e) {
      console.error('Failed to delete data', e);
    } finally {
      setDeleting(false);
    }
  };

  const resetFlow = () => {
    setStep('intro');
    setCapturedImage(null);
    setPreviewUrl('');
    setResult(null);
    setVerdict(null);
    stopCamera();
  };

  const processingStages = [
    { icon: Scan, label: 'Detecting face...' },
    { icon: Eye, label: 'Checking liveness...' },
    { icon: Fingerprint, label: 'Analyzing authenticity...' },
    { icon: ShieldCheck, label: 'Running spoof detection...' },
    { icon: Brain, label: 'Verifying identity...' }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background pb-20">
      <div className="max-w-xl mx-auto p-6">

        {/* Header */}
        <div className="mb-8 text-center space-y-2">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="inline-flex items-center gap-2 px-3 py-1 bg-blue-500/10 rounded-full text-blue-500 border border-blue-500/20 mb-4"
          >
            <Fingerprint className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Identity Verification</span>
          </motion.div>
          <h1 className="text-3xl font-black tracking-tight text-foreground">
            Verify Your Identity
          </h1>
          <p className="text-muted-foreground text-sm">
            Prove you're a real human and earn the Verified badge
          </p>
        </div>

        {/* Already Verified Banner */}
        {status?.isVerified && !status.needsReverification && step === 'intro' && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-green-500/10 border border-green-500/20 rounded-2xl p-6 mb-6"
          >
            <div className="flex items-center gap-3 mb-3">
              <CheckCircle2 className="w-8 h-8 text-green-500" />
              <div>
                <h3 className="text-lg font-bold text-green-500">Identity Verified</h3>
                <p className="text-xs text-muted-foreground">
                  Verified on {new Date(status.verifiedAt!).toLocaleDateString()}
                </p>
              </div>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Your profile displays the Verified Human badge. Other users can see you are a real person.
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => startCamera()}
                className="flex-1 py-2 bg-muted text-foreground rounded-xl text-sm font-semibold hover:bg-muted/80 transition-all flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-4 h-4" /> Re-verify
              </button>
              <button
                onClick={deleteVerificationData}
                disabled={deleting}
                className="py-2 px-4 bg-destructive/10 text-destructive rounded-xl text-sm font-semibold hover:bg-destructive/20 transition-all flex items-center justify-center gap-2"
              >
                {deleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
              </button>
            </div>
          </motion.div>
        )}

        {/* Reverification Needed */}
        {status?.needsReverification && step === 'intro' && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-amber-500/10 border border-amber-500/20 rounded-2xl p-6 mb-6"
          >
            <div className="flex items-center gap-3 mb-2">
              <AlertTriangle className="w-6 h-6 text-amber-500" />
              <h3 className="text-lg font-bold text-amber-500">Reverification Required</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              Your verification has expired (90-day limit). Please verify again to keep your badge.
            </p>
          </motion.div>
        )}

        <AnimatePresence mode="wait">
          {/* STEP 1: Intro */}
          {step === 'intro' && (
            <motion.div
              key="intro"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              {/* What You Get */}
              <div className="bg-card rounded-2xl border border-border p-6 mb-6">
                <h3 className="text-sm font-bold uppercase text-muted-foreground mb-4">What You Get</h3>
                <div className="space-y-3">
                  {[
                    { icon: CheckCircle2, color: 'text-blue-500', text: 'Verified Human badge on your profile' },
                    { icon: ShieldCheck, color: 'text-green-500', text: 'Badge displayed on all your posts' },
                    { icon: Eye, color: 'text-purple-500', text: 'Higher trust score & feed visibility' },
                  ].map((item, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <item.icon className={`w-5 h-5 ${item.color}`} />
                      <span className="text-sm font-medium">{item.text}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* How It Works */}
              <div className="bg-card rounded-2xl border border-border p-6 mb-6">
                <h3 className="text-sm font-bold uppercase text-muted-foreground mb-4">How It Works</h3>
                <div className="space-y-4">
                  {[
                    { step: '1', title: 'Take a Selfie', desc: 'Use the in-app camera (no gallery uploads)' },
                    { step: '2', title: 'AI Analysis', desc: 'Liveness, deepfake, and spoof detection' },
                    { step: '3', title: 'Get Verified', desc: 'Badge appears instantly if you pass' },
                  ].map((item, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                        <span className="text-xs font-bold text-primary">{item.step}</span>
                      </div>
                      <div>
                        <p className="text-sm font-bold">{item.title}</p>
                        <p className="text-xs text-muted-foreground">{item.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Privacy */}
              <div className="bg-muted/30 rounded-2xl p-4 mb-6 flex items-start gap-3">
                <Lock className="w-5 h-5 text-muted-foreground flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-xs font-bold text-muted-foreground uppercase mb-1">Privacy First</p>
                  <p className="text-[11px] text-muted-foreground leading-relaxed">
                    Your selfie is analyzed in real-time and immediately deleted. We never store raw biometric data.
                    You can delete all verification data at any time.
                  </p>
                </div>
              </div>

              {/* CTA */}
              {(!status?.isVerified || status?.needsReverification) && (
                <button
                  onClick={startCamera}
                  className="w-full h-14 rounded-2xl gradient-primary text-primary-foreground font-black text-lg shadow-xl shadow-primary/20 hover:shadow-primary/40 transition-all flex items-center justify-center gap-3"
                >
                  <Camera className="w-6 h-6" />
                  <span>Start Verification</span>
                  <ArrowRight className="w-5 h-5" />
                </button>
              )}
            </motion.div>
          )}

          {/* STEP 2: Camera */}
          {step === 'camera' && (
            <motion.div
              key="camera"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
            >
              {!capturedImage ? (
                <>
                  {/* Live Camera */}
                  <div className="relative mb-6">
                    <video
                      ref={cameraVideoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full aspect-[3/4] rounded-[2.5rem] object-cover bg-black"
                      style={{ transform: 'scaleX(-1)' }}
                    />
                    <canvas ref={canvasRef} className="hidden" />

                    {/* Face Guide Overlay */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="w-56 h-72 border-2 border-dashed border-white/40 rounded-[50%]" />
                    </div>

                    {/* Instructions */}
                    <div className="absolute top-4 inset-x-0 flex justify-center">
                      <div className="bg-black/60 backdrop-blur-sm text-white text-xs font-bold px-4 py-2 rounded-full">
                        Position your face inside the oval
                      </div>
                    </div>

                    {/* LIVE indicator */}
                    <div className="absolute top-4 left-4 bg-green-500/90 text-white text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                      LIVE
                    </div>

                    {/* Capture Button */}
                    <div className="absolute bottom-6 inset-x-0 flex justify-center">
                      <button
                        onClick={capturePhoto}
                        className="w-20 h-20 bg-white rounded-full border-4 border-blue-500 shadow-2xl hover:scale-110 transition-transform flex items-center justify-center"
                      >
                        <Camera className="w-8 h-8 text-blue-600" />
                      </button>
                    </div>
                  </div>

                  <div className="text-center space-y-2">
                    <p className="text-sm text-muted-foreground">
                      Make sure you're in a well-lit area with your face clearly visible
                    </p>
                    <button
                      onClick={resetFlow}
                      className="text-xs text-muted-foreground underline hover:text-foreground"
                    >
                      Cancel
                    </button>
                  </div>
                </>
              ) : (
                <>
                  {/* Preview Captured Photo */}
                  <div className="relative mb-6">
                    <img
                      src={previewUrl}
                      alt="Captured selfie"
                      className="w-full aspect-[3/4] rounded-[2.5rem] object-cover"
                    />
                    <div className="absolute top-4 inset-x-0 flex justify-center">
                      <div className="bg-black/60 backdrop-blur-sm text-white text-xs font-bold px-4 py-2 rounded-full">
                        Review your selfie
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={retakePhoto}
                      className="flex-1 h-14 bg-muted text-foreground font-bold rounded-2xl hover:bg-muted/80 transition-all flex items-center justify-center gap-2"
                    >
                      <RefreshCw className="w-5 h-5" />
                      Retake
                    </button>
                    <button
                      onClick={submitVerification}
                      className="flex-1 h-14 gradient-primary text-primary-foreground font-bold rounded-2xl shadow-xl shadow-primary/20 hover:shadow-primary/40 transition-all flex items-center justify-center gap-2"
                    >
                      <ShieldCheck className="w-5 h-5" />
                      Verify Me
                    </button>
                  </div>
                </>
              )}
            </motion.div>
          )}

          {/* STEP 3: Processing */}
          {step === 'processing' && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-card rounded-[2.5rem] border border-border p-8 shadow-2xl"
            >
              <div className="flex flex-col items-center py-8">
                <div className="relative w-32 h-32 mb-8">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                    className="w-full h-full border-4 border-dotted border-blue-500 rounded-full"
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Fingerprint className="w-12 h-12 text-blue-500 animate-pulse" />
                  </div>
                </div>

                <h2 className="text-2xl font-bold mb-6">Analyzing Identity</h2>

                <div className="space-y-3 w-full max-w-xs">
                  {processingStages.map((stage, i) => {
                    const Icon = stage.icon;
                    const isDone = i < processingStage;
                    const isCurrent = i === processingStage;

                    return (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.15 }}
                        className="flex items-center gap-3"
                      >
                        {isDone ? (
                          <CheckCircle2 className="w-4 h-4 text-green-500" />
                        ) : isCurrent ? (
                          <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                        ) : (
                          <div className="w-4 h-4 rounded-full border border-muted-foreground/30" />
                        )}
                        <span className={`text-sm font-medium ${isDone ? 'text-green-500' : isCurrent ? 'text-foreground' : 'text-muted-foreground'}`}>
                          {stage.label}
                        </span>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}

          {/* STEP 4: Result */}
          {step === 'result' && (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-card rounded-[2.5rem] border border-border p-8 shadow-2xl"
            >
              <div className="flex flex-col items-center py-8">
                {verdict === 'VERIFIED' && (
                  <>
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: 'spring', bounce: 0.5 }}
                      className="w-32 h-32 bg-green-500/10 rounded-full flex items-center justify-center mb-6"
                    >
                      <CheckCircle2 className="w-16 h-16 text-green-500" />
                    </motion.div>
                    <h2 className="text-3xl font-black text-green-500 mb-2">Verified Human</h2>
                    <p className="text-muted-foreground text-center max-w-sm mb-6">
                      Your identity has been verified. The Verified Human badge is now visible on your profile and posts.
                    </p>

                    {/* Verification Badge Preview */}
                    <div className="bg-blue-500/10 border border-blue-500/20 rounded-2xl p-4 mb-6 flex items-center gap-3">
                      <CheckCircle2 className="w-6 h-6 text-blue-500" />
                      <div>
                        <p className="text-sm font-bold text-blue-500">Verified Human</p>
                        <p className="text-xs text-muted-foreground">Valid for 90 days</p>
                      </div>
                    </div>

                    <button
                      onClick={() => navigate('/profile')}
                      className="w-full h-14 bg-foreground text-background font-bold rounded-2xl hover:opacity-90 transition-all"
                    >
                      View Your Profile
                    </button>
                  </>
                )}

                {verdict === 'REVIEW' && (
                  <>
                    <div className="w-32 h-32 bg-amber-500/10 rounded-full flex items-center justify-center mb-6">
                      <AlertTriangle className="w-16 h-16 text-amber-500" />
                    </div>
                    <h2 className="text-3xl font-black text-amber-500 mb-2">Under Review</h2>
                    <p className="text-muted-foreground text-center max-w-sm mb-6">
                      Your selfie scored in the borderline range. A manual review may be needed. You can try again with better lighting.
                    </p>
                    <div className="flex gap-3 w-full">
                      <button
                        onClick={resetFlow}
                        className="flex-1 h-14 bg-muted text-foreground font-bold rounded-2xl hover:bg-muted/80 transition-all"
                      >
                        Try Again
                      </button>
                      <button
                        onClick={() => navigate('/profile')}
                        className="flex-1 h-14 bg-foreground text-background font-bold rounded-2xl hover:opacity-90 transition-all"
                      >
                        Go to Profile
                      </button>
                    </div>
                  </>
                )}

                {verdict === 'REJECTED' && (
                  <>
                    <div className="w-32 h-32 bg-destructive/10 rounded-full flex items-center justify-center mb-6">
                      <ShieldAlert className="w-16 h-16 text-destructive" />
                    </div>
                    <h2 className="text-3xl font-black text-destructive mb-2">Verification Failed</h2>

                    <div className="bg-destructive/5 border border-destructive/20 rounded-2xl p-4 mb-6 w-full">
                      <p className="text-sm text-foreground font-medium">
                        {result?.reason || 'Your selfie did not pass identity verification.'}
                      </p>
                    </div>

                    {/* Score Breakdown */}
                    {result?.scoreBreakdown && (
                      <div className="bg-card border border-border rounded-2xl p-6 mb-6 w-full">
                        <p className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-4">Detection Breakdown</p>
                        <div className="grid grid-cols-2 gap-4">
                          {[
                            { label: 'Liveness', key: 'liveness' as const, icon: Eye },
                            { label: 'Anti-Spoof', key: 'spoof' as const, icon: Scan },
                            { label: 'AI Face Check', key: 'aiFace' as const, icon: Brain },
                            { label: 'Deepfake', key: 'deepfake' as const, icon: Fingerprint },
                          ].map((item) => {
                            const score = result.scoreBreakdown[item.key] ?? 0;
                            const color = score < 0.3 ? 'bg-green-500' : score < 0.6 ? 'bg-amber-500' : 'bg-red-500';
                            const textColor = score < 0.3 ? 'text-green-600' : score < 0.6 ? 'text-amber-600' : 'text-red-600';
                            const Icon = item.icon;
                            return (
                              <div key={item.key} className="space-y-1.5">
                                <div className="flex items-center gap-2">
                                  <Icon className={`w-3.5 h-3.5 ${textColor}`} />
                                  <span className="text-[10px] font-bold text-muted-foreground uppercase">{item.label}</span>
                                </div>
                                <div className="h-2 bg-muted rounded-full overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${Math.round(score * 100)}%` }}
                                    transition={{ duration: 0.8, delay: 0.2 }}
                                    className={`h-full rounded-full ${color}`}
                                  />
                                </div>
                                <p className={`text-xs font-bold ${textColor}`}>{(score * 100).toFixed(1)}%</p>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    <div className="space-y-3 w-full">
                      <button
                        onClick={resetFlow}
                        className="w-full h-14 gradient-primary text-primary-foreground font-bold rounded-2xl shadow-xl hover:shadow-primary/40 transition-all flex items-center justify-center gap-2"
                      >
                        <RefreshCw className="w-5 h-5" />
                        Try Again
                      </button>
                      <p className="text-xs text-muted-foreground text-center">
                        Tips: Use good lighting, face the camera directly, remove glasses if possible
                      </p>
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Info Footer */}
        <div className="mt-8 grid grid-cols-2 gap-4">
          <div className="bg-muted/30 p-4 rounded-2xl space-y-2">
            <Lock className="w-5 h-5 text-muted-foreground" />
            <p className="text-xs font-bold uppercase text-muted-foreground">No Data Stored</p>
            <p className="text-[10px] leading-relaxed text-muted-foreground">
              Selfies are analyzed in real-time and deleted immediately. No biometric data is retained.
            </p>
          </div>
          <div className="bg-muted/30 p-4 rounded-2xl space-y-2">
            <RefreshCw className="w-5 h-5 text-muted-foreground" />
            <p className="text-xs font-bold uppercase text-muted-foreground">90-Day Validity</p>
            <p className="text-[10px] leading-relaxed text-muted-foreground">
              Verification expires every 90 days. You'll be prompted to re-verify to keep your badge.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
