import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload as UploadIcon,
  CheckCircle2,
  XCircle,
  Loader2,
  ShieldCheck,
  AlertTriangle,
  FileVideo,
  FileImage,
  ArrowRight,
  ShieldAlert,
  Lock,
  Brain,
  Fingerprint,
  Clock,
  FileSearch,
  Camera,
  ImageIcon
} from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { AuthenticityLabel } from '@/components/ui/AuthenticityLabel';

import { BACKEND_URL } from '@/lib/api';

type VerificationState = 'idle' | 'uploading' | 'verifying' | 'approved' | 'rejected';

export default function Upload() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [caption, setCaption] = useState('');
  const [state, setState] = useState<VerificationState>('idle');
  const [result, setResult] = useState<any>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [uploadMode, setUploadMode] = useState<'GALLERY' | 'CAMERA'>('GALLERY');
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (e) {
      console.error('Camera access denied', e);
      setUploadMode('GALLERY');
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        const capturedFile = new File([blob], `trueframe_capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
        setFile(capturedFile);
        setPreviewUrl(URL.createObjectURL(blob));
        stopCamera();
      }
    }, 'image/jpeg', 0.95);
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
  };

  useEffect(() => {
    return () => { stopCamera(); };
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setState('idle');
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    try {
      setState('uploading');

      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error("Authentication required");

      const formData = new FormData();
      formData.append('caption', caption);
      formData.append('file', file);
      formData.append('uploadSource', uploadMode);
      if (uploadMode === 'CAMERA') {
        formData.append('deviceMetadata', JSON.stringify({
          userAgent: navigator.userAgent,
          timestamp: new Date().toISOString(),
          source: 'TrueFrame Camera'
        }));
      }

      setState('verifying');

      const response = await fetch(`${BACKEND_URL}/api/upload/verify-upload`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`
        },
        body: formData,
      });

      const data = await response.json();
      setResult(data);

      if (data.verified && !data.fakeNews) {
        setState('approved');
        // Navigate to feed after 2 seconds to show success message
        setTimeout(() => {
          navigate('/feed');
        }, 2000);
      } else {
        setState('rejected');
      }
    } catch (error: any) {
      console.error("Upload failed", error);
      setState('rejected');
      setResult({
        verified: false,
        reason: error.message || 'System error encountered during verification'
      });
    }
  };

  const resetUpload = () => {
    setFile(null);
    setCaption('');
    setState('idle');
    setResult(null);
    setPreviewUrl('');
    stopCamera();
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="min-h-screen bg-background pb-20">
      <div className="max-w-xl mx-auto p-6">

        {/* Header Section */}
        <div className="mb-10 text-center space-y-2">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="inline-flex items-center gap-2 px-3 py-1 bg-primary/10 rounded-full text-primary border border-primary/20 mb-4"
          >
            <ShieldCheck className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Synchronous Truth Verification</span>
          </motion.div>
          <h1 className="text-4xl font-black tracking-tight text-foreground">
            Publish Reality
          </h1>
          <p className="text-muted-foreground">
            Our AI engine blocks manipulation instantly. No exceptions.
          </p>
        </div>

        {/* Upload Mode Selector */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => { setUploadMode('GALLERY'); stopCamera(); }}
            className={`flex-1 py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 transition-all ${uploadMode === 'GALLERY' ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:text-foreground'
              }`}
          >
            <ImageIcon className="w-4 h-4" />
            Gallery Upload
          </button>
          <button
            onClick={() => { setUploadMode('CAMERA'); startCamera(); }}
            className={`flex-1 py-3 rounded-xl text-sm font-bold flex items-center justify-center gap-2 transition-all ${uploadMode === 'CAMERA' ? 'bg-green-500 text-white' : 'bg-muted text-muted-foreground hover:text-foreground'
              }`}
          >
            <Camera className="w-4 h-4" />
            Verified Capture
          </button>
        </div>

        {uploadMode === 'CAMERA' && !file && (
          <div className="mb-6 bg-muted/30 rounded-2xl p-3 flex items-start gap-2">
            <ShieldCheck className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-muted-foreground">
              <span className="font-bold text-green-500">Verified Capture mode</span> — Photos taken inside TrueFrame are signed with device metadata and receive the "Captured on TrueFrame" badge.
            </p>
          </div>
        )}

        {/* Camera View */}
        {uploadMode === 'CAMERA' && cameraStream && !file && (
          <div className="mb-6 relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full aspect-[4/5] rounded-[2.5rem] object-cover bg-black"
            />
            <canvas ref={canvasRef} className="hidden" />
            <div className="absolute bottom-6 inset-x-0 flex justify-center">
              <button
                onClick={capturePhoto}
                className="w-16 h-16 bg-white rounded-full border-4 border-green-500 shadow-xl hover:scale-110 transition-transform flex items-center justify-center"
              >
                <Camera className="w-7 h-7 text-green-600" />
              </button>
            </div>
            <div className="absolute top-4 left-4 bg-green-500/90 text-white text-xs font-bold px-3 py-1 rounded-full flex items-center gap-1">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
              LIVE
            </div>
          </div>
        )}

        <div className="relative">
          <AnimatePresence mode="wait">
            {state === 'idle' ? (
              <motion.div
                key="dropzone"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.05 }}
              >
                <div
                  onClick={() => uploadMode === 'GALLERY' ? fileInputRef.current?.click() : undefined}
                  className={`group relative aspect-square w-full rounded-[2.5rem] border-2 border-dashed border-border flex flex-col items-center justify-center gap-6 cursor-pointer hover:border-primary/50 transition-all bg-card/50 hover:bg-primary/5 shadow-inner ${uploadMode === 'CAMERA' && cameraStream && !file ? 'hidden' : ''
                    }`}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    className="hidden"
                    onChange={handleFileSelect}
                    accept="image/*,video/*"
                  />

                  {previewUrl ? (
                    <div className="absolute inset-4 rounded-[1.8rem] overflow-hidden">
                      {file?.type.startsWith('video/') ? (
                        <video src={previewUrl} className="w-full h-full object-cover" muted loop autoPlay />
                      ) : (
                        <img src={previewUrl} className="w-full h-full object-cover" alt="Preview" />
                      )}
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                        <p className="text-white font-bold">Change File</p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="w-24 h-24 rounded-full bg-primary/5 flex items-center justify-center group-hover:scale-110 group-hover:bg-primary/10 transition-all duration-500">
                        <UploadIcon className="w-10 h-10 text-primary" />
                      </div>
                      <div className="text-center">
                        <p className="text-xl font-bold">Drop Media to Verify</p>
                        <p className="text-muted-foreground text-sm mt-1">Images or short videos (Max 50MB)</p>
                      </div>
                    </>
                  )}
                </div>

                <div className="mt-8 space-y-4">
                  <div className="bg-card rounded-2xl border border-border p-4">
                    <label className="text-xs font-bold text-muted-foreground uppercase mb-2 block">Caption (Context Required)</label>
                    <textarea
                      value={caption}
                      onChange={(e) => setCaption(e.target.value)}
                      placeholder="What is happening in this content?"
                      className="w-full bg-transparent border-none focus:ring-0 resize-none text-lg min-h-[100px]"
                    />
                  </div>

                  <button
                    disabled={!file || !caption}
                    onClick={handleUpload}
                    className="w-full h-16 rounded-2xl gradient-primary text-primary-foreground font-black text-lg shadow-xl shadow-primary/20 hover:shadow-primary/40 disabled:opacity-50 disabled:grayscale transition-all flex items-center justify-center gap-3"
                  >
                    <span>Initiate Verification</span>
                    <ArrowRight className="w-6 h-6" />
                  </button>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="status-view"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-card rounded-[2.5rem] border border-border p-8 shadow-2xl relative overflow-hidden"
              >
                {/* Background Progress Visual for Verifying */}
                {state === 'verifying' && (
                  <motion.div
                    initial={{ height: "0%" }}
                    animate={{ height: "100%" }}
                    transition={{ duration: 5, ease: "linear" }}
                    className="absolute inset-0 bg-primary/5 z-0"
                  />
                )}

                <div className="relative z-10 flex flex-col items-center py-10">
                  {state === 'uploading' && (
                    <>
                      <div className="relative w-32 h-32 mb-8">
                        <Loader2 className="w-full h-full text-primary animate-spin-slow" strokeWidth={1} />
                        <div className="absolute inset-0 flex items-center justify-center">
                          <UploadIcon className="w-10 h-10 text-primary" />
                        </div>
                      </div>
                      <h2 className="text-2xl font-bold mb-2">Transporting Media</h2>
                      <p className="text-muted-foreground">Moving content to secure inspection layer...</p>
                    </>
                  )}

                  {state === 'verifying' && (
                    <>
                      <div className="relative w-32 h-32 mb-8">
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                          className="w-full h-full border-4 border-dotted border-primary rounded-full"
                        />
                        <div className="absolute inset-0 flex items-center justify-center">
                          <ShieldCheck className="w-10 h-10 text-primary animate-pulse" />
                        </div>
                      </div>
                      <h2 className="text-2xl font-bold mb-2">Analyzing Truth</h2>
                      <div className="space-y-3 w-full max-w-xs px-4">
                        <div className="flex items-center gap-3">
                          <CheckCircle2 className="w-4 h-4 text-green-500" />
                          <span className="text-xs font-medium">Deepfake Analysis Complete</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <Loader2 className="w-4 h-4 text-primary animate-spin" />
                          <span className="text-xs font-medium">Verifying Factual Credibility...</span>
                        </div>
                      </div>
                    </>
                  )}

                  {state === 'approved' && (
                    <>
                      <div className="w-32 h-32 bg-green-500/10 rounded-full flex items-center justify-center mb-8">
                        <CheckCircle2 className="w-16 h-16 text-green-500" />
                      </div>
                      <h2 className="text-3xl font-black mb-2 text-green-600">Verification Passed</h2>
                      {result?.authenticityLabel && (
                        <div className="mb-4">
                          <AuthenticityLabel label={result.authenticityLabel} />
                        </div>
                      )}
                      <p className="text-muted-foreground mb-8 text-center max-w-sm">
                        This content is visually authentic and factually credible. It has been permanently recorded.
                      </p>
                      <button
                        onClick={() => window.location.href = '/feed'}
                        className="h-14 px-8 bg-foreground text-background font-bold rounded-xl hover:scale-105 transition-transform"
                      >
                        View in Feed
                      </button>
                    </>
                  )}

                  {state === 'rejected' && (
                    <>
                      <div className="w-32 h-32 bg-destructive/10 rounded-full flex items-center justify-center mb-8">
                        <ShieldAlert className="w-16 h-16 text-destructive" />
                      </div>
                      <h2 className="text-3xl font-black mb-2 text-destructive">Upload Blocked</h2>
                      <div className="bg-destructive/5 border border-destructive/20 rounded-2xl p-6 mb-6 w-full">
                        <p className="text-sm font-bold text-destructive uppercase mb-2">
                          {result?.fakeNews ? "CONTEXTUAL INCONSISTENCY" : "AUTHENTICITY FAILURE"}
                        </p>
                        <p className="text-lg font-medium text-foreground">{result?.reason || "Synthetic manipulation detected."}</p>
                        {result?.details && (
                          <div className="mt-4 space-y-2 border-t border-destructive/10 pt-4">
                            {result.details.map((detail: string, i: number) => (
                              <div key={i} className="flex gap-2 text-xs text-destructive/80 font-medium">
                                <span className="opacity-50">•</span>
                                <span>{detail}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Score Breakdown Grid */}
                      {result?.scoreBreakdown && (
                        <div className="bg-card border border-border rounded-2xl p-6 mb-8 w-full">
                          <p className="text-xs font-bold text-muted-foreground uppercase tracking-wider mb-4">Detection Breakdown</p>
                          <div className="grid grid-cols-2 gap-4">
                            {[
                              { label: "Neural Network", key: "model_score", icon: Brain },
                              { label: "Artifact Analysis", key: "artifact_score", icon: Fingerprint },
                              { label: "Temporal Check", key: "temporal_score", icon: Clock },
                              { label: "Metadata Scan", key: "metadata_score", icon: FileSearch },
                            ].map((item) => {
                              const score = result.scoreBreakdown[item.key] ?? 0;
                              const color = score < 0.3 ? "bg-green-500" : score < 0.6 ? "bg-amber-500" : "bg-red-500";
                              const textColor = score < 0.3 ? "text-green-600" : score < 0.6 ? "text-amber-600" : "text-red-600";
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

                      <div className="flex gap-4 w-full">
                        <button
                          onClick={resetUpload}
                          className="flex-1 h-14 bg-muted hover:bg-muted/80 text-foreground font-bold rounded-xl transition-all"
                        >
                          Try Different File
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Global Principles (Always Visible) */}
        <div className="mt-12 grid grid-cols-2 gap-4">
          <div className="bg-muted/30 p-4 rounded-2xl space-y-2">
            <Lock className="w-5 h-5 text-muted-foreground" />
            <p className="text-xs font-bold uppercase text-muted-foreground">Fail-Closed Policy</p>
            <p className="text-[10px] leading-relaxed text-muted-foreground">
              If an error occurs or verification times out, the upload is automatically blocked. Safety over convenience.
            </p>
          </div>
          <div className="bg-muted/30 p-4 rounded-2xl space-y-2">
            <AlertTriangle className="w-5 h-5 text-muted-foreground" />
            <p className="text-xs font-bold uppercase text-muted-foreground">Immutable Trust</p>
            <p className="text-[10px] leading-relaxed text-muted-foreground">
              This verification is tied to your account's Trust Status. Repeated failed attempts will lead to your profile being placed under review.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
