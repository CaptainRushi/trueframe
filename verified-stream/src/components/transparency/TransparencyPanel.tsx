import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X, ShieldCheck, Brain, Fingerprint, Clock, FileSearch, Camera,
  Upload, Link2, AlertTriangle, CheckCircle2, Eye
} from "lucide-react";
import { BACKEND_URL } from "@/lib/api";

interface TransparencyPanelProps {
  postId: string;
  isOpen: boolean;
  onClose: () => void;
}

interface TransparencyData {
  postId: string;
  authenticityScore: number;
  deepfakeProbability: number;
  aiGeneratedProbability: number;
  metadataIntegrity: string;
  uploadSource: string;
  uploadType: string;
  mediaType: string;
  authenticityLabel: string;
  verifiedAt: string;
  modelUsed: string;
  modelVersion: string;
  scoreBreakdown: {
    neuralNetwork: number;
    artifactAnalysis: number;
    temporalConsistency: number;
    metadataScan: number;
    compressionAnalysis: number;
  };
  author: {
    username: string;
    display_name: string;
    trust_score: number;
    trust_status: string;
    is_verified_creator: boolean;
    identity_verified: boolean;
  };
  contentProof: {
    proofHash: string;
    chainIndex: number;
    verifiedAt: string;
  } | null;
  communityFlags: {
    total: number;
    confirmed: number;
  };
  secondaryReview: {
    status: string;
    decision: string | null;
    secondaryScore: number | null;
    signals: string[];
    triggeredAt: string;
    completedAt: string | null;
  } | null;
}

export function TransparencyPanel({ postId, isOpen, onClose }: TransparencyPanelProps) {
  const [data, setData] = useState<TransparencyData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen && postId) {
      fetchTransparencyData();
    }
  }, [isOpen, postId]);

  const fetchTransparencyData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/transparency/post/${postId}`);
      if (res.ok) {
        const result = await res.json();
        setData(result);
      }
    } catch (e) {
      console.error("Failed to fetch transparency data", e);
    } finally {
      setLoading(false);
    }
  };

  const scoreItems = data ? [
    { label: "Neural Network", value: data.scoreBreakdown.neuralNetwork, icon: Brain, color: "text-blue-500" },
    { label: "Artifact Analysis", value: data.scoreBreakdown.artifactAnalysis, icon: Fingerprint, color: "text-purple-500" },
    { label: "Temporal Check", value: data.scoreBreakdown.temporalConsistency, icon: Clock, color: "text-cyan-500" },
    { label: "Metadata Scan", value: data.scoreBreakdown.metadataScan, icon: FileSearch, color: "text-amber-500" },
  ] : [];

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/80 flex items-end sm:items-center justify-center"
          onClick={onClose}
        >
          <motion.div
            initial={{ y: 100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 100, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="bg-card w-full max-w-md max-h-[85vh] overflow-y-auto rounded-t-3xl sm:rounded-3xl border border-border shadow-2xl"
          >
            {/* Header */}
            <div className="sticky top-0 bg-card border-b border-border p-4 flex items-center justify-between rounded-t-3xl">
              <div className="flex items-center gap-2">
                <Eye className="w-5 h-5 text-primary" />
                <h2 className="text-lg font-black">Content Transparency</h2>
              </div>
              <button onClick={onClose} className="p-2 hover:bg-muted rounded-full transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            {loading ? (
              <div className="p-8 text-center">
                <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-4" />
                <p className="text-muted-foreground text-sm">Loading verification data...</p>
              </div>
            ) : data ? (
              <div className="p-5 space-y-5">
                {/* Authenticity Score */}
                <div className="bg-muted/30 rounded-2xl p-5">
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest">Authenticity Score</p>
                    <div className={`px-3 py-1 rounded-full text-xs font-bold ${data.authenticityScore >= 90 ? 'bg-green-500/10 text-green-500' :
                      data.authenticityScore >= 70 ? 'bg-yellow-500/10 text-yellow-500' :
                        'bg-red-500/10 text-red-500'
                      }`}>
                      {data.authenticityScore}%
                    </div>
                  </div>
                  <div className="h-3 bg-muted rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${data.authenticityScore}%` }}
                      transition={{ duration: 1 }}
                      className={`h-full rounded-full ${data.authenticityScore >= 90 ? 'bg-green-500' :
                        data.authenticityScore >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                    />
                  </div>
                </div>

                {/* Key Metrics */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-muted/30 rounded-xl p-4 space-y-1">
                    <p className="text-[10px] font-bold uppercase text-muted-foreground">Deepfake Probability</p>
                    <p className={`text-2xl font-black ${data.deepfakeProbability <= 5 ? 'text-green-500' : data.deepfakeProbability <= 20 ? 'text-yellow-500' : 'text-red-500'}`}>
                      {data.deepfakeProbability}%
                    </p>
                  </div>
                  <div className="bg-muted/30 rounded-xl p-4 space-y-1">
                    <p className="text-[10px] font-bold uppercase text-muted-foreground">AI Generated Prob.</p>
                    <p className={`text-2xl font-black ${data.aiGeneratedProbability <= 5 ? 'text-green-500' : data.aiGeneratedProbability <= 20 ? 'text-yellow-500' : 'text-red-500'}`}>
                      {data.aiGeneratedProbability}%
                    </p>
                  </div>
                  <div className="bg-muted/30 rounded-xl p-4 space-y-1">
                    <p className="text-[10px] font-bold uppercase text-muted-foreground">Metadata Integrity</p>
                    <div className="flex items-center gap-1.5">
                      <CheckCircle2 className={`w-4 h-4 ${data.metadataIntegrity === 'Valid' ? 'text-green-500' : 'text-yellow-500'}`} />
                      <p className="text-sm font-bold">{data.metadataIntegrity}</p>
                    </div>
                  </div>
                  <div className="bg-muted/30 rounded-xl p-4 space-y-1">
                    <p className="text-[10px] font-bold uppercase text-muted-foreground">Upload Type</p>
                    <div className="flex items-center gap-1.5">
                      {data.uploadSource === 'CAMERA' ? (
                        <Camera className="w-4 h-4 text-primary" />
                      ) : (
                        <Upload className="w-4 h-4 text-muted-foreground" />
                      )}
                      <p className="text-sm font-bold">{data.uploadType}</p>
                    </div>
                  </div>
                </div>

                {/* Detection Breakdown */}
                <div className="bg-muted/30 rounded-2xl p-5">
                  <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest mb-4">Detection Breakdown</p>
                  <div className="space-y-3">
                    {scoreItems.map((item) => {
                      const percentage = Math.round(item.value * 100);
                      const barColor = percentage < 20 ? 'bg-green-500' : percentage < 50 ? 'bg-yellow-500' : 'bg-red-500';
                      return (
                        <div key={item.label} className="space-y-1.5">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <item.icon className={`w-3.5 h-3.5 ${item.color}`} />
                              <span className="text-xs font-medium">{item.label}</span>
                            </div>
                            <span className="text-xs font-bold">{percentage}%</span>
                          </div>
                          <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${percentage}%` }}
                              transition={{ duration: 0.8 }}
                              className={`h-full rounded-full ${barColor}`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Verification Info */}
                <div className="bg-muted/30 rounded-2xl p-5 space-y-3">
                  <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest">Verification Details</p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Model</span>
                      <span className="font-medium">{data.modelUsed} v{data.modelVersion}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Verified At</span>
                      <span className="font-medium">{data.verifiedAt ? new Date(data.verifiedAt).toLocaleString() : 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Media Type</span>
                      <span className="font-medium capitalize">{data.mediaType}</span>
                    </div>
                  </div>
                </div>

                {/* Content Proof */}
                {data.contentProof && (
                  <div className="bg-primary/5 border border-primary/20 rounded-2xl p-5 space-y-3">
                    <div className="flex items-center gap-2">
                      <Link2 className="w-4 h-4 text-primary" />
                      <p className="text-xs font-bold uppercase text-primary tracking-widest">Proof of Authenticity</p>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div>
                        <span className="text-muted-foreground text-xs">Hash Proof</span>
                        <p className="font-mono text-[10px] text-foreground/80 break-all mt-0.5">
                          {data.contentProof.proofHash}
                        </p>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Chain Index</span>
                        <span className="font-bold text-primary">#{data.contentProof.chainIndex}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Community Flags */}
                {data.communityFlags.total > 0 && (
                  <div className="bg-yellow-500/5 border border-yellow-500/20 rounded-2xl p-4 flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-bold">Community Review</p>
                      <p className="text-xs text-muted-foreground">
                        {data.communityFlags.total} flag(s), {data.communityFlags.confirmed} confirmed
                      </p>
                    </div>
                  </div>
                )}

                {/* Secondary AI Review */}
                {data.secondaryReview && (
                  <div className={`rounded-2xl p-4 border space-y-3 ${
                    data.secondaryReview.status === 'PROCESSING' || data.secondaryReview.status === 'PENDING'
                      ? 'bg-yellow-500/5 border-yellow-500/20'
                      : data.secondaryReview.decision === 'RESTORE'
                        ? 'bg-green-500/5 border-green-500/20'
                        : data.secondaryReview.decision === 'REMOVE'
                          ? 'bg-red-500/5 border-red-500/20'
                          : 'bg-yellow-500/5 border-yellow-500/20'
                  }`}>
                    <div className="flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4 text-primary" />
                      <p className="text-xs font-bold uppercase tracking-widest">Secondary AI Review</p>
                    </div>

                    {(data.secondaryReview.status === 'PENDING' || data.secondaryReview.status === 'PROCESSING') && (
                      <div className="flex items-center gap-3">
                        <div className="w-5 h-5 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin" />
                        <p className="text-sm text-yellow-400">Secondary AI analysis in progress...</p>
                      </div>
                    )}

                    {data.secondaryReview.status === 'COMPLETED' && (
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-xs text-muted-foreground">Secondary Score</span>
                          <span className={`text-sm font-bold ${
                            (data.secondaryReview.secondaryScore || 0) < 0.6 ? 'text-green-500' :
                            (data.secondaryReview.secondaryScore || 0) < 0.8 ? 'text-yellow-500' : 'text-red-500'
                          }`}>
                            {Math.round((data.secondaryReview.secondaryScore || 0) * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-xs text-muted-foreground">Decision</span>
                          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${
                            data.secondaryReview.decision === 'RESTORE' ? 'bg-green-500/10 text-green-500' :
                            data.secondaryReview.decision === 'MANUAL_REVIEW' ? 'bg-yellow-500/10 text-yellow-500' :
                            'bg-red-500/10 text-red-500'
                          }`}>
                            {data.secondaryReview.decision === 'RESTORE' ? 'Verified Authentic' :
                             data.secondaryReview.decision === 'MANUAL_REVIEW' ? 'Pending Manual Review' :
                             data.secondaryReview.decision === 'REMOVE' ? 'Confirmed Deepfake' :
                             data.secondaryReview.decision || 'Unknown'}
                          </span>
                        </div>
                        {data.secondaryReview.completedAt && (
                          <div className="flex justify-between items-center">
                            <span className="text-xs text-muted-foreground">Reviewed At</span>
                            <span className="text-xs font-medium">
                              {new Date(data.secondaryReview.completedAt).toLocaleString()}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {data.secondaryReview.status === 'FAILED' && (
                      <p className="text-xs text-yellow-400">
                        Review could not be completed — post remains under review.
                      </p>
                    )}
                  </div>
                )}

                {/* Author Info */}
                <div className="bg-muted/30 rounded-2xl p-5 space-y-3">
                  <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest">Author</p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-bold">@{data.author?.username}</span>
                      {data.author?.is_verified_creator && (
                        <ShieldCheck className="w-4 h-4 text-primary" />
                      )}
                    </div>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${data.author?.trust_status === 'TRUSTED' ? 'bg-green-500/10 text-green-500' :
                      data.author?.trust_status === 'NEW_USER' ? 'bg-blue-500/10 text-blue-500' :
                        'bg-yellow-500/10 text-yellow-500'
                      }`}>
                      Trust: {data.author?.trust_score}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center">
                <AlertTriangle className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                <p className="text-muted-foreground text-sm">Failed to load transparency data</p>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
