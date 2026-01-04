import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload as UploadIcon,
  Image,
  Video,
  Scan,
  ShieldCheck,
  ShieldX,
  AlertTriangle,
  X,
  Send,
} from "lucide-react";
import { UploadStep } from "@/components/upload/UploadStep";

type VerificationStatus = "idle" | "scanning" | "analyzing" | "verified" | "rejected" | "warning";

const statusConfig = {
  idle: { message: "Select media to upload", icon: UploadIcon },
  scanning: { message: "Scanning for deepfake indicators...", icon: Scan },
  analyzing: { message: "Analyzing authenticity...", icon: Scan },
  verified: { message: "Authentic content verified!", icon: ShieldCheck },
  rejected: { message: "Deepfake detected — upload blocked", icon: ShieldX },
  warning: { message: "Potential manipulation detected", icon: AlertTriangle },
};

export default function Upload() {
  const [status, setStatus] = useState<VerificationStatus>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [caption, setCaption] = useState("");

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    
    // Simulate verification process
    setStatus("scanning");
    setTimeout(() => setStatus("analyzing"), 1500);
    setTimeout(() => {
      // Randomly choose outcome for demo
      const outcomes: VerificationStatus[] = ["verified", "verified", "verified", "warning"];
      setStatus(outcomes[Math.floor(Math.random() * outcomes.length)]);
    }, 3500);
  }, []);

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
    setStatus("idle");
    setCaption("");
  };

  const getStepStatus = (step: number) => {
    if (status === "idle") return "pending";
    if (status === "scanning" && step === 1) return "active";
    if (status === "scanning" && step > 1) return "pending";
    if (status === "analyzing" && step <= 1) return "complete";
    if (status === "analyzing" && step === 2) return "active";
    if (status === "analyzing" && step > 2) return "pending";
    if (status === "verified") return step === 3 ? "complete" : "complete";
    if (status === "rejected") return step === 3 ? "error" : "complete";
    if (status === "warning") return step === 3 ? "warning" : "complete";
    return "pending";
  };

  return (
    <div className="min-h-screen bg-background pb-24">
      {/* Header */}
      <header className="sticky top-0 z-40 glass border-b border-border p-4">
        <div className="max-w-lg mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold text-foreground">Create Post</h1>
          {selectedFile && (
            <button onClick={handleReset} className="p-2 hover:bg-muted rounded-full">
              <X className="w-5 h-5 text-muted-foreground" />
            </button>
          )}
        </div>
      </header>

      <main className="max-w-lg mx-auto p-4 space-y-6">
        {/* Upload Area */}
        <AnimatePresence mode="wait">
          {!selectedFile ? (
            <motion.label
              key="upload"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="flex flex-col items-center justify-center gap-6 p-12 border-2 border-dashed border-primary/30 rounded-3xl bg-primary/5 cursor-pointer hover:border-primary/50 transition-colors"
            >
              <div className="p-6 rounded-full gradient-primary">
                <UploadIcon className="w-10 h-10 text-primary-foreground" />
              </div>
              <div className="text-center">
                <p className="text-lg font-semibold text-foreground">Upload Media</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Images up to 10MB • Videos up to 60s
                </p>
              </div>
              <div className="flex gap-4">
                <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-full">
                  <Image className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Photos</span>
                </div>
                <div className="flex items-center gap-2 px-4 py-2 bg-muted rounded-full">
                  <Video className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-muted-foreground">Videos</span>
                </div>
              </div>
              <input
                type="file"
                accept="image/*,video/*"
                className="hidden"
                onChange={handleFileSelect}
              />
            </motion.label>
          ) : (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-6"
            >
              {/* Preview */}
              <div className="relative rounded-3xl overflow-hidden shadow-card">
                {selectedFile.type.startsWith("image") ? (
                  <img
                    src={preview!}
                    alt="Preview"
                    className="w-full aspect-[4/5] object-cover"
                  />
                ) : (
                  <video
                    src={preview!}
                    className="w-full aspect-[4/5] object-cover"
                    controls
                  />
                )}
                
                {/* Status Overlay */}
                <AnimatePresence>
                  {status !== "verified" && status !== "idle" && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 bg-foreground/80 flex items-center justify-center"
                    >
                      <div className="text-center text-primary-foreground">
                        {status === "scanning" || status === "analyzing" ? (
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            className="w-16 h-16 mx-auto mb-4 rounded-full border-4 border-primary-foreground/30 border-t-primary-foreground"
                          />
                        ) : status === "rejected" ? (
                          <ShieldX className="w-16 h-16 mx-auto mb-4 text-destructive" />
                        ) : (
                          <AlertTriangle className="w-16 h-16 mx-auto mb-4 text-warning" />
                        )}
                        <p className="text-lg font-semibold">{statusConfig[status].message}</p>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Verified Badge */}
                {status === "verified" && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute top-4 right-4 flex items-center gap-2 bg-verified px-4 py-2 rounded-full shadow-verified"
                  >
                    <ShieldCheck className="w-5 h-5 text-verified-foreground" />
                    <span className="font-semibold text-verified-foreground">Verified</span>
                  </motion.div>
                )}
              </div>

              {/* Verification Steps */}
              <div className="space-y-3">
                <UploadStep
                  icon={Scan}
                  title="Scanning Media"
                  description="Extracting frames and metadata"
                  status={getStepStatus(1)}
                />
                <UploadStep
                  icon={Scan}
                  title="AI Analysis"
                  description="Checking for synthetic content"
                  status={getStepStatus(2)}
                />
                <UploadStep
                  icon={ShieldCheck}
                  title="Verification Result"
                  description={
                    status === "verified"
                      ? "Content is authentic"
                      : status === "rejected"
                      ? "Deepfake detected"
                      : status === "warning"
                      ? "Needs review"
                      : "Awaiting scan completion"
                  }
                  status={getStepStatus(3)}
                />
              </div>

              {/* Caption & Publish */}
              {status === "verified" && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-4"
                >
                  <textarea
                    value={caption}
                    onChange={(e) => setCaption(e.target.value)}
                    placeholder="Write a caption..."
                    className="w-full p-4 bg-muted rounded-2xl resize-none h-24 focus:outline-none focus:ring-2 focus:ring-primary/50"
                  />
                  <button className="w-full flex items-center justify-center gap-3 py-4 gradient-primary text-primary-foreground rounded-2xl font-semibold text-lg shadow-glow hover:opacity-90 transition-opacity">
                    <Send className="w-5 h-5" />
                    Publish
                  </button>
                </motion.div>
              )}

              {/* Rejection Message */}
              {(status === "rejected" || status === "warning") && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`p-4 rounded-2xl ${
                    status === "rejected" ? "bg-destructive/10 border border-destructive/30" : "bg-warning/10 border border-warning/30"
                  }`}
                >
                  <p className={`font-semibold ${status === "rejected" ? "text-destructive" : "text-warning"}`}>
                    {status === "rejected" ? "Upload Blocked" : "Content Flagged"}
                  </p>
                  <p className="text-sm text-muted-foreground mt-1">
                    {status === "rejected"
                      ? "Synthetic facial artifacts detected. This content cannot be published."
                      : "Potential manipulation detected. You can request a manual review."}
                  </p>
                  {status === "warning" && (
                    <button className="mt-4 px-4 py-2 bg-warning text-warning-foreground rounded-xl font-medium text-sm">
                      Request Review
                    </button>
                  )}
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
