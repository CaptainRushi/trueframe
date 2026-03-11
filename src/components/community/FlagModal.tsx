import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Flag, AlertTriangle, Link2, Send } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";

interface FlagModalProps {
  postId: string;
  isOpen: boolean;
  onClose: () => void;
}

const FLAG_TYPES = [
  { value: 'MISINFORMATION', label: 'Misinformation', desc: 'Contains false or misleading claims' },
  { value: 'MANIPULATED', label: 'Manipulated Media', desc: 'Image/video has been altered' },
  { value: 'OUT_OF_CONTEXT', label: 'Out of Context', desc: 'Real content used misleadingly' },
  { value: 'SPAM', label: 'Spam', desc: 'Promotional or repetitive content' },
  { value: 'OTHER', label: 'Other', desc: 'Other concern' },
];

export function FlagModal({ postId, isOpen, onClose }: FlagModalProps) {
  const [flagType, setFlagType] = useState('');
  const [reason, setReason] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);

  const handleSubmit = async () => {
    if (!flagType) return;
    setSubmitting(true);

    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error("Not authenticated");

      const res = await fetch(`${BACKEND_URL}/api/community/flag/${postId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ flagType, reason, sourceUrl: sourceUrl || undefined })
      });

      const data = await res.json();
      if (res.ok) {
        const reviewMsg = data.reviewTriggered
          ? 'Your flag triggered an automatic secondary AI review of this content.'
          : 'Community verifiers will review this content.';
        setResult({ success: true, message: `Flag submitted successfully. ${reviewMsg}` });
      } else {
        setResult({ success: false, message: data.error || data.message || 'Failed to submit flag' });
      }
    } catch (e: any) {
      setResult({ success: false, message: e.message });
    } finally {
      setSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-end sm:items-center justify-center"
        onClick={onClose}
      >
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 100, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
          className="bg-card w-full max-w-md max-h-[80vh] overflow-y-auto rounded-t-3xl sm:rounded-3xl border border-border shadow-2xl"
        >
          <div className="sticky top-0 bg-card/95 backdrop-blur-md border-b border-border p-4 flex items-center justify-between rounded-t-3xl">
            <div className="flex items-center gap-2">
              <Flag className="w-5 h-5 text-yellow-500" />
              <h2 className="text-lg font-black">Report Content</h2>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-muted rounded-full transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>

          {result ? (
            <div className="p-6 text-center space-y-4">
              <div className={`w-16 h-16 rounded-full mx-auto flex items-center justify-center ${result.success ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
                {result.success ? (
                  <Flag className="w-8 h-8 text-green-500" />
                ) : (
                  <AlertTriangle className="w-8 h-8 text-red-500" />
                )}
              </div>
              <p className="text-sm font-medium">{result.message}</p>
              <button onClick={onClose} className="px-6 py-2 bg-muted rounded-xl text-sm font-bold">
                Close
              </button>
            </div>
          ) : (
            <div className="p-5 space-y-5">
              <div className="bg-yellow-500/5 border border-yellow-500/20 rounded-xl p-3 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-muted-foreground">
                  Only users with a trust score of 70+ can flag content. False reports may impact your trust score.
                </p>
              </div>

              {/* Flag Type Selection */}
              <div className="space-y-2">
                <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest">Type of Issue</p>
                <div className="space-y-2">
                  {FLAG_TYPES.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => setFlagType(type.value)}
                      className={`w-full text-left p-3 rounded-xl border transition-all ${
                        flagType === type.value
                          ? 'border-primary bg-primary/5'
                          : 'border-border hover:border-primary/30'
                      }`}
                    >
                      <p className="text-sm font-bold">{type.label}</p>
                      <p className="text-xs text-muted-foreground">{type.desc}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Reason */}
              <div className="space-y-2">
                <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest">Details (Optional)</p>
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="Provide additional context..."
                  className="w-full bg-muted/50 border border-border rounded-xl p-3 text-sm min-h-[80px] resize-none focus:outline-none focus:border-primary/30"
                />
              </div>

              {/* Source URL */}
              <div className="space-y-2">
                <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest">
                  <Link2 className="w-3 h-3 inline mr-1" />
                  Source URL (Optional)
                </p>
                <input
                  type="url"
                  value={sourceUrl}
                  onChange={(e) => setSourceUrl(e.target.value)}
                  placeholder="https://factcheck.org/..."
                  className="w-full bg-muted/50 border border-border rounded-xl p-3 text-sm focus:outline-none focus:border-primary/30"
                />
              </div>

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={!flagType || submitting}
                className="w-full py-3 bg-yellow-500 text-black font-bold rounded-xl disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {submitting ? (
                  <div className="w-5 h-5 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    Submit Flag
                  </>
                )}
              </button>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
