import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Send, MessageCircle, Loader2, ShieldCheck, AlertTriangle } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";

interface Comment {
  id: string;
  content: string;
  type: string;
  visibility: string;
  created_at: string;
  profiles: {
    username: string;
    display_name: string | null;
    avatar_url: string | null;
    trust_status: string;
  };
}

interface CommentSectionProps {
  postId: string;
  isOpen: boolean;
  onClose: () => void;
  onCommentAdded?: () => void;
}

export function CommentSection({ postId, isOpen, onClose, onCommentAdded }: CommentSectionProps) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [newComment, setNewComment] = useState("");
  const [loading, setLoading] = useState(false);
  const [posting, setPosting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen) {
      fetchComments();
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [isOpen, postId]);

  const fetchComments = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${BACKEND_URL}/api/social/comments/${postId}`);
      if (res.ok) {
        const data = await res.json();
        setComments(data.comments || []);
      }
    } catch (e) {
      console.error("Failed to fetch comments", e);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!newComment.trim() || posting) return;

    setPosting(true);
    setError(null);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        setError("Please log in to comment");
        return;
      }

      const res = await fetch(`${BACKEND_URL}/api/social/comment/${postId}`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${session.access_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ content: newComment.trim() }),
      });

      if (res.ok) {
        setNewComment("");
        fetchComments();
        onCommentAdded?.();
        // Scroll to top to see newest comment
        setTimeout(() => scrollRef.current?.scrollTo({ top: 0, behavior: "smooth" }), 200);
      } else {
        const data = await res.json();
        setError(data.error || data.reason || "Failed to post comment");
      }
    } catch (e: any) {
      setError(e.message || "Failed to post comment");
    } finally {
      setPosting(false);
    }
  };

  const getTimeAgo = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "now";
    if (mins < 60) return `${mins}m`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d`;
    return `${Math.floor(days / 7)}w`;
  };

  const getTrustBadge = (status: string) => {
    if (status === "TRUSTED") return "text-green-500";
    if (status === "AT_RISK" || status === "WARNING") return "text-yellow-500";
    if (status === "UNDER_REVIEW") return "text-red-500";
    return "text-blue-500";
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-end md:items-center justify-center"
          onClick={onClose}
        >
          <motion.div
            initial={{ y: "100%", opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: "100%", opacity: 0 }}
            transition={{ type: "spring", damping: 28, stiffness: 300 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full md:max-w-lg bg-card rounded-t-3xl md:rounded-3xl border border-border shadow-2xl max-h-[85vh] md:max-h-[70vh] flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-border flex-shrink-0">
              <div className="flex items-center gap-2">
                <MessageCircle className="w-5 h-5 text-primary" />
                <h3 className="font-bold text-lg">Comments</h3>
                <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                  {comments.length}
                </span>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-muted rounded-full transition-colors"
              >
                <X className="w-5 h-5 text-muted-foreground" />
              </button>
            </div>

            {/* Drag handle for mobile */}
            <div className="flex justify-center py-1 md:hidden flex-shrink-0">
              <div className="w-10 h-1 bg-muted-foreground/20 rounded-full" />
            </div>

            {/* Comments List */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-6 h-6 text-primary animate-spin" />
                </div>
              ) : comments.length === 0 ? (
                <div className="text-center py-12">
                  <MessageCircle className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground font-medium">No comments yet</p>
                  <p className="text-xs text-muted-foreground mt-1">Be the first to share your thoughts</p>
                </div>
              ) : (
                comments.map((comment) => (
                  <motion.div
                    key={comment.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-3 ${comment.visibility === "COLLAPSED" ? "opacity-50" : ""}`}
                  >
                    {/* Avatar */}
                    <div className="flex-shrink-0">
                      {comment.profiles?.avatar_url ? (
                        <img
                          src={comment.profiles.avatar_url}
                          alt={comment.profiles.username}
                          className="w-8 h-8 rounded-full object-cover"
                        />
                      ) : (
                        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold text-sm">
                          {(comment.profiles?.username?.[0] || "?").toUpperCase()}
                        </div>
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-semibold text-sm">
                          {comment.profiles?.display_name || comment.profiles?.username || "User"}
                        </span>
                        <ShieldCheck className={`w-3 h-3 ${getTrustBadge(comment.profiles?.trust_status)}`} />
                        <span className="text-xs text-muted-foreground">
                          {getTimeAgo(comment.created_at)}
                        </span>
                        {comment.type === "CORRECTION" && (
                          <span className="text-[10px] bg-yellow-500/10 text-yellow-600 px-1.5 py-0.5 rounded-full font-bold">
                            CORRECTION
                          </span>
                        )}
                        {comment.type === "CLAIM" && (
                          <span className="text-[10px] bg-blue-500/10 text-blue-600 px-1.5 py-0.5 rounded-full font-bold">
                            CLAIM
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-foreground/90 mt-0.5 break-words">{comment.content}</p>
                      {comment.visibility === "COLLAPSED" && (
                        <p className="text-[10px] text-yellow-600 mt-1 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3" />
                          Low-trust claim — requires verification
                        </p>
                      )}
                    </div>
                  </motion.div>
                ))
              )}
            </div>

            {/* Error banner */}
            {error && (
              <div className="px-4 py-2 bg-destructive/10 border-t border-destructive/20 flex-shrink-0">
                <p className="text-xs text-destructive font-medium">{error}</p>
              </div>
            )}

            {/* Input */}
            <div className="p-3 border-t border-border bg-card flex-shrink-0">
              <div className="flex items-center gap-2">
                <input
                  ref={inputRef}
                  type="text"
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                  placeholder="Add a verified comment..."
                  className="flex-1 px-4 py-2.5 bg-muted/50 rounded-full text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 transition-all"
                  disabled={posting}
                />
                <button
                  onClick={handleSubmit}
                  disabled={!newComment.trim() || posting}
                  className="p-2.5 bg-primary text-primary-foreground rounded-full disabled:opacity-40 hover:opacity-90 transition-all flex-shrink-0"
                >
                  {posting ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </button>
              </div>
              <p className="text-[10px] text-muted-foreground text-center mt-2">
                Comments are trust-verified • Claims from unverified users may be collapsed
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
