import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { ShieldCheck, AlertTriangle, CheckCircle2, XCircle, Eye } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";

interface SecondaryReview {
  id: string;
  post_id: string;
  trigger_flag_count: number;
  secondary_score: number;
  decision: string;
  decision_reason: string;
  signals: string[];
  created_at: string;
  completed_at: string;
  flagCount: number;
  posts: {
    id: string;
    media_url: string;
    media_type: string;
    caption: string;
    user_id: string;
    created_at: string;
    profiles: {
      username: string;
      display_name: string;
      avatar_url: string;
      trust_score: number;
      trust_status: string;
    };
  };
}

export function ModerationQueue() {
  const [reviews, setReviews] = useState<SecondaryReview[]>([]);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState<string | null>(null);

  useEffect(() => {
    fetchPendingReviews();
  }, []);

  const fetchPendingReviews = async () => {
    setLoading(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/moderation/pending`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setReviews(data.reviews || []);
      }
    } catch (e) {
      console.error('Failed to fetch moderation queue', e);
    } finally {
      setLoading(false);
    }
  };

  const handleDecision = async (reviewId: string, decision: 'RESTORE' | 'REMOVE') => {
    const actionLabel = decision === 'RESTORE' ? 'restore' : 'remove';
    if (!confirm(`Are you sure you want to ${actionLabel} this post? This action cannot be undone.`)) {
      return;
    }

    setProcessing(reviewId);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/moderation/review/${reviewId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ decision })
      });

      if (res.ok) {
        setReviews(prev => prev.filter(r => r.id !== reviewId));
      } else {
        const err = await res.json();
        alert(err.error || `Failed to ${actionLabel} post`);
      }
    } catch (e) {
      console.error(`Failed to ${actionLabel} post`, e);
      alert(`Failed to ${actionLabel} post`);
    } finally {
      setProcessing(null);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4">
        <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full" />
        <p className="text-muted-foreground text-sm">Loading moderation queue...</p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <ShieldCheck className="w-6 h-6 text-primary" />
        <div>
          <h1 className="text-xl font-black">Moderation Queue</h1>
          <p className="text-sm text-muted-foreground">
            Posts requiring your manual review decision
          </p>
        </div>
      </div>

      {reviews.length === 0 ? (
        <div className="bg-muted/30 rounded-2xl p-8 text-center space-y-3">
          <CheckCircle2 className="w-10 h-10 text-green-500 mx-auto" />
          <p className="font-bold">Queue is clear</p>
          <p className="text-sm text-muted-foreground">No posts need manual review right now.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {reviews.map((review) => (
            <motion.div
              key={review.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-card border border-border rounded-2xl overflow-hidden"
            >
              {/* Post Preview */}
              <div className="flex gap-4 p-4">
                <div className="w-24 h-24 rounded-xl overflow-hidden flex-shrink-0 bg-muted">
                  <img
                    src={review.posts?.media_url}
                    alt="Post under review"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1 min-w-0 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <img
                      src={review.posts?.profiles?.avatar_url || ''}
                      alt=""
                      className="w-5 h-5 rounded-full"
                    />
                    <span className="text-sm font-bold truncate">
                      @{review.posts?.profiles?.username}
                    </span>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                      review.posts?.profiles?.trust_status === 'TRUSTED' ? 'bg-green-500/10 text-green-500' :
                      review.posts?.profiles?.trust_status === 'AT_RISK' ? 'bg-yellow-500/10 text-yellow-500' :
                      'bg-red-500/10 text-red-500'
                    }`}>
                      Trust: {review.posts?.profiles?.trust_score}
                    </span>
                  </div>
                  {review.posts?.caption && (
                    <p className="text-xs text-muted-foreground line-clamp-2">{review.posts.caption}</p>
                  )}
                  <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                    <span>Flags: {review.flagCount}</span>
                    <span>AI Score: {Math.round(review.secondary_score * 100)}%</span>
                    <span>{new Date(review.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>

              {/* Signals */}
              {review.signals && review.signals.length > 0 && (
                <div className="px-4 pb-2">
                  <div className="flex flex-wrap gap-1.5">
                    {review.signals.slice(0, 5).map((signal, i) => (
                      <span key={i} className="text-[10px] bg-yellow-500/10 text-yellow-500 px-2 py-0.5 rounded-full">
                        {signal.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex border-t border-border">
                <button
                  onClick={() => handleDecision(review.id, 'RESTORE')}
                  disabled={processing === review.id}
                  className="flex-1 flex items-center justify-center gap-2 py-3 text-sm font-bold text-green-500 hover:bg-green-500/5 transition-colors disabled:opacity-50"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  Restore
                </button>
                <div className="w-px bg-border" />
                <button
                  onClick={() => handleDecision(review.id, 'REMOVE')}
                  disabled={processing === review.id}
                  className="flex-1 flex items-center justify-center gap-2 py-3 text-sm font-bold text-red-500 hover:bg-red-500/5 transition-colors disabled:opacity-50"
                >
                  <XCircle className="w-4 h-4" />
                  Remove
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
