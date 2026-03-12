import { Heart, MessageCircle, Send, Bookmark, MoreHorizontal, Trash2, Eye, Flag, AlertTriangle } from "lucide-react";
import { useState, useEffect, useCallback, memo, lazy, Suspense } from "react";
import { VerifiedBadge } from "@/components/ui/VerifiedBadge";
import { TrustShield } from "@/components/ui/TrustShield";
import { AuthenticityLabel } from "@/components/ui/AuthenticityLabel";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";

// Lazy load heavy modal components - only loaded when user interacts
const TransparencyPanel = lazy(() => import("@/components/transparency/TransparencyPanel").then(m => ({ default: m.TransparencyPanel })));
const FlagModal = lazy(() => import("@/components/community/FlagModal").then(m => ({ default: m.FlagModal })));
const CommentSection = lazy(() => import("@/components/feed/CommentSection").then(m => ({ default: m.CommentSection })));
const ShareModal = lazy(() => import("@/components/share/ShareModal").then(m => ({ default: m.ShareModal })));

interface PostCardProps {
  id: string;
  userId: string;
  userAvatar: string;
  username: string;
  image: string;
  caption: string;
  likes: number;
  comments: number;
  timestamp: string;
  isVerified?: boolean;
  authenticityLabel?: string;
  authorTrustScore?: number;
  authorTrustStatus?: string;
  visibility?: string;
  onDelete?: (postId: string) => void;
  currentUserId?: string; // Pass from parent to avoid repeated auth calls
}

export const PostCard = memo(function PostCard({
  id,
  userId,
  userAvatar,
  username,
  image,
  caption,
  likes: initialLikes,
  comments: initialComments,
  timestamp,
  isVerified = true,
  authenticityLabel = "VERIFIED_REAL",
  authorTrustScore = 50,
  authorTrustStatus = "NEW_USER",
  visibility = "PUBLIC",
  onDelete,
  currentUserId,
}: PostCardProps) {
  const [isLiked, setIsLiked] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [likeCount, setLikeCount] = useState(initialLikes);
  const [commentCount, setCommentCount] = useState(initialComments);
  const [showMenu, setShowMenu] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  
  // Lazy modal states - modals only mount when first opened
  const [showShareModal, setShowShareModal] = useState(false);
  const [showTransparency, setShowTransparency] = useState(false);
  const [showFlagModal, setShowFlagModal] = useState(false);
  const [showComments, setShowComments] = useState(false);
  // Track if modals have ever been opened (keep mounted after first open for snappier re-open)
  const [mountedModals, setMountedModals] = useState({
    share: false, transparency: false, flag: false, comments: false
  });

  // Derive ownership from passed-in currentUserId instead of calling auth
  const isOwner = !!currentUserId && currentUserId === userId;

  useEffect(() => {
    checkLikeStatus();
  }, [id]);

  const checkLikeStatus = useCallback(async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/social/like/${id}/status`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setIsLiked(data.liked);
      }
    } catch (e) {
      // Silently fail
    }
  }, [id]);

  const handleLike = useCallback(async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        alert('Please log in to like posts');
        return;
      }

      // Optimistic update
      setIsLiked(prev => !prev);
      setLikeCount(prev => isLiked ? prev - 1 : prev + 1);

      const res = await fetch(`${BACKEND_URL}/api/social/like/${id}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setIsLiked(data.liked);
        setLikeCount(prev => data.liked ? initialLikes + 1 : initialLikes);
      } else {
        // Revert on failure
        setIsLiked(prev => !prev);
        setLikeCount(prev => isLiked ? prev + 1 : prev - 1);
      }
    } catch (e) {
      console.error('Failed to toggle like', e);
    }
  }, [id, isLiked, initialLikes]);

  const handleShare = useCallback(() => {
    if (!isVerified) return;
    setShowShareModal(true);
    setMountedModals(prev => ({ ...prev, share: true }));
  }, [isVerified]);

  const openTransparency = useCallback(() => {
    setShowTransparency(true);
    setMountedModals(prev => ({ ...prev, transparency: true }));
  }, []);

  const openFlag = useCallback(() => {
    setShowFlagModal(true);
    setMountedModals(prev => ({ ...prev, flag: true }));
  }, []);

  const openComments = useCallback(() => {
    setShowComments(true);
    setMountedModals(prev => ({ ...prev, comments: true }));
  }, []);

  const handleDelete = useCallback(async () => {
    if (!confirm('Are you sure you want to delete this post? This action cannot be undone.')) {
      return;
    }

    setIsDeleting(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        alert('Please log in to delete posts');
        return;
      }

      const res = await fetch(`${BACKEND_URL}/api/social/post/${id}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        alert('Post deleted successfully');
        onDelete?.(id);
      } else {
        const error = await res.json();
        alert(error.error || 'Failed to delete post');
      }
    } catch (e) {
      console.error('Failed to delete post', e);
      alert('Failed to delete post');
    } finally {
      setIsDeleting(false);
      setShowMenu(false);
    }
  }, [id, onDelete]);

  return (
    <article className="bg-card rounded-3xl shadow-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <img
              src={userAvatar}
              alt={username}
              className="w-10 h-10 rounded-full object-cover ring-2 ring-primary/20"
              loading="lazy"
              decoding="async"
            />
            {isVerified && (
              <div className="absolute -bottom-1 -right-1">
                <VerifiedBadge size="sm" timestamp={timestamp} />
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div>
              <p className="font-semibold text-foreground">{username}</p>
              <p className="text-xs text-muted-foreground">@{username.toLowerCase()}</p>
            </div>
            <TrustShield
              trustScore={authorTrustScore}
              status={authorTrustStatus}
              size="sm"
              showScore={false}
            />
            {visibility === 'UNDER_REVIEW' && (
              <span className="w-2.5 h-2.5 rounded-full bg-yellow-400 animate-pulse" title="Under Review" />
            )}
          </div>
        </div>
        {isOwner && (
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-2 hover:bg-muted rounded-full transition-colors"
            >
              <MoreHorizontal className="w-5 h-5 text-muted-foreground" />
            </button>
            {showMenu && (
              <div className="absolute right-0 top-full mt-2 bg-card border border-border rounded-lg shadow-lg overflow-hidden z-10">
                <button
                  onClick={handleDelete}
                  disabled={isDeleting}
                  className="flex items-center gap-2 px-4 py-2 text-sm text-destructive hover:bg-destructive/10 transition-colors w-full text-left disabled:opacity-50"
                >
                  <Trash2 className="w-4 h-4" />
                  {isDeleting ? 'Deleting...' : 'Delete Post'}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Image */}
      <div className="relative aspect-[4/5] bg-muted">
        <img
          src={image}
          alt="Post content"
          loading="lazy"
          decoding="async"
          className="w-full h-full object-cover"
        />
        {isVerified && visibility !== 'UNDER_REVIEW' && (
          <div className="absolute top-3 right-3">
            <AuthenticityLabel label={authenticityLabel} />
          </div>
        )}
        {visibility === 'UNDER_REVIEW' && (
          <div className="absolute inset-0 bg-yellow-500/15 backdrop-blur-[2px] flex flex-col items-center justify-center gap-2">
            <AlertTriangle className="w-10 h-10 text-yellow-400" />
            <p className="text-yellow-400 font-bold text-sm">Under Community Review</p>
            <p className="text-yellow-400/70 text-xs text-center px-8">
              This content has been reported and is under secondary AI review.
            </p>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={handleLike}
              className="flex items-center gap-1 active:scale-90 transition-transform"
            >
              <Heart
                className={`w-6 h-6 transition-colors ${isLiked ? "text-destructive fill-destructive" : "text-foreground"
                  }`}
              />
              <span className="text-sm font-medium">{likeCount.toLocaleString()}</span>
            </button>
            <button
              onClick={openComments}
              className="flex items-center gap-1 hover:text-primary transition-colors"
            >
              <MessageCircle className="w-6 h-6 text-foreground" />
              <span className="text-sm font-medium">{commentCount}</span>
            </button>
            <button onClick={handleShare} disabled={!isVerified || visibility === 'UNDER_REVIEW'} className={(!isVerified || visibility === 'UNDER_REVIEW') ? "opacity-30 cursor-not-allowed" : ""}>
              <Send className="w-6 h-6 text-foreground" />
            </button>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={openTransparency} title="View Transparency" className="active:scale-90 transition-transform">
              <Eye className="w-5 h-5 text-muted-foreground hover:text-primary transition-colors" />
            </button>
            {!isOwner && (
              <button onClick={openFlag} title="Flag Content" className="active:scale-90 transition-transform">
                <Flag className="w-5 h-5 text-muted-foreground hover:text-yellow-500 transition-colors" />
              </button>
            )}
            <button onClick={() => setIsSaved(!isSaved)} className="active:scale-90 transition-transform">
              <Bookmark
                className={`w-6 h-6 transition-colors ${isSaved ? "text-primary fill-primary" : "text-foreground"
                  }`}
              />
            </button>
          </div>
        </div>

        {/* Caption */}
        <p className="text-sm">
          <span className="font-semibold">{username}</span>{" "}
          <span className="text-foreground/90">{caption}</span>
        </p>

        <p className="text-xs text-muted-foreground uppercase">{timestamp}</p>
      </div>

      {/* Lazy-loaded modals - only mount when first opened */}
      <Suspense fallback={null}>
        {mountedModals.share && (
          <ShareModal
            isOpen={showShareModal}
            onClose={() => setShowShareModal(false)}
            post={{ id, caption, verification_status: isVerified ? 'APPROVED' : 'PENDING' }}
          />
        )}
        {mountedModals.transparency && (
          <TransparencyPanel
            postId={id}
            isOpen={showTransparency}
            onClose={() => setShowTransparency(false)}
          />
        )}
        {mountedModals.flag && (
          <FlagModal
            postId={id}
            isOpen={showFlagModal}
            onClose={() => setShowFlagModal(false)}
          />
        )}
        {mountedModals.comments && (
          <CommentSection
            postId={id}
            isOpen={showComments}
            onClose={() => setShowComments(false)}
            onCommentAdded={() => setCommentCount(prev => prev + 1)}
          />
        )}
      </Suspense>
    </article>
  );
});
