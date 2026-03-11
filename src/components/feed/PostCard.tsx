import { Heart, MessageCircle, Send, Bookmark, MoreHorizontal, Trash2, Eye, Flag, AlertTriangle } from "lucide-react";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { VerifiedBadge } from "@/components/ui/VerifiedBadge";
import { TrustShield } from "@/components/ui/TrustShield";
import { AuthenticityLabel } from "@/components/ui/AuthenticityLabel";
import { supabase } from "@/lib/supabase";
import { TransparencyPanel } from "@/components/transparency/TransparencyPanel";
import { FlagModal } from "@/components/community/FlagModal";

import { ShareModal } from "@/components/share/ShareModal";
import { BACKEND_URL } from "@/lib/api";

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
}

export function PostCard({
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
}: PostCardProps) {
  const [isLiked, setIsLiked] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [likeCount, setLikeCount] = useState(initialLikes);
  const [commentCount, setCommentCount] = useState(initialComments);
  const [isOwner, setIsOwner] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showTransparency, setShowTransparency] = useState(false);
  const [showFlagModal, setShowFlagModal] = useState(false);

  useEffect(() => {
    checkLikeStatus();
    checkOwnership();
  }, [id, userId]); // Added userId to dependency array for checkOwnership

  const checkOwnership = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (user && user.id === userId) {
        setIsOwner(true);
      }
    } catch (e) {
      console.error('Failed to check ownership', e);
    }
  };

  const checkLikeStatus = async () => {
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
      console.error('Failed to check like status', e);
    }
  };

  const handleLike = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        alert('Please log in to like posts');
        return;
      }

      const res = await fetch(`${BACKEND_URL}/api/social/like/${id}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setIsLiked(data.liked);
        setLikeCount(prev => data.liked ? prev + 1 : prev - 1);
      }
    } catch (e) {
      console.error('Failed to toggle like', e);
    }
  };

  const handleShare = () => {
    if (!isVerified) return; // Prevent sharing unverified content
    setShowShareModal(true);
  };

  const handleDelete = async () => {
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
        if (onDelete) {
          onDelete(id);
        }
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
  };

  return (
    <motion.article
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-card rounded-3xl shadow-card overflow-hidden"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <img
              src={userAvatar}
              alt={username}
              className="w-10 h-10 rounded-full object-cover ring-2 ring-primary/20"
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
            <motion.button
              whileTap={{ scale: 0.8 }}
              onClick={handleLike}
              className="flex items-center gap-1"
            >
              <Heart
                className={`w-6 h-6 transition-colors ${isLiked ? "text-destructive fill-destructive" : "text-foreground"
                  }`}
              />
              <span className="text-sm font-medium">{likeCount.toLocaleString()}</span>
            </motion.button>
            <button className="flex items-center gap-1">
              <MessageCircle className="w-6 h-6 text-foreground" />
              <span className="text-sm font-medium">{commentCount}</span>
            </button>
            <button onClick={handleShare} disabled={!isVerified || visibility === 'UNDER_REVIEW'} className={(!isVerified || visibility === 'UNDER_REVIEW') ? "opacity-30 cursor-not-allowed" : ""}>
              <Send className="w-6 h-6 text-foreground" />
            </button>
          </div>
          <div className="flex items-center gap-3">
            <motion.button whileTap={{ scale: 0.8 }} onClick={() => setShowTransparency(true)} title="View Transparency">
              <Eye className="w-5 h-5 text-muted-foreground hover:text-primary transition-colors" />
            </motion.button>
            {!isOwner && (
              <motion.button whileTap={{ scale: 0.8 }} onClick={() => setShowFlagModal(true)} title="Flag Content">
                <Flag className="w-5 h-5 text-muted-foreground hover:text-yellow-500 transition-colors" />
              </motion.button>
            )}
            <motion.button whileTap={{ scale: 0.8 }} onClick={() => setIsSaved(!isSaved)}>
              <Bookmark
                className={`w-6 h-6 transition-colors ${isSaved ? "text-primary fill-primary" : "text-foreground"
                  }`}
              />
            </motion.button>
          </div>
        </div>

        {/* Caption */}
        <p className="text-sm">
          <span className="font-semibold">{username}</span>{" "}
          <span className="text-foreground/90">{caption}</span>
        </p>

        <p className="text-xs text-muted-foreground uppercase">{timestamp}</p>
      </div>
      <ShareModal
        isOpen={showShareModal}
        onClose={() => setShowShareModal(false)}
        post={{ id, caption, verification_status: isVerified ? 'APPROVED' : 'PENDING' }}
      />
      <TransparencyPanel
        postId={id}
        isOpen={showTransparency}
        onClose={() => setShowTransparency(false)}
      />
      <FlagModal
        postId={id}
        isOpen={showFlagModal}
        onClose={() => setShowFlagModal(false)}
      />
    </motion.article>
  );
}
