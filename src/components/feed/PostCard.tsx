import { Heart, MessageCircle, Send, Bookmark, MoreHorizontal } from "lucide-react";
import { motion } from "framer-motion";
import { useState } from "react";
import { VerifiedBadge } from "@/components/ui/VerifiedBadge";

interface PostCardProps {
  id: string;
  userAvatar: string;
  username: string;
  image: string;
  caption: string;
  likes: number;
  comments: number;
  timestamp: string;
  isVerified?: boolean;
}

export function PostCard({
  userAvatar,
  username,
  image,
  caption,
  likes,
  comments,
  timestamp,
  isVerified = true,
}: PostCardProps) {
  const [isLiked, setIsLiked] = useState(false);
  const [isSaved, setIsSaved] = useState(false);
  const [likeCount, setLikeCount] = useState(likes);

  const handleLike = () => {
    setIsLiked(!isLiked);
    setLikeCount(isLiked ? likeCount - 1 : likeCount + 1);
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
          <div>
            <p className="font-semibold text-foreground">{username}</p>
            <p className="text-xs text-muted-foreground">@{username.toLowerCase()}</p>
          </div>
        </div>
        <button className="p-2 hover:bg-muted rounded-full transition-colors">
          <MoreHorizontal className="w-5 h-5 text-muted-foreground" />
        </button>
      </div>

      {/* Image */}
      <div className="relative aspect-[4/5] bg-muted">
        <img
          src={image}
          alt="Post content"
          className="w-full h-full object-cover"
        />
        {isVerified && (
          <div className="absolute top-3 right-3 flex items-center gap-2 bg-verified/90 backdrop-blur-sm px-3 py-1.5 rounded-full">
            <VerifiedBadge size="sm" showTooltip={false} />
            <span className="text-xs font-medium text-verified-foreground">Verified Real</span>
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
                className={`w-6 h-6 transition-colors ${
                  isLiked ? "text-destructive fill-destructive" : "text-foreground"
                }`}
              />
              <span className="text-sm font-medium">{likeCount.toLocaleString()}</span>
            </motion.button>
            <button className="flex items-center gap-1">
              <MessageCircle className="w-6 h-6 text-foreground" />
              <span className="text-sm font-medium">{comments}</span>
            </button>
            <button>
              <Send className="w-6 h-6 text-foreground" />
            </button>
          </div>
          <motion.button whileTap={{ scale: 0.8 }} onClick={() => setIsSaved(!isSaved)}>
            <Bookmark
              className={`w-6 h-6 transition-colors ${
                isSaved ? "text-primary fill-primary" : "text-foreground"
              }`}
            />
          </motion.button>
        </div>

        {/* Caption */}
        <p className="text-sm">
          <span className="font-semibold">{username}</span>{" "}
          <span className="text-foreground/90">{caption}</span>
        </p>

        <p className="text-xs text-muted-foreground uppercase">{timestamp}</p>
      </div>
    </motion.article>
  );
}
