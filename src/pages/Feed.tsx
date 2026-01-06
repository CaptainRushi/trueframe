import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Bell, MessageSquare, ShieldCheck, Loader2 } from "lucide-react";
import { PostCard } from "@/components/feed/PostCard";

import { BACKEND_URL } from "@/lib/api";

export default function Feed() {
  const [posts, setPosts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFeed();
  }, []);

  const fetchFeed = async () => {
    try {
      console.log('[FEED] Fetching from:', `${BACKEND_URL}/api/feed`);
      const res = await fetch(`${BACKEND_URL}/api/feed`);
      console.log('[FEED] Response status:', res.status, res.statusText);

      if (res.ok) {
        const data = await res.json();
        console.log('[FEED] Data received:', data);
        console.log('[FEED] Posts count:', data.posts?.length || 0);
        setPosts(data.posts || []);
      } else {
        const errorText = await res.text();
        console.error('[FEED] Error response:', errorText);
      }
    } catch (e) {
      console.error("[FEED] Failed to fetch feed", e);
    } finally {
      setLoading(false);
    }
  };

  const handlePostDelete = (postId: string) => {
    setPosts(prevPosts => prevPosts.filter(post => post.id !== postId));
  };

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass border-b border-border">
        <div className="max-w-lg mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-primary" />
            <h1 className="text-xl font-black tracking-tight">Truth Feed</h1>
          </div>
          <div className="flex items-center gap-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="p-2 hover:bg-muted rounded-full transition-colors relative"
            >
              <Bell className="w-6 h-6 text-foreground" />
              <span className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full border-2 border-background" />
            </motion.button>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto py-4">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
            <p className="text-muted-foreground animate-pulse">Computing Truth Feed...</p>
          </div>
        ) : posts.length > 0 ? (
          <section className="space-y-6 px-4 pb-20">
            {posts.map((post, index) => (
              <motion.div
                key={post.id}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <PostCard
                  id={post.id}
                  userId={post.user_id}
                  userAvatar={post.profiles?.avatar_url || `https://api.dicebear.com/7.x/avataaars/svg?seed=${post.profiles?.username || 'deleted'}`}
                  username={post.profiles?.display_name || post.profiles?.username || "Deleted User"}
                  image={post.media_url}
                  caption={post.caption || ""}
                  likes={post.like_count || 0}
                  comments={post.comment_count || 0}
                  timestamp={new Date(post.created_at).toLocaleDateString()}
                  isVerified={post.verification?.verdict === 'REAL'}
                  onDelete={handlePostDelete}
                />
              </motion.div>
            ))}
          </section>
        ) : (
          <div className="text-center py-20 px-8">
            <div className="bg-primary/5 w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6">
              <ShieldCheck className="w-10 h-10 text-primary/40" />
            </div>
            <h2 className="text-xl font-bold mb-2">No Verified Content Yet</h2>
            <p className="text-muted-foreground max-w-xs mx-auto">
              Be the first to upload a verified authentic piece of media to the platform.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
