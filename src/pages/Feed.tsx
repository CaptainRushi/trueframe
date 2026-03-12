import { useState, useEffect, useCallback } from "react";
import { ShieldCheck, Loader2 } from "lucide-react";
import { PostCard } from "@/components/feed/PostCard";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";

export default function Feed() {
  const [posts, setPosts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentUserId, setCurrentUserId] = useState<string | null>(null);

  useEffect(() => {
    // Get current user once for all PostCards
    const getUser = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.user?.id) {
          setCurrentUserId(session.user.id);
        }
      } catch (e) {
        // silently fail
      }
    };
    getUser();
    fetchFeed();
  }, []);

  const fetchFeed = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/feed`);
      if (res.ok) {
        const data = await res.json();
        setPosts(data.posts || []);
      }
    } catch (e) {
      console.error("[FEED] Failed to fetch feed", e);
    } finally {
      setLoading(false);
    }
  }, []);

  const handlePostDelete = useCallback((postId: string) => {
    setPosts(prevPosts => prevPosts.filter(post => post.id !== postId));
  }, []);

  return (
    <div className="min-h-screen bg-background pb-20 md:pb-6">
      {/* Header */}
      <header className="sticky top-0 z-40 glass border-b border-border">
        <div className="w-full px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-6 h-6 text-primary" />
            <h1 className="text-xl font-black tracking-tight">Truth Feed</h1>
          </div>
        </div>
      </header>

      <main className="w-full py-4">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
            <p className="text-muted-foreground animate-pulse">Computing Truth Feed...</p>
          </div>
        ) : posts.length > 0 ? (
          <section className="space-y-6 px-4 pb-20 md:pb-6">
            {posts.map((post) => (
              <PostCard
                key={post.id}
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
                authenticityLabel={post.verification?.authenticity_label || post.authenticity_label || 'VERIFIED_REAL'}
                authorTrustScore={post.profiles?.trust_score ?? 50}
                authorTrustStatus={post.profiles?.trust_status || 'NEW_USER'}
                onDelete={handlePostDelete}
                currentUserId={currentUserId || undefined}
              />
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
