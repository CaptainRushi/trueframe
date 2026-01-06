import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, TrendingUp, ShieldCheck, User, Grid, Play } from "lucide-react";
import { VerifiedBadge } from "@/components/ui/VerifiedBadge";
import { supabase } from "@/lib/supabase";
import { Link } from "react-router-dom";

import { BACKEND_URL } from "@/lib/api";

interface Profile {
  id: string;
  username: string;
  displayName: string | null;
  avatarUrl: string | null;
  trustStatus: string;
  realPercentage: number;
}

interface Post {
  id: string;
  media_url: string;
  media_type: 'image' | 'video';
  caption: string;
  profiles?: any; // Nested from Supabase
}

export default function Explore() {
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [exploreFeed, setExploreFeed] = useState<Post[]>([]);
  const [searchUsers, setSearchUsers] = useState<Profile[]>([]);
  const [searchPosts, setSearchPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'posts' | 'users'>('posts');

  // Debounce
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 500);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Fetch Explore Feed (Initial)
  useEffect(() => {
    if (!debouncedQuery) {
      fetchExploreFeed();
    } else {
      handleSearch();
    }
  }, [debouncedQuery]);

  const fetchExploreFeed = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/explore`);
      if (res.ok) {
        const data = await res.json();
        setExploreFeed(data);
      }
    } catch (e) {
      console.error("Explore fetch failed", e);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async () => {
    if (debouncedQuery.length < 2) return;
    setLoading(true);
    try {
      // Parallel fetch
      const [usersRes, postsRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/search/users?q=${encodeURIComponent(debouncedQuery)}`),
        fetch(`${BACKEND_URL}/api/search/posts?q=${encodeURIComponent(debouncedQuery)}`)
      ]);

      if (usersRes.ok) {
        const users = await usersRes.json();
        // Map to Profile interface
        const mappedUsers = users.map((u: any) => ({
          id: u.id,
          username: u.username,
          displayName: u.display_name,
          avatarUrl: u.avatar_url,
          trustStatus: u.trust_status,
          realPercentage: u.real_percentage
        }));
        setSearchUsers(mappedUsers);
      }

      if (postsRes.ok) {
        const posts = await postsRes.json();
        setSearchPosts(posts);
      }
    } catch (e) {
      console.error("Search failed", e);
    } finally {
      setLoading(false);
    }
  };

  const isSearching = debouncedQuery.length >= 2;

  return (
    <div className="min-h-screen bg-background pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-background/80 backdrop-blur-md border-b border-border p-4">
        <div className="max-w-2xl mx-auto space-y-4">
          <div className="flex items-center gap-2 mb-2">
            {!isSearching ? <TrendingUp className="w-5 h-5 text-primary" /> : <Search className="w-5 h-5 text-primary" />}
            <h1 className="text-xl font-bold tracking-tight">
              {isSearching ? "Search Results" : "Explore"}
            </h1>
          </div>

          {/* Search Bar */}
          <div className="relative group">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground group-focus-within:text-primary transition-colors" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search verified users, posts, or tags..."
              className="w-full pl-12 pr-4 py-3 bg-muted/50 border border-transparent focus:border-primary/20 rounded-2xl text-foreground placeholder:text-muted-foreground focus:outline-none focus:bg-background transition-all"
            />
          </div>

          {/* Trust Indicator */}
          <div className="flex items-center gap-2 px-1">
            <ShieldCheck className="w-4 h-4 text-green-500" />
            <p className="text-xs text-muted-foreground font-medium">
              Only <span className="text-green-500 font-bold">Verified & Trusted</span> content
            </p>
          </div>

          {/* Search Tabs */}
          {isSearching && (
            <div className="flex gap-2 border-b border-border">
              <button
                onClick={() => setActiveTab('posts')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'posts' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground'}`}
              >
                Verified Posts ({searchPosts.length})
              </button>
              <button
                onClick={() => setActiveTab('users')}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${activeTab === 'users' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground'}`}
              >
                Trusted Users ({searchUsers.length})
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-2xl mx-auto p-4">
        {/* EXPLORE FEED */}
        {!isSearching && (
          <>
            {loading && exploreFeed.length === 0 ? (
              <div className="text-center py-20 text-muted-foreground">Loading trusted content...</div>
            ) : exploreFeed.length > 0 ? (
              <div className="grid grid-cols-3 gap-1 md:gap-4">
                {exploreFeed.map((post, i) => (
                  <Link key={post.id} to={`/profile/${post.profiles?.username}`} className="block relative aspect-square bg-muted rounded-xl overflow-hidden group">
                    {post.media_type === 'video' ? (
                      <video src={post.media_url} className="w-full h-full object-cover" />
                    ) : (
                      <img src={post.media_url} alt={post.caption} className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" />
                    )}

                    {/* Overlay */}
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100">
                      <div className="absolute top-2 right-2">
                        <VerifiedBadge size="sm" />
                      </div>
                      {post.media_type === 'video' && <Play className="w-8 h-8 text-white fill-white" />}
                    </div>
                  </Link>
                ))}
              </div>
            ) : (
              <div className="text-center py-20">
                <ShieldCheck className="w-12 h-12 text-muted-foreground mx-auto mb-4 opacity-50" />
                <h3 className="text-lg font-semibold">No verified content available right now.</h3>
                <p className="text-muted-foreground text-sm">Check back later for trusted uploads.</p>
              </div>
            )}
          </>
        )}

        {/* SEARCH RESULTS */}
        {isSearching && (
          <>
            {activeTab === 'posts' && (
              searchPosts.length > 0 ? (
                <div className="grid grid-cols-3 gap-1 md:gap-4">
                  {searchPosts.map((post) => (
                    <Link key={post.id} to={`/profile/${post.profiles?.username}`} className="block relative aspect-square bg-muted rounded-xl overflow-hidden group">
                      {post.media_type === 'video' ? (
                        <video src={post.media_url} className="w-full h-full object-cover" />
                      ) : (
                        <img src={post.media_url} alt={post.caption} className="w-full h-full object-cover" />
                      )}
                      <div className="absolute top-2 right-2">
                        <VerifiedBadge size="sm" />
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="text-center py-20 text-muted-foreground">
                  No verified posts found for "{debouncedQuery}"
                </div>
              )
            )}

            {activeTab === 'users' && (
              searchUsers.length > 0 ? (
                <div className="space-y-4">
                  {searchUsers.map((user) => (
                    <Link key={user.id} to={`/profile/${user.username}`} className="flex items-center gap-4 p-4 bg-card border border-border rounded-2xl hover:bg-muted/50 transition-colors">
                      <div className="w-12 h-12 rounded-full bg-muted overflow-hidden flex-shrink-0">
                        {user.avatarUrl ? (
                          <img src={user.avatarUrl} className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-lg font-bold bg-primary/10 text-primary">
                            {user.username[0].toUpperCase()}
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h3 className="font-bold truncate">{user.displayName || user.username}</h3>
                          <VerifiedBadge size="sm" />
                        </div>
                        <p className="text-sm text-muted-foreground truncate">@{user.username}</p>
                      </div>
                      <div className={`px-2 py-1 rounded-full text-xs font-bold ${user.trustStatus === 'TRUSTED' ? 'bg-green-500/10 text-green-600' : 'bg-yellow-500/10 text-yellow-600'
                        }`}>
                        {user.trustStatus}
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="text-center py-20 text-muted-foreground">
                  No trusted users found for "{debouncedQuery}"
                </div>
              )
            )}
          </>
        )}
      </main>
    </div>
  );
}
