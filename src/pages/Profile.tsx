import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Upload,
  Grid,
  Video,
  Camera,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Lock,
  LogOut,
  Trash2,
  UserPlus,
  UserCheck,
  Users,
  Heart,
  MessageCircle
} from "lucide-react";
import { supabase } from "@/lib/supabase";

import { BACKEND_URL } from "@/lib/api";

interface ProfileData {
  id: string;
  username: string;
  displayName: string | null;
  avatarUrl: string | null;
  bio: string | null;
  totalUploads: number;
  realUploads: number;
  fakeUploads: number;
  realPercentage: number;
  fakePercentage: number;
  total_attempts?: number;
  real_count?: number;
  fake_count?: number;
  status: 'TRUSTED' | 'AT_RISK' | 'RESTRICTED' | 'WARNING' | 'NEW_USER';
  posts: any[];
  reels: any[];
  followersCount: number;
  followingCount: number;
  isFollowing: boolean;
}

interface HistoryItem {
  id: string;
  created_at: string; // From verification_logs schema
  verdict: string;
  score: number;
  reason: string | null;
  media_type?: string;
}

const AVATAR_SIZE_LIMIT = 300 * 1024; // 300KB

export default function Profile() {
  const { username: paramUsername } = useParams();
  const navigate = useNavigate();
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOwner, setIsOwner] = useState(false);
  const [activeTab, setActiveTab] = useState<'posts' | 'reels'>('posts');
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({ displayName: '', bio: '' });
  const [savingProfile, setSavingProfile] = useState(false);
  const [selectedPost, setSelectedPost] = useState<any>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deletePhrase, setDeletePhrase] = useState("");
  const [isDeletingAccount, setIsDeletingAccount] = useState(false);

  useEffect(() => {
    initProfile();
  }, [paramUsername]);

  const initProfile = async () => {
    try {
      setLoading(true);
      setError(null);

      // Get Current User
      if (!supabase) {
        throw new Error("Supabase configuration missing.");
      }
      const { data: { session } } = await supabase.auth.getSession();
      const currentUser = session?.user;

      let targetUsername = paramUsername;

      // If no param, try to get my own profile or init it
      if (!targetUsername && currentUser) {
        const res = await fetch(`${BACKEND_URL}/api/profile/init`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${session.access_token}` }
        });

        if (res.status === 401) {
          // Session invalid, treat as guest
          targetUsername = undefined;
        } else if (res.ok) {
          const myProfile = await res.json();
          targetUsername = myProfile.username;
          if (targetUsername) {
            navigate(`/profile/${targetUsername}`, { replace: true });
            return;
          }
        }
      }

      if (!targetUsername) {
        setError("User not found");
        setLoading(false);
        return;
      }

      // Fetch Profile Data
      const headers: any = {};
      if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`;
      }

      const response = await fetch(`${BACKEND_URL}/api/profile/${targetUsername}`, { headers });

      if (!response.ok) {
        throw new Error("Failed to load profile");
      }
      const data = await response.json();
      setProfile(data);
      setEditData({ displayName: data.displayName || '', bio: data.bio || '' });

      // Check Ownership (Robust ID comparison)
      if (currentUser && data.id === currentUser.id) {
        setIsOwner(true);
        fetchHistory(session.access_token);
      } else {
        setIsOwner(false);
      }

    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const fetchHistory = async (token: string) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/profile/me/history`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setHistory(data);
      }
    } catch (e) {
      console.error("Failed to fetch history", e);
    }
  };

  const handleAvatarUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > AVATAR_SIZE_LIMIT) {
      alert("File size exceeds 300KB limit.");
      return;
    }

    try {
      setUploadingAvatar(true);
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${BACKEND_URL}/api/profile/avatar`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${session.access_token}` },
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Upload failed");
      }

      const { avatarUrl } = await res.json();
      setProfile(prev => prev ? ({ ...prev, avatarUrl }) : null);

    } catch (err: any) {
      alert(err.message);
    } finally {
      setUploadingAvatar(false);
    }
  };

  const handleUpdateProfile = async () => {
    try {
      setSavingProfile(true);
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/profile/update`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(editData)
      });

      if (!res.ok) throw new Error("Update failed");

      const updated = await res.json();
      setProfile(prev => prev ? ({
        ...prev,
        displayName: updated.display_name,
        bio: updated.bio
      }) : null);
      setIsEditing(false);

    } catch (err: any) {
      alert(err.message);
    } finally {
      setSavingProfile(false);
    }
  };

  const handlePostDelete = async (postId: string) => {
    if (!confirm('Are you sure you want to delete this post? This action cannot be undone.')) return;

    try {
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        alert("Please log in to delete posts");
        return;
      }

      const res = await fetch(`${BACKEND_URL}/api/social/post/${postId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        setProfile(prev => {
          if (!prev) return null;
          return {
            ...prev,
            posts: prev.posts.filter(p => p.id !== postId),
            reels: prev.reels.filter(p => p.id !== postId),
            totalUploads: Math.max(0, prev.totalUploads - 1),
            realUploads: Math.max(0, prev.realUploads - 1)
          };
        });
        alert("Post deleted successfully");
      } else {
        const err = await res.json();
        alert(err.error || "Failed to delete post");
      }
    } catch (e) {
      console.error(e);
      alert("Failed to delete post");
    }
  };

  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'TRUSTED':
        return { icon: ShieldCheck, color: 'text-green-500', bg: 'bg-green-500/10', label: 'Trusted Profile' };
      case 'AT_RISK':
      case 'WARNING':
        return { icon: ShieldAlert, color: 'text-yellow-500', bg: 'bg-yellow-500/10', label: 'At Risk' };
      case 'RESTRICTED':
        return { icon: ShieldX, color: 'text-red-500', bg: 'bg-red-500/10', label: 'Restricted' };
      case 'NEW_USER':
        return { icon: Loader2, color: 'text-blue-500', bg: 'bg-blue-500/10', label: 'New Creator' };
      default:
        return { icon: ShieldCheck, color: 'text-gray-500', bg: 'bg-gray-500/10', label: status || 'Member' };
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] text-center p-4">
        <XCircle className="w-12 h-12 text-destructive mb-4" />
        <h2 className="text-xl font-bold mb-2">Profile Not Found</h2>
        <p className="text-muted-foreground">{error || "Could not load user profile"}</p>
      </div>
    );
  }

  const handleLogout = async () => {
    try {
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        await fetch(`${BACKEND_URL}/api/account/signout`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${session.access_token}` }
        }).catch(e => console.warn('Backend signout warning', e));
      }
      await supabase.auth.signOut().catch(e => console.warn('Supabase signout warning', e));
      window.location.href = "/";
    } catch (e) {
      console.error(e);
      window.location.href = "/";
    }
  };

  const handleDeleteAccount = async () => {
    if (deletePhrase !== "DELETE MY ACCOUNT") return;

    setIsDeletingAccount(true);
    try {
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/account`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        await supabase.auth.signOut();
        navigate('/', { replace: true });
        alert("Your account has been permanently deleted.");
      } else {
        const data = await res.json();
        alert(data.error || "Failed to delete account");
      }
    } catch (e) {
      console.error('Delete account error', e);
      alert("Failed to delete account");
    } finally {
      setIsDeletingAccount(false);
    }
  };

  const handleFollowToggle = async () => {
    if (!profile) return;

    try {
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();

      if (!session) {
        alert("Please log in to follow users");
        return;
      }

      // Optimistic update
      const newStatus = !profile.isFollowing;
      setProfile(prev => prev ? ({
        ...prev,
        isFollowing: newStatus,
        followersCount: newStatus ? prev.followersCount + 1 : prev.followersCount - 1
      }) : null);

      const method = profile.isFollowing ? 'DELETE' : 'POST';
      const res = await fetch(`${BACKEND_URL}/api/profile/${profile.username}/follow`, {
        method,
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (!res.ok) {
        throw new Error("Action failed");
      }
    } catch (e) {
      console.error(e);
      // Revert on error
      setProfile(prev => prev ? ({
        ...prev,
        isFollowing: !prev.isFollowing,
        followersCount: !prev.isFollowing ? prev.followersCount + 1 : prev.followersCount - 1
      }) : null);
      alert("Failed to update follow status");
    }
  };

  const statusConfig = getStatusConfig(profile.status);

  return (
    <div className="max-w-4xl mx-auto pb-20 p-4">

      {/* 1. USER IDENTITY SECTION (TOP) */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative bg-card rounded-3xl border border-border p-8 mb-8 overflow-hidden"
      >
        <div className="flex flex-col md:flex-row items-center gap-8 z-10 relative">
          {/* Avatar */}
          <div className="relative group">
            <div className="w-32 h-32 rounded-full border-4 border-background shadow-xl overflow-hidden bg-muted">
              {profile.avatarUrl ? (
                <img src={profile.avatarUrl} alt={profile.username} className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-primary/10 text-primary font-bold text-4xl">
                  {profile.username.charAt(0).toUpperCase()}
                </div>
              )}
            </div>
            {isOwner && (
              <label className="absolute bottom-0 right-0 p-2 bg-primary text-primary-foreground rounded-full cursor-pointer hover:bg-primary/90 transition-colors shadow-lg">
                {uploadingAvatar ? <Loader2 className="w-4 h-4 animate-spin" /> : <Camera className="w-4 h-4" />}
                <input type="file" className="hidden" accept="image/png, image/jpeg, image/webp" onChange={handleAvatarUpload} />
              </label>
            )}
          </div>

          {/* Info */}
          <div className="flex-1 text-center md:text-left">
            {isEditing ? (
              <div className="space-y-4">
                <input
                  type="text"
                  value={editData.displayName}
                  onChange={e => setEditData(prev => ({ ...prev, displayName: e.target.value }))}
                  className="w-full bg-background border border-border rounded-xl px-4 py-2 text-xl font-bold focus:ring-2 focus:ring-primary outline-none"
                  placeholder="Display Name"
                />
                <textarea
                  value={editData.bio}
                  onChange={e => setEditData(prev => ({ ...prev, bio: e.target.value }))}
                  className="w-full bg-background border border-border rounded-xl px-4 py-2 text-sm text-muted-foreground focus:ring-2 focus:ring-primary outline-none resize-none"
                  placeholder="Tell us about yourself..."
                  rows={2}
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleUpdateProfile}
                    disabled={savingProfile}
                    className="flex-1 gradient-primary text-primary-foreground py-2 rounded-xl font-semibold shadow-lg hover:shadow-primary/20 transition-all disabled:opacity-50"
                  >
                    {savingProfile ? <Loader2 className="w-4 h-4 animate-spin mx-auto" /> : "Save Changes"}
                  </button>
                  <button
                    onClick={() => setIsEditing(false)}
                    className="px-4 py-2 bg-muted text-foreground border border-border rounded-xl font-semibold hover:bg-muted/80 transition-all"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-center justify-center md:justify-start gap-3 mb-2">
                  <h1 className="text-3xl font-bold">{profile.displayName || profile.username}</h1>
                </div>
                <p className="text-muted-foreground mb-4">@{profile.username}</p>

                {/* Follow Stats */}
                <div className="flex items-center gap-6 mb-4">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-foreground">{profile.followersCount}</span>
                    <span className="text-sm text-muted-foreground">Followers</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-foreground">{profile.followingCount}</span>
                    <span className="text-sm text-muted-foreground">Following</span>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 mb-6">
                  {!isOwner && (
                    <button
                      onClick={handleFollowToggle}
                      className={`flex items-center gap-2 px-6 py-2 rounded-full font-semibold transition-all ${profile.isFollowing
                        ? "bg-muted text-foreground border border-border hover:bg-destructive/10 hover:text-destructive hover:border-destructive/20"
                        : "bg-foreground text-background hover:opacity-90 shadow-lg hover:shadow-xl"
                        }`}
                    >
                      {profile.isFollowing ? (
                        <>
                          <UserCheck className="w-4 h-4" />
                          Following
                        </>
                      ) : (
                        <>
                          <UserPlus className="w-4 h-4" />
                          Follow
                        </>
                      )}
                    </button>
                  )}
                  {isOwner && (
                    <>
                      <button
                        onClick={() => setIsEditing(true)}
                        className="px-6 py-2 bg-muted text-foreground border border-border rounded-full font-semibold hover:bg-muted/80 transition-all"
                      >
                        Edit Profile
                      </button>
                      <button
                        onClick={handleLogout}
                        className="p-2 bg-muted text-muted-foreground border border-border rounded-full hover:bg-destructive/10 hover:text-destructive transition-all"
                        title="Sign Out"
                      >
                        <LogOut className="w-5 h-5" />
                      </button>
                    </>
                  )}
                </div>

                {profile.bio && <p className="text-sm text-foreground/80 mb-4 max-w-md">{profile.bio}</p>}

                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${statusConfig.bg} ${statusConfig.color} font-medium border border-current/20`}>
                  <statusConfig.icon className="w-5 h-5" />
                  {statusConfig.label}
                </div>
              </>
            )}
          </div>

          {/* 2. TRUST STATUS SUMMARY */}
          <div className="w-full md:w-64 bg-background/50 rounded-2xl p-4 border border-border">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-green-500" />
              Trust Score
            </h3>

            {/* Real % */}
            <div className="mb-3">
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-green-600">Real Uploads ({profile.realUploads})</span>
                <span className="font-bold">{profile.realPercentage}%</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${profile.realPercentage}%` }}
                  className="h-full bg-green-500 rounded-full"
                />
              </div>
            </div>

            {/* Fake % */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-red-600">Fake Uploads ({profile.fakeUploads})</span>
                <span className="font-bold">{profile.fakePercentage}%</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${profile.fakePercentage}%` }}
                  className="h-full bg-red-500 rounded-full"
                />
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* 4. UPLOAD VERIFICATION STATUS SECTION */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="bg-card p-4 rounded-2xl border border-border text-center">
          <h4 className="text-muted-foreground text-xs font-semibold uppercase mb-1">Total</h4>
          <p className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            {profile.totalUploads}
          </p>
        </div>
        <div className="bg-green-500/5 p-4 rounded-2xl border border-green-500/10 text-center">
          <h4 className="text-green-600 text-xs font-semibold uppercase mb-1">Verified</h4>
          <p className="text-2xl font-bold text-green-600">
            {profile.realUploads}
          </p>
        </div>
        <div className="bg-red-500/5 p-4 rounded-2xl border border-red-500/10 text-center">
          <h4 className="text-red-600 text-xs font-semibold uppercase mb-1">Fake Uploads</h4>
          <p className="text-2xl font-bold text-red-600">
            {profile.fakeUploads}
          </p>
        </div>
      </div>

      {/* 3. CONTENT TABS (POSTS & REELS) */}
      <div className="mb-8">
        <div className="flex items-center gap-1 bg-muted/30 p-1 rounded-xl w-fit mb-6">
          <button
            onClick={() => setActiveTab('posts')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-semibold transition-all ${activeTab === 'posts' ? 'bg-white dark:bg-gray-800 shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
          >
            <Grid className="w-4 h-4" />
            Posts
          </button>
          <button
            onClick={() => setActiveTab('reels')}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-semibold transition-all ${activeTab === 'reels' ? 'bg-white dark:bg-gray-800 shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
          >
            <Video className="w-4 h-4" />
            Reels
          </button>
        </div>

        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'posts' && (
            <div className="grid grid-cols-3 gap-4">
              {profile.posts.length > 0 ? (
                profile.posts.map((post) => (
                  <div
                    key={post.id}
                    onClick={() => setSelectedPost(post)}
                    className="aspect-square bg-muted rounded-xl overflow-hidden relative group cursor-pointer"
                  >
                    <img src={post.media_url} alt="" className="w-full h-full object-cover transition-transform group-hover:scale-105" />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center p-4">
                      {post.caption && (
                        <p className="text-white text-xs text-center line-clamp-2 mb-4">{post.caption}</p>
                      )}
                      {isOwner && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePostDelete(post.id);
                          }}
                          className="p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors shadow-lg"
                          title="Delete Post"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="col-span-3 text-center py-12 text-muted-foreground">
                  <Upload className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No verified posts yet</p>
                </div>
              )}
            </div>
          )}
          {activeTab === 'reels' && (
            <div className="grid grid-cols-4 gap-4">
              {profile.reels.length > 0 ? (
                profile.reels.map((reel) => (
                  <div
                    key={reel.id}
                    onClick={() => setSelectedPost(reel)}
                    className="aspect-[9/16] bg-muted rounded-xl overflow-hidden relative group cursor-pointer"
                  >
                    <video src={reel.media_url} className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-black/20 group-hover:bg-black/40 transition-colors flex items-center justify-center">
                      <div className="w-8 h-8 bg-white/20 backdrop-blur rounded-full flex items-center justify-center">
                        <div className="w-0 h-0 border-t-[6px] border-t-transparent border-l-[10px] border-l-white border-b-[6px] border-b-transparent ml-1" />
                      </div>
                    </div>
                    {isOwner && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePostDelete(reel.id);
                        }}
                        className="absolute top-2 right-2 p-1.5 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600 shadow-lg"
                        title="Delete Reel"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))
              ) : (
                <div className="col-span-4 text-center py-12 text-muted-foreground">
                  <Video className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p>No verified reels yet</p>
                </div>
              )}
            </div>
          )}
        </motion.div>
      </div>

      {/* 5. VERIFICATION HISTORY (PRIVATE) */}
      {
        isOwner && (
          <div className="border-t border-border pt-8">
            <button
              onClick={() => setShowHistory(!showHistory)}
              className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground mb-4"
            >
              <Lock className="w-4 h-4" />
              {showHistory ? "Hide Private Verification History" : "Show Private Verification History"}
            </button>

            <AnimatePresence>
              {showHistory && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="bg-card rounded-2xl border border-border overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="bg-muted/50 border-b border-border">
                          <tr>
                            <th className="px-4 py-3 text-left font-medium">Date</th>
                            <th className="px-4 py-3 text-left font-medium">Verdict</th>
                            <th className="px-4 py-3 text-left font-medium">Score</th>
                            <th className="px-4 py-3 text-left font-medium">Reason</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border">
                          {history.map((item) => (
                            <tr key={item.id} className="hover:bg-muted/20">
                              <td className="px-4 py-3 text-muted-foreground">
                                {new Date(item.created_at).toLocaleDateString()}
                              </td>
                              <td className="px-4 py-3">
                                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${(item.verdict === 'REAL' || item.verdict === 'APPROVED')
                                  ? 'bg-green-500/10 text-green-600'
                                  : 'bg-red-500/10 text-red-600'
                                  }`}>
                                  {(item.verdict === 'REAL' || item.verdict === 'APPROVED')
                                    ? <CheckCircle2 className="w-3 h-3" />
                                    : <XCircle className="w-3 h-3" />
                                  }
                                  {item.verdict}
                                </span>
                              </td>
                              <td className="px-4 py-3 font-mono text-muted-foreground">
                                {Number(item.score).toFixed(4)}
                              </td>
                              <td className="px-4 py-3 text-muted-foreground max-w-xs truncate">
                                {item.reason || '-'}
                              </td>
                            </tr>
                          ))}
                          {history.length === 0 && (
                            <tr>
                              <td colSpan={4} className="p-8 text-center text-muted-foreground">
                                No verification history available.
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      }

      {/* 6. ACCOUNT SETTINGS (DANGER ZONE) */}
      {
        isOwner && (
          <div className="mt-12 border-t border-border pt-12">
            <h3 className="text-lg font-bold mb-6">Account Settings</h3>

            <div className="space-y-6">
              {/* Sign Out */}
              <div className="bg-card border border-border rounded-2xl p-6 flex flex-col md:flex-row items-center justify-between gap-4">
                <div>
                  <h4 className="font-semibold">Sign Out</h4>
                  <p className="text-sm text-muted-foreground">Securely end your current session.</p>
                </div>
                <button
                  onClick={handleLogout}
                  className="px-4 py-2 text-sm bg-muted hover:bg-muted/80 text-foreground font-medium rounded-lg transition-colors flex items-center gap-2"
                >
                  <LogOut className="w-3.5 h-3.5" />
                  Sign Out
                </button>
              </div>

              {/* Delete Account */}
              <div className="bg-destructive/5 border border-destructive/10 rounded-2xl p-6">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                  <div>
                    <h4 className="font-bold text-destructive mb-2 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      Delete Account
                    </h4>
                    <p className="text-sm text-muted-foreground max-w-xl">
                      Permanently delete your account and all associated data. All posts, verification logs, and social information will be removed from the platform. This action is irreversible.
                    </p>
                  </div>

                  {!showDeleteConfirm ? (
                    <button
                      onClick={() => setShowDeleteConfirm(true)}
                      className="px-4 py-2 text-sm bg-red-600 hover:bg-red-700 text-white font-bold rounded-lg transition-colors shadow-lg shadow-red-500/20 flex items-center gap-2"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </button>
                  ) : (
                    <div className="w-full md:w-auto p-4 bg-background border border-destructive/20 rounded-xl animate-in fade-in zoom-in-95 duration-200">
                      <p className="text-xs font-bold uppercase text-destructive mb-2">
                        Type "DELETE MY ACCOUNT"
                      </p>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={deletePhrase}
                          onChange={(e) => setDeletePhrase(e.target.value)}
                          className="bg-muted px-3 py-2 rounded-lg text-sm font-mono border border-border focus:border-destructive outline-none w-full md:w-48"
                          placeholder="DELETE MY ACCOUNT"
                        />
                        <button
                          onClick={handleDeleteAccount}
                          disabled={deletePhrase !== "DELETE MY ACCOUNT" || isDeletingAccount}
                          className="px-4 py-2 bg-destructive text-white text-sm font-bold rounded-lg disabled:opacity-50 hover:bg-destructive/90 transition-colors"
                        >
                          {isDeletingAccount ? <Loader2 className="w-4 h-4 animate-spin" /> : "Confirm"}
                        </button>
                        <button
                          onClick={() => {
                            setShowDeleteConfirm(false);
                            setDeletePhrase("");
                          }}
                          className="px-3 py-2 text-sm text-muted-foreground hover:text-foreground"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )
      }

      {/* Lightbox Modal */}
      <AnimatePresence>
        {selectedPost && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedPost(null)}
            className="fixed inset-0 z-50 bg-black/95 backdrop-blur-md flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              // Responsive Layout: Stacked on mobile, Side-by-Side on Desktop
              // Fixed height ensuring it fits within viewport with safe margins
              className="relative max-w-6xl w-full h-[85vh] md:h-[80vh] bg-card border border-border rounded-xl overflow-hidden shadow-2xl flex flex-col md:flex-row"
            >
              {/* Close Button Mobile - Floating */}
              <button
                onClick={() => setSelectedPost(null)}
                className="absolute top-2 right-2 z-50 p-2 text-white/90 bg-black/60 rounded-full md:hidden backdrop-blur-sm"
              >
                <XCircle className="w-6 h-6" />
              </button>

              {/* Close Button Desktop - Floating */}
              <button
                onClick={() => setSelectedPost(null)}
                className="absolute top-4 right-4 z-50 p-2 text-foreground/50 hover:text-destructive transition-colors hidden md:block bg-background/50 rounded-full backdrop-blur-sm"
              >
                <XCircle className="w-8 h-8" />
              </button>

              {/* MEDIA SECTION (Left/Top) - Takes remaining space on Mobile, 60% Width on Desktop */}
              <div className="flex-1 md:flex-[1.5] bg-black flex items-center justify-center overflow-hidden min-h-0 relative">
                {selectedPost.media_type === 'video' ? (
                  <video
                    src={selectedPost.media_url}
                    controls
                    autoPlay
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <img
                    src={selectedPost.media_url}
                    alt={selectedPost.caption}
                    className="w-full h-full object-contain"
                  />
                )}
              </div>

              {/* DETAILS SECTION (Right/Bottom) - Fixed Height on Mobile, Side on Desktop */}
              <div className="flex flex-col w-full md:w-[400px] bg-background md:border-l border-border h-[40vh] md:h-full flex-shrink-0">

                {/* Header - Fixed to top of details */}
                <div className="flex-shrink-0 p-4 border-b border-border flex items-center gap-3 bg-background/95 backdrop-blur z-10">
                  <div className="w-10 h-10 rounded-full bg-muted overflow-hidden flex-shrink-0 border border-border">
                    {profile?.avatarUrl ? (
                      <img src={profile.avatarUrl} className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-primary/10 text-primary font-bold">
                        {profile?.username?.[0]?.toUpperCase()}
                      </div>
                    )}
                  </div>
                  <div>
                    <h4 className="font-bold text-sm">{profile?.username}</h4>
                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                      <ShieldCheck className="w-3 h-3 text-verified" />
                      <span>Verified Authentic</span>
                    </div>
                  </div>
                </div>

                {/* Caption & Comments - Scrollable */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
                  {selectedPost.caption ? (
                    <div className="text-foreground text-sm leading-relaxed whitespace-pre-wrap">
                      <span className="font-bold mr-2">{profile?.username}</span>
                      {selectedPost.caption}
                    </div>
                  ) : (
                    <p className="text-muted-foreground italic text-sm">No caption</p>
                  )}

                  {/* Placeholder for comments */}
                  <div className="pt-4 border-t border-border/50">
                    <p className="text-xs text-muted-foreground font-medium mb-2">Comments are verified</p>
                    <div className="space-y-3">
                      <div className="h-2 w-3/4 bg-muted animate-pulse rounded"></div>
                      <div className="h-2 w-1/2 bg-muted animate-pulse rounded"></div>
                    </div>
                  </div>
                </div>

                {/* Footer (Actions) - Fixed to bottom */}
                <div className="flex-shrink-0 p-3 border-t border-border bg-muted/10">
                  <div className="flex gap-4 text-muted-foreground mb-3 text-sm">
                    <div className="flex items-center gap-1">
                      <Heart className="w-5 h-5 text-foreground" />
                      <span className="font-bold text-foreground">{selectedPost.like_count || 0}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MessageCircle className="w-5 h-5" />
                      <span>{selectedPost.comment_count || 0}</span>
                    </div>
                    <div className="flex-1 text-right text-xs text-muted-foreground pt-1">
                      {new Date(selectedPost.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Add a verified comment..."
                      className="w-full px-4 py-2 bg-muted/50 rounded-full text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                      readOnly
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div >
  );
}
