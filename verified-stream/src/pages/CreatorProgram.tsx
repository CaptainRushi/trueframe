import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Award, ShieldCheck, TrendingUp, Star, Eye, Users,
  CheckCircle2, XCircle, Loader2, Crown, ArrowRight
} from "lucide-react";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";
import { Link } from "react-router-dom";

interface CreatorStatus {
  isVerifiedCreator: boolean;
  verifiedAt: string | null;
  eligible: boolean;
  criteria: {
    trustScore: { required: number; current: number; met: boolean };
    totalUploads: { required: number; current: number; met: boolean };
    authenticityRate: { required: number; current: number; met: boolean };
    identityVerified: { required: boolean; current: boolean; met: boolean };
  };
  benefits: string[];
}

interface LeaderboardEntry {
  rank: number;
  id: string;
  username: string;
  display_name: string;
  avatar_url: string;
  trust_score: number;
  real_percentage: number;
  total_attempts: number;
  is_verified_creator: boolean;
  identity_verified: boolean;
  trustLevel: string;
}

export default function CreatorProgram() {
  const [status, setStatus] = useState<CreatorStatus | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [applying, setApplying] = useState(false);
  const [activeTab, setActiveTab] = useState<'status' | 'leaderboard'>('status');

  useEffect(() => {
    fetchCreatorStatus();
    fetchLeaderboard();
  }, []);

  const fetchCreatorStatus = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/creator/status`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });
      if (res.ok) setStatus(await res.json());
    } catch (e) {
      console.error("Failed to fetch creator status", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchLeaderboard = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/creator/leaderboard`);
      if (res.ok) {
        const data = await res.json();
        setLeaderboard(data.leaderboard || []);
      }
    } catch (e) {
      console.error("Failed to fetch leaderboard", e);
    }
  };

  const handleApply = async () => {
    setApplying(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/creator/apply`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        await fetchCreatorStatus();
      } else {
        const data = await res.json();
        alert(data.error || 'Application failed');
      }
    } catch (e: any) {
      alert(e.message);
    } finally {
      setApplying(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background pb-20 md:pb-6">
      <div className="max-w-2xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 rounded-full text-primary border border-primary/20 mb-4"
          >
            <Award className="w-5 h-5" />
            <span className="text-sm font-bold uppercase tracking-wider">Creator Program</span>
          </motion.div>
          <h1 className="text-4xl font-black tracking-tight mb-2">Verified Authentic Creator</h1>
          <p className="text-muted-foreground">Earn the highest badge of authenticity on TrueFrame</p>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-border pb-2">
          <button
            onClick={() => setActiveTab('status')}
            className={`px-4 py-2 text-sm font-bold rounded-lg transition-colors ${
              activeTab === 'status' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            My Status
          </button>
          <button
            onClick={() => setActiveTab('leaderboard')}
            className={`px-4 py-2 text-sm font-bold rounded-lg transition-colors flex items-center gap-1.5 ${
              activeTab === 'leaderboard' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Crown className="w-4 h-4" />
            Leaderboard
          </button>
        </div>

        {activeTab === 'status' && status && (
          <div className="space-y-6">
            {/* Current Status */}
            {status.isVerifiedCreator ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-primary/10 border border-primary/30 rounded-2xl p-6 text-center space-y-3"
              >
                <div className="w-20 h-20 bg-primary/20 rounded-full flex items-center justify-center mx-auto">
                  <Award className="w-10 h-10 text-primary" />
                </div>
                <h2 className="text-2xl font-black text-primary">Verified Authentic Creator</h2>
                <p className="text-sm text-muted-foreground">
                  Verified since {status.verifiedAt ? new Date(status.verifiedAt).toLocaleDateString() : 'N/A'}
                </p>
                <div className="flex flex-wrap justify-center gap-2 mt-4">
                  <span className="px-3 py-1 bg-primary/20 rounded-full text-xs font-bold text-primary">Authenticity Badge</span>
                  <span className="px-3 py-1 bg-green-500/20 rounded-full text-xs font-bold text-green-500">Community Verifier</span>
                  <span className="px-3 py-1 bg-blue-500/20 rounded-full text-xs font-bold text-blue-500">Priority Verification</span>
                </div>
              </motion.div>
            ) : (
              <>
                {/* Eligibility Criteria */}
                <div className="bg-card rounded-2xl border border-border p-6 space-y-5">
                  <h3 className="text-lg font-black flex items-center gap-2">
                    <Star className="w-5 h-5 text-primary" />
                    Eligibility Requirements
                  </h3>

                  <div className="space-y-4">
                    {Object.entries(status.criteria).map(([key, criterion]) => (
                      <div key={key} className="flex items-center gap-4">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                          criterion.met ? 'bg-green-500/10' : 'bg-muted'
                        }`}>
                          {criterion.met ? (
                            <CheckCircle2 className="w-5 h-5 text-green-500" />
                          ) : (
                            <XCircle className="w-5 h-5 text-muted-foreground" />
                          )}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm font-bold capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</p>
                          <p className="text-xs text-muted-foreground">
                            {typeof criterion.current === 'boolean'
                              ? (criterion.current ? 'Verified' : 'Not verified')
                              : `${criterion.current} / ${criterion.required} required`
                            }
                          </p>
                        </div>
                        {typeof criterion.current !== 'boolean' && (
                          <div className="w-20">
                            <div className="h-2 bg-muted rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${criterion.met ? 'bg-green-500' : 'bg-primary'}`}
                                style={{ width: `${Math.min(100, (Number(criterion.current) / Number(criterion.required)) * 100)}%` }}
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={handleApply}
                    disabled={!status.eligible || applying}
                    className="w-full py-4 rounded-xl font-black text-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:grayscale bg-primary text-primary-foreground shadow-lg"
                  >
                    {applying ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : status.eligible ? (
                      <>
                        <Award className="w-5 h-5" />
                        Claim Creator Badge
                        <ArrowRight className="w-5 h-5" />
                      </>
                    ) : (
                      'Requirements Not Met'
                    )}
                  </button>
                </div>

                {/* Benefits */}
                <div className="bg-card rounded-2xl border border-border p-6 space-y-4">
                  <h3 className="text-lg font-black">Creator Benefits</h3>
                  <div className="space-y-3">
                    {[
                      { icon: ShieldCheck, text: 'Verified Authentic Creator badge on your profile' },
                      { icon: TrendingUp, text: 'Higher visibility in feed algorithm ranking' },
                      { icon: Eye, text: 'Priority content verification processing' },
                      { icon: Users, text: 'Community Verifier eligibility to flag content' },
                      { icon: Star, text: 'Authenticity badge displayed on all your posts' }
                    ].map((benefit, i) => (
                      <div key={i} className="flex items-center gap-3 p-3 bg-muted/30 rounded-xl">
                        <benefit.icon className="w-5 h-5 text-primary flex-shrink-0" />
                        <p className="text-sm">{benefit.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'leaderboard' && (
          <div className="space-y-3">
            {leaderboard.map((entry) => (
              <Link
                key={entry.id}
                to={`/profile/${entry.username}`}
                className="flex items-center gap-4 p-4 bg-card border border-border rounded-2xl hover:bg-muted/50 transition-colors"
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center font-black text-sm ${
                  entry.rank <= 3 ? 'bg-primary/10 text-primary' : 'bg-muted text-muted-foreground'
                }`}>
                  {entry.rank <= 3 ? (
                    <Crown className={`w-4 h-4 ${entry.rank === 1 ? 'text-yellow-500' : entry.rank === 2 ? 'text-gray-400' : 'text-amber-600'}`} />
                  ) : (
                    entry.rank
                  )}
                </div>

                <div className="w-10 h-10 rounded-full bg-muted overflow-hidden flex-shrink-0">
                  {entry.avatar_url ? (
                    <img src={entry.avatar_url} className="w-full h-full object-cover" alt="" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-sm font-bold bg-primary/10 text-primary">
                      {entry.username[0].toUpperCase()}
                    </div>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5">
                    <p className="font-bold truncate">{entry.display_name || entry.username}</p>
                    {entry.is_verified_creator && <Award className="w-4 h-4 text-primary flex-shrink-0" />}
                  </div>
                  <p className="text-xs text-muted-foreground">@{entry.username} · {entry.trustLevel}</p>
                </div>

                <div className="text-right">
                  <p className="text-lg font-black text-primary">{entry.trust_score}</p>
                  <p className="text-[10px] text-muted-foreground font-bold">{entry.real_percentage}% real</p>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
