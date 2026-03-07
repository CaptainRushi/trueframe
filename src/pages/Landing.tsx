import { motion } from "framer-motion";
import { ShieldCheck, Lock, Sparkles, ChevronRight, Camera, Eye, Users, Award, Brain, Globe } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import heroModel from "@/assets/hero-man.png";
import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";

const features = [
  {
    icon: ShieldCheck,
    title: "AI Verification",
    description: "Every post verified by advanced deepfake detection",
  },
  {
    icon: Lock,
    title: "Zero Fakes",
    description: "No synthetic content makes it to your feed",
  },
  {
    icon: Sparkles,
    title: "Trust First",
    description: "Build authentic connections with real people",
  },
];

const advancedFeatures = [
  {
    icon: Brain,
    title: "Multi-Layer AI Detection",
    description: "Face swap, GAN faces, photoshop, AI generation, voice cloning, and lip sync deepfake detection",
  },
  {
    icon: Camera,
    title: "Verified Capture Mode",
    description: "Photos taken inside TrueFrame are signed with device metadata and get the 'Captured on TrueFrame' badge",
  },
  {
    icon: Eye,
    title: "Content Transparency",
    description: "Tap any post to see full verification details — authenticity score, deepfake probability, and AI analysis breakdown",
  },
  {
    icon: Users,
    title: "Community Fact-Checking",
    description: "Trusted users become Community Verifiers who can flag misinformation and add fact sources",
  },
  {
    icon: Award,
    title: "Verified Creator Program",
    description: "Earn the Verified Authentic Creator badge with higher visibility and priority verification",
  },
  {
    icon: Globe,
    title: "Reputation-Based Feed",
    description: "Feed algorithm prioritizes high-trust creators, verified content, and authentic media",
  },
];

export default function Landing() {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    verifiedPosts: 0,
    detectionRate: "99.9%",
    deepfakesBlocked: 0
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        if (!supabase) return;

        // Fetch real data
        const [postsRes, logsRes] = await Promise.all([
          supabase.from("posts").select("*", { count: "exact", head: true }),
          supabase.from("verification_logs")
            .select("*", { count: "exact", head: true })
            .in("verdict", ["REJECTED", "FAKE", "BLOCK_FAKE", "BLOCK_UNVERIFIED"])
        ]);

        setStats({
          verifiedPosts: postsRes.count || 0,
          detectionRate: "99.9%", // Model rating remains static metric
          deepfakesBlocked: logsRes.count || 0
        });

      } catch (err) {
        console.error("Failed to fetch platform stats", err);
      }
    };

    fetchStats();

    const checkAuth = async () => {
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        navigate("/feed", { replace: true });
      }
    };
    checkAuth();
  }, [navigate]);

  return (
    <div className="min-h-screen gradient-hero overflow-hidden">
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-8 h-8 text-primary-foreground" />
            <span className="text-xl font-bold text-primary-foreground">TrueFrame</span>
          </div>
          <Link
            to="/login"
            className="text-sm font-medium text-primary-foreground/80 hover:text-primary-foreground transition-colors"
          >
            Sign In
          </Link>
        </motion.header>

        {/* Hero */}
        <main className="mt-12 lg:mt-20 grid lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-8"
          >
            <h1 className="text-5xl lg:text-6xl font-bold text-primary-foreground leading-tight">
              Join the{" "}
              <span className="relative">
                Authentic
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ delay: 0.8 }}
                  className="absolute -bottom-2 left-0 right-0 h-1 bg-primary-foreground/40 rounded-full origin-left"
                />
              </span>{" "}
              Social Network
            </h1>
            <p className="text-xl text-primary-foreground/80 max-w-md">
              Every photo and video verified real. No deepfakes, no manipulation, just truth.
            </p>

            {/* Features */}
            <div className="space-y-4">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                  className="flex items-center gap-4 text-primary-foreground/90"
                >
                  <div className="p-2 rounded-xl bg-primary-foreground/20">
                    <feature.icon className="w-5 h-5" />
                  </div>
                  <div>
                    <span className="font-semibold">{feature.title}</span>
                    <span className="text-primary-foreground/70"> — {feature.description}</span>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* CTA */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
            >
              <Link
                to="/login"
                className="inline-flex items-center gap-3 px-8 py-4 bg-foreground text-background rounded-full font-semibold text-lg shadow-elevated hover:shadow-lg transition-all hover:scale-105"
              >
                <Lock className="w-5 h-5" />
                Get Started
                <ChevronRight className="w-5 h-5" />
              </Link>
            </motion.div>
          </motion.div>

          {/* Hero Image */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="relative hidden lg:block"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-t from-primary/40 to-transparent rounded-[3rem] z-10" />
              <img
                src={heroModel}
                alt="TrueFrame user"
                className="w-full max-w-md mx-auto rounded-[3rem] shadow-elevated object-cover aspect-[3/4]"
              />
              {/* Verified Badge Overlay */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1 }}
                className="absolute bottom-8 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-card/95 backdrop-blur-xl px-6 py-4 rounded-2xl shadow-elevated"
              >
                <div className="p-2 rounded-full bg-verified">
                  <ShieldCheck className="w-6 h-6 text-verified-foreground" />
                </div>
                <div>
                  <p className="font-semibold text-foreground">100% Verified</p>
                  <p className="text-sm text-muted-foreground">AI-authenticated content</p>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </main>

        {/* Trust Stats */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9 }}
          className="mt-20 grid grid-cols-2 lg:grid-cols-3 gap-6 max-w-2xl mx-auto"
        >
          <div className="text-center">
            <p className="text-3xl lg:text-4xl font-bold text-primary-foreground">{stats.verifiedPosts.toLocaleString()}</p>
            <p className="text-sm text-primary-foreground/70">Verified Posts</p>
          </div>
          <div className="text-center hidden lg:block">
            <p className="text-3xl lg:text-4xl font-bold text-primary-foreground">{stats.detectionRate}</p>
            <p className="text-sm text-primary-foreground/70">Detection Rate</p>
          </div>
          <div className="text-center">
            <p className="text-3xl lg:text-4xl font-bold text-primary-foreground">{stats.deepfakesBlocked.toLocaleString()}</p>
            <p className="text-sm text-primary-foreground/70">Deepfakes Blocked</p>
          </div>
        </motion.section>

        {/* Advanced Features Grid */}
        <motion.section
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1 }}
          className="mt-24 mb-20 max-w-4xl mx-auto"
        >
          <h2 className="text-3xl lg:text-4xl font-bold text-primary-foreground text-center mb-4">
            Advanced Platform Features
          </h2>
          <p className="text-center text-primary-foreground/70 mb-12 max-w-xl mx-auto">
            TrueFrame goes beyond basic verification with a complete authenticity ecosystem
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {advancedFeatures.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 + index * 0.1 }}
                className="bg-primary-foreground/10 backdrop-blur-sm border border-primary-foreground/20 rounded-2xl p-6 hover:bg-primary-foreground/15 transition-colors"
              >
                <div className="p-3 rounded-xl bg-primary-foreground/20 w-fit mb-4">
                  <feature.icon className="w-6 h-6 text-primary-foreground" />
                </div>
                <h3 className="text-lg font-bold text-primary-foreground mb-2">{feature.title}</h3>
                <p className="text-sm text-primary-foreground/70 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.section>
      </div>
    </div>
  );
}
