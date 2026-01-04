import { motion } from "framer-motion";
import { Settings, Grid3X3, Heart, Bookmark, ShieldCheck, BadgeCheck } from "lucide-react";
import { VerifiedBadge } from "@/components/ui/VerifiedBadge";
import user2 from "@/assets/user-2.jpg";
import post1 from "@/assets/post-1.jpg";
import post2 from "@/assets/post-2.jpg";
import heroModel from "@/assets/hero-model.jpg";

const stats = [
  { value: "567K", label: "Followers" },
  { value: "1,665", label: "Following" },
  { value: "166", label: "Posts" },
];

const tabs = [
  { icon: Grid3X3, label: "Posts" },
  { icon: Heart, label: "Liked" },
  { icon: Bookmark, label: "Saved" },
];

const posts = [post1, post2, heroModel, post1, post2, heroModel];

export default function Profile() {
  return (
    <div className="min-h-screen bg-background pb-24">
      {/* Header Banner */}
      <div className="relative h-32 bg-gradient-to-br from-foreground/90 to-foreground overflow-hidden">
        <div className="absolute inset-0 opacity-30">
          <svg viewBox="0 0 100 100" className="w-full h-full">
            <pattern id="mountains" patternUnits="userSpaceOnUse" width="100" height="100">
              <path d="M0 80 L25 50 L50 70 L75 40 L100 60 L100 100 L0 100 Z" fill="currentColor" className="text-background/20" />
            </pattern>
            <rect width="100" height="100" fill="url(#mountains)" />
          </svg>
        </div>
        <button className="absolute top-4 right-4 p-2 bg-background/20 backdrop-blur-sm rounded-full">
          <Settings className="w-5 h-5 text-primary-foreground" />
        </button>
      </div>

      {/* Profile Info */}
      <div className="relative px-4">
        {/* Avatar */}
        <div className="absolute -top-16 left-1/2 -translate-x-1/2">
          <div className="relative">
            <div className="p-1 bg-background rounded-full">
              <img
                src={user2}
                alt="Profile"
                className="w-28 h-28 rounded-full object-cover ring-4 ring-background"
              />
            </div>
            <div className="absolute -bottom-1 -right-1">
              <VerifiedBadge size="lg" timestamp="Account verified" />
            </div>
          </div>
        </div>

        <div className="pt-16 text-center">
          <div className="flex items-center justify-center gap-2">
            <h1 className="text-2xl font-bold text-foreground">Katty Abrahams</h1>
            <BadgeCheck className="w-6 h-6 text-primary" />
          </div>
          <p className="text-muted-foreground mt-1">
            I'm delighted to introduce myself as a professional model 🌟
          </p>
        </div>

        {/* Stats */}
        <div className="flex justify-center gap-8 mt-6">
          {stats.map((stat) => (
            <div key={stat.label} className="text-center">
              <p className="text-xl font-bold text-foreground">{stat.value}</p>
              <p className="text-sm text-muted-foreground">{stat.label}</p>
            </div>
          ))}
        </div>

        {/* Actions */}
        <div className="flex gap-3 mt-6 max-w-sm mx-auto">
          <button className="flex-1 py-3 gradient-primary text-primary-foreground rounded-xl font-semibold shadow-glow">
            Follow
          </button>
          <button className="flex-1 py-3 bg-muted text-foreground rounded-xl font-semibold">
            Message
          </button>
          <button className="px-4 py-3 bg-muted text-foreground rounded-xl font-semibold">
            ···
          </button>
        </div>

        {/* Trust Score */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-verified/10 border border-verified/30 rounded-2xl"
        >
          <div className="flex items-center gap-3">
            <div className="p-2 bg-verified rounded-xl">
              <ShieldCheck className="w-6 h-6 text-verified-foreground" />
            </div>
            <div className="flex-1">
              <p className="font-semibold text-verified">100% Verified Creator</p>
              <p className="text-sm text-muted-foreground">All uploads verified authentic • 0 violations</p>
            </div>
          </div>
        </motion.div>

        {/* Tabs */}
        <div className="flex border-b border-border mt-6">
          {tabs.map((tab, index) => (
            <button
              key={tab.label}
              className={`flex-1 flex items-center justify-center gap-2 py-4 border-b-2 transition-colors ${
                index === 0
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground"
              }`}
            >
              <tab.icon className="w-5 h-5" />
            </button>
          ))}
        </div>

        {/* Posts Grid */}
        <div className="grid grid-cols-3 gap-1 mt-2">
          {posts.map((post, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className="relative aspect-square cursor-pointer overflow-hidden"
            >
              <img
                src={post}
                alt={`Post ${index + 1}`}
                className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
              />
              <div className="absolute top-1 right-1">
                <VerifiedBadge size="sm" showTooltip={false} />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
