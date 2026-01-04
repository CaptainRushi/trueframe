import { motion } from "framer-motion";
import { Bell, MessageSquare, ShieldCheck } from "lucide-react";
import { StoryCircle } from "@/components/feed/StoryCircle";
import { PostCard } from "@/components/feed/PostCard";
import user1 from "@/assets/user-1.jpg";
import user2 from "@/assets/user-2.jpg";
import user3 from "@/assets/user-3.jpg";
import post1 from "@/assets/post-1.jpg";
import post2 from "@/assets/post-2.jpg";

const stories = [
  { image: user1, username: "Alana", isLive: true },
  { image: user2, username: "Maya" },
  { image: user3, username: "Marcus" },
  { image: user1, username: "Sofia" },
  { image: user2, username: "Emma" },
];

const posts = [
  {
    id: "1",
    userAvatar: user1,
    username: "Alana Maesya",
    image: post1,
    caption: "Living my best life ☀️ #authentic #trueframe #verified",
    likes: 1245,
    comments: 173,
    timestamp: "2 hours ago",
  },
  {
    id: "2",
    userAvatar: user3,
    username: "Marcus Cole",
    image: post2,
    caption: "Golden hour vibes 🌅 Nothing beats a real sunset. #nofilter #trueframe",
    likes: 892,
    comments: 64,
    timestamp: "5 hours ago",
  },
];

export default function Feed() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 glass border-b border-border">
        <div className="flex items-center justify-between p-4 max-w-lg mx-auto">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-7 h-7 text-primary" />
            <span className="text-xl font-bold text-foreground">TrueFrame</span>
          </div>
          <div className="flex items-center gap-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="p-2 hover:bg-muted rounded-full transition-colors"
            >
              <Bell className="w-6 h-6 text-foreground" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="p-2 hover:bg-muted rounded-full transition-colors"
            >
              <MessageSquare className="w-6 h-6 text-foreground" />
            </motion.button>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto">
        {/* Stories */}
        <motion.section
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 overflow-x-auto"
        >
          <div className="flex gap-4">
            {stories.map((story, index) => (
              <StoryCircle key={index} {...story} />
            ))}
          </div>
        </motion.section>

        {/* Feed */}
        <section className="space-y-6 px-4 pb-8">
          {posts.map((post, index) => (
            <motion.div
              key={post.id}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.15 }}
            >
              <PostCard {...post} />
            </motion.div>
          ))}
        </section>
      </main>
    </div>
  );
}
