import { motion } from "framer-motion";
import { Search, TrendingUp, ShieldCheck } from "lucide-react";
import { VerifiedBadge } from "@/components/ui/VerifiedBadge";
import user1 from "@/assets/user-1.jpg";
import user2 from "@/assets/user-2.jpg";
import post1 from "@/assets/post-1.jpg";
import post2 from "@/assets/post-2.jpg";
import heroModel from "@/assets/hero-model.jpg";

const categories = ["All", "Trending", "Lifestyle", "Travel", "Food", "Fitness"];

const exploreItems = [
  { image: post1, size: "large" },
  { image: user2, size: "small" },
  { image: post2, size: "small" },
  { image: heroModel, size: "small" },
  { image: user1, size: "small" },
  { image: post1, size: "large" },
];

export default function Explore() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 glass border-b border-border p-4">
        <div className="max-w-lg mx-auto space-y-4">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-primary" />
            <h1 className="text-xl font-bold text-foreground">Explore</h1>
          </div>
          
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search verified content..."
              className="w-full pl-12 pr-4 py-3 bg-muted rounded-2xl text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          {/* Categories */}
          <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
            {categories.map((category, index) => (
              <motion.button
                key={category}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors ${
                  index === 0
                    ? "gradient-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-muted/80"
                }`}
              >
                {category}
              </motion.button>
            ))}
          </div>
        </div>
      </header>

      {/* Grid */}
      <main className="max-w-lg mx-auto p-2">
        <div className="flex items-center gap-2 mb-4 px-2">
          <ShieldCheck className="w-5 h-5 text-verified" />
          <p className="text-sm text-muted-foreground">
            All content is AI-verified authentic
          </p>
        </div>

        <div className="grid grid-cols-3 gap-1">
          {exploreItems.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className={`relative cursor-pointer overflow-hidden rounded-lg ${
                item.size === "large" ? "col-span-2 row-span-2" : ""
              }`}
            >
              <img
                src={item.image}
                alt="Explore content"
                className={`w-full object-cover hover:scale-105 transition-transform duration-300 ${
                  item.size === "large" ? "aspect-square" : "aspect-square"
                }`}
              />
              <div className="absolute top-2 right-2">
                <VerifiedBadge size="sm" />
              </div>
            </motion.div>
          ))}
        </div>
      </main>
    </div>
  );
}
