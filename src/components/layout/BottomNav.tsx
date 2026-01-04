import { Home, Search, PlusSquare, Heart, User } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";

const navItems = [
  { icon: Home, label: "Home", path: "/feed" },
  { icon: Search, label: "Explore", path: "/explore" },
  { icon: PlusSquare, label: "Upload", path: "/upload" },
  { icon: Heart, label: "Activity", path: "/notifications" },
  { icon: User, label: "Profile", path: "/profile" },
];

export function BottomNav() {
  const location = useLocation();

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 glass border-t border-border">
      <div className="flex items-center justify-around py-2 px-4 max-w-lg mx-auto">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const isUpload = item.label === "Upload";

          return (
            <Link
              key={item.path}
              to={item.path}
              className="relative flex flex-col items-center p-2"
            >
              {isUpload ? (
                <motion.div
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  className="gradient-primary p-3 rounded-2xl shadow-glow"
                >
                  <item.icon className="w-6 h-6 text-primary-foreground" />
                </motion.div>
              ) : (
                <>
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <item.icon
                      className={`w-6 h-6 transition-colors ${
                        isActive ? "text-primary" : "text-muted-foreground"
                      }`}
                      fill={isActive ? "currentColor" : "none"}
                    />
                  </motion.div>
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute -bottom-1 w-1 h-1 rounded-full bg-primary"
                    />
                  )}
                </>
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
