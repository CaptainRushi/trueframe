import { motion } from "framer-motion";
import { Heart, MessageCircle, UserPlus, ShieldCheck, Clock } from "lucide-react";
import user1 from "@/assets/user-1.jpg";
import user2 from "@/assets/user-2.jpg";
import user3 from "@/assets/user-3.jpg";

const notifications = [
  {
    id: 1,
    type: "like",
    user: { name: "Alana Maesya", avatar: user1 },
    message: "liked your photo",
    time: "2m ago",
    isVerified: true,
  },
  {
    id: 2,
    type: "follow",
    user: { name: "Marcus Cole", avatar: user3 },
    message: "started following you",
    time: "15m ago",
    isVerified: true,
  },
  {
    id: 3,
    type: "comment",
    user: { name: "Maya Chen", avatar: user2 },
    message: "commented: \"Amazing shot! 📸\"",
    time: "1h ago",
    isVerified: true,
  },
  {
    id: 4,
    type: "verification",
    user: null,
    message: "Your latest post was verified as authentic",
    time: "2h ago",
    isVerified: true,
  },
  {
    id: 5,
    type: "like",
    user: { name: "Sofia Rodriguez", avatar: user1 },
    message: "and 23 others liked your photo",
    time: "3h ago",
    isVerified: true,
  },
];

const typeIcons = {
  like: Heart,
  comment: MessageCircle,
  follow: UserPlus,
  verification: ShieldCheck,
};

const typeColors = {
  like: "text-destructive bg-destructive/10",
  comment: "text-primary bg-primary/10",
  follow: "text-primary bg-primary/10",
  verification: "text-verified bg-verified/10",
};

export default function Notifications() {
  return (
    <div className="min-h-screen bg-background pb-24">
      {/* Header */}
      <header className="sticky top-0 z-40 glass border-b border-border p-4">
        <div className="max-w-lg mx-auto flex items-center gap-2">
          <Heart className="w-6 h-6 text-primary" />
          <h1 className="text-xl font-bold text-foreground">Activity</h1>
        </div>
      </header>

      <main className="max-w-lg mx-auto">
        {/* Today Section */}
        <div className="p-4">
          <h2 className="text-sm font-semibold text-muted-foreground mb-4">Today</h2>
          <div className="space-y-2">
            {notifications.slice(0, 3).map((notification, index) => {
              const Icon = typeIcons[notification.type as keyof typeof typeIcons];
              const colorClass = typeColors[notification.type as keyof typeof typeColors];

              return (
                <motion.div
                  key={notification.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center gap-4 p-4 bg-card rounded-2xl shadow-soft"
                >
                  {notification.user ? (
                    <div className="relative">
                      <img
                        src={notification.user.avatar}
                        alt={notification.user.name}
                        className="w-12 h-12 rounded-full object-cover"
                      />
                      <div className={`absolute -bottom-1 -right-1 p-1 rounded-full ${colorClass}`}>
                        <Icon className="w-3 h-3" />
                      </div>
                    </div>
                  ) : (
                    <div className={`p-3 rounded-full ${colorClass}`}>
                      <Icon className="w-6 h-6" />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-foreground">
                      {notification.user && (
                        <span className="font-semibold">{notification.user.name} </span>
                      )}
                      {notification.message}
                    </p>
                    <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                      <Clock className="w-3 h-3" />
                      {notification.time}
                    </p>
                  </div>
                  {notification.type === "follow" && (
                    <button className="px-4 py-2 gradient-primary text-primary-foreground rounded-xl text-sm font-medium">
                      Follow
                    </button>
                  )}
                </motion.div>
              );
            })}
          </div>
        </div>

        {/* Earlier Section */}
        <div className="p-4">
          <h2 className="text-sm font-semibold text-muted-foreground mb-4">Earlier</h2>
          <div className="space-y-2">
            {notifications.slice(3).map((notification, index) => {
              const Icon = typeIcons[notification.type as keyof typeof typeIcons];
              const colorClass = typeColors[notification.type as keyof typeof typeColors];

              return (
                <motion.div
                  key={notification.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="flex items-center gap-4 p-4 bg-card rounded-2xl shadow-soft"
                >
                  {notification.user ? (
                    <div className="relative">
                      <img
                        src={notification.user.avatar}
                        alt={notification.user.name}
                        className="w-12 h-12 rounded-full object-cover"
                      />
                      <div className={`absolute -bottom-1 -right-1 p-1 rounded-full ${colorClass}`}>
                        <Icon className="w-3 h-3" />
                      </div>
                    </div>
                  ) : (
                    <div className={`p-3 rounded-full ${colorClass}`}>
                      <Icon className="w-6 h-6" />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-foreground">
                      {notification.user && (
                        <span className="font-semibold">{notification.user.name} </span>
                      )}
                      {notification.message}
                    </p>
                    <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                      <Clock className="w-3 h-3" />
                      {notification.time}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </main>
    </div>
  );
}
