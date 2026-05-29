import { Home, Search, PlusSquare, Bell, User } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useState, useEffect, useCallback, memo } from "react";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";

const navItems = [
  { icon: Home, label: "Home", path: "/feed" },
  { icon: Search, label: "Explore", path: "/explore" },
  { icon: PlusSquare, label: "Upload", path: "/upload" },
  { icon: Bell, label: "Activity", path: "/notifications" },
  { icon: User, label: "Profile", path: "/profile" },
];

export const BottomNav = memo(function BottomNav() {
  const location = useLocation();
  const [unreadCount, setUnreadCount] = useState(0);

  const fetchUnreadCount = useCallback(async () => {
    try {
      if (!supabase) return;
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/notifications/unread-count`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setUnreadCount(data.count || 0);
      }
    } catch (e) {
      // Silently fail
    }
  }, []);

  useEffect(() => {
    fetchUnreadCount();
    // Reduced polling: 60s instead of 30s
    const interval = setInterval(fetchUnreadCount, 60000);
    return () => clearInterval(interval);
  }, [fetchUnreadCount]);

  return (
    <nav className="glass border-t border-border safe-bottom">
      <div className="flex items-center justify-around py-2 px-4 max-w-lg mx-auto">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const isUpload = item.label === "Upload";
          const isNotifications = item.label === "Activity";

          return (
            <Link
              key={item.path}
              to={item.path}
              className="relative flex flex-col items-center p-2"
            >
              {isUpload ? (
                <div className="gradient-primary p-3 rounded-2xl shadow-glow active:scale-95 transition-transform">
                  <item.icon className="w-6 h-6 text-primary-foreground" />
                </div>
              ) : (
                <>
                  <div className="relative active:scale-95 transition-transform">
                    <item.icon
                      className={`w-6 h-6 transition-colors ${isActive ? "text-primary" : "text-muted-foreground"
                        }`}
                      fill={isActive ? "currentColor" : "none"}
                    />
                    {isNotifications && unreadCount > 0 && (
                      <span className="absolute -top-1 -right-1.5 bg-red-500 text-white text-[9px] font-bold min-w-[16px] h-4 flex items-center justify-center rounded-full px-0.5">
                        {unreadCount > 9 ? '9+' : unreadCount}
                      </span>
                    )}
                  </div>
                  {isActive && (
                    <div className="absolute -bottom-1 w-1 h-1 rounded-full bg-primary" />
                  )}
                </>
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
});
