import { Home, Search, PlusSquare, Bell, User, ShieldCheck } from "lucide-react";
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

export const SideNav = memo(function SideNav() {
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
    <nav className="w-64 h-full glass border-r border-border p-6 flex flex-col pt-10">
      {/* Brand logo */}
      <div className="flex items-center gap-3 mb-10 px-4">
        <ShieldCheck className="w-8 h-8 text-primary" />
        <span className="text-2xl font-black text-foreground">TrueFrame</span>
      </div>

      <div className="flex flex-col gap-2 flex-1">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          const isUpload = item.label === "Upload";
          const isNotifications = item.label === "Activity";

          return (
            <Link
              key={item.path}
              to={item.path}
              className={`relative flex items-center gap-4 px-4 py-3 rounded-2xl transition-all group ${
                isActive ? "bg-primary/10 text-primary" : "text-muted-foreground hover:bg-muted hover:text-foreground"
              }`}
            >
              <div className="relative">
                <item.icon
                  className={`w-6 h-6 transition-transform group-hover:scale-110 ${
                    isUpload && "text-primary border-2 border-primary rounded-lg p-0.5"
                  }`}
                  fill={isActive && !isUpload ? "currentColor" : "none"}
                />
                {isNotifications && unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1.5 bg-red-500 text-white text-[10px] font-bold min-w-[18px] h-[18px] flex items-center justify-center rounded-full px-1 border-2 border-background">
                    {unreadCount > 9 ? '9+' : unreadCount}
                  </span>
                )}
              </div>
              <span className={`text-lg font-bold ${isActive ? "text-primary" : "text-foreground"}`}>
                {item.label}
              </span>
              
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1.5 h-8 bg-primary rounded-r-full" />
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
});
