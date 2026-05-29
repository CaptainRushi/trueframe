import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Heart, MessageCircle, UserPlus, ShieldCheck, ShieldAlert,
  Clock, Bell, AlertTriangle, Award, Flag, Loader2, CheckCircle2
} from "lucide-react";
import { supabase } from "@/lib/supabase";
import { BACKEND_URL } from "@/lib/api";
import { Link } from "react-router-dom";

interface Notification {
  id: string;
  type: string;
  title: string;
  message: string;
  is_read: boolean;
  created_at: string;
  related_post_id: string | null;
  related_user: {
    username: string;
    display_name: string;
    avatar_url: string;
  } | null;
}

interface DeepfakeAlert {
  id: string;
  title: string;
  description: string;
  severity: string;
  detection_count: number;
  created_at: string;
}

const typeConfig: Record<string, { icon: any; color: string; bg: string }> = {
  LIKE: { icon: Heart, color: "text-red-500", bg: "bg-red-500/10" },
  COMMENT: { icon: MessageCircle, color: "text-blue-500", bg: "bg-blue-500/10" },
  FOLLOW: { icon: UserPlus, color: "text-primary", bg: "bg-primary/10" },
  VERIFICATION_PASSED: { icon: ShieldCheck, color: "text-green-500", bg: "bg-green-500/10" },
  VERIFICATION_FAILED: { icon: ShieldAlert, color: "text-red-500", bg: "bg-red-500/10" },
  TRUST_CHANGE: { icon: ShieldCheck, color: "text-yellow-500", bg: "bg-yellow-500/10" },
  CREATOR_BADGE: { icon: Award, color: "text-primary", bg: "bg-primary/10" },
  FLAG_RESULT: { icon: Flag, color: "text-yellow-500", bg: "bg-yellow-500/10" },
  DEEPFAKE_ALERT: { icon: AlertTriangle, color: "text-red-500", bg: "bg-red-500/10" },
  SYSTEM: { icon: Bell, color: "text-muted-foreground", bg: "bg-muted/30" },
};

export default function Notifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [alerts, setAlerts] = useState<DeepfakeAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [unreadCount, setUnreadCount] = useState(0);
  const [activeTab, setActiveTab] = useState<'activity' | 'alerts'>('activity');

  useEffect(() => {
    fetchNotifications();
    fetchAlerts();
  }, []);

  const fetchNotifications = async () => {
    setLoading(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const res = await fetch(`${BACKEND_URL}/api/notifications`, {
        headers: { 'Authorization': `Bearer ${session.access_token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setNotifications(data.notifications || []);
        setUnreadCount(data.unreadCount || 0);
      }
    } catch (e) {
      console.error("Failed to fetch notifications", e);
    } finally {
      setLoading(false);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/notifications/alerts`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.alerts || []);
      }
    } catch (e) {
      console.error("Failed to fetch alerts", e);
    }
  };

  const markAllRead = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      await fetch(`${BACKEND_URL}/api/notifications/read`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${session.access_token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
      });

      setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
      setUnreadCount(0);
    } catch (e) {
      console.error("Failed to mark as read", e);
    }
  };

  const getTimeAgo = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    return `${days}d`;
  };

  const today = new Date().toDateString();
  const todayNotifications = notifications.filter(n => new Date(n.created_at).toDateString() === today);
  const earlierNotifications = notifications.filter(n => new Date(n.created_at).toDateString() !== today);

  return (
    <div className="min-h-screen bg-background pb-24 md:pb-6">
      {/* Header */}
      <header className="sticky top-0 z-40 glass border-b border-border p-4">
        <div className="max-w-lg mx-auto">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Bell className="w-6 h-6 text-primary" />
              <h1 className="text-xl font-bold text-foreground">Activity</h1>
              {unreadCount > 0 && (
                <span className="bg-primary text-primary-foreground text-xs font-bold px-2 py-0.5 rounded-full">
                  {unreadCount}
                </span>
              )}
            </div>
            {unreadCount > 0 && (
              <button onClick={markAllRead} className="text-xs font-bold text-primary">
                Mark all read
              </button>
            )}
          </div>

          {/* Tabs */}
          <div className="flex gap-2 border-b border-border -mx-4 px-4">
            <button
              onClick={() => setActiveTab('activity')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'activity' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground'
              }`}
            >
              Activity
            </button>
            <button
              onClick={() => setActiveTab('alerts')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
                activeTab === 'alerts' ? 'border-red-500 text-red-500' : 'border-transparent text-muted-foreground'
              }`}
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              Deepfake Alerts
              {alerts.length > 0 && (
                <span className="bg-red-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-full">
                  {alerts.length}
                </span>
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-lg mx-auto">
        {activeTab === 'activity' ? (
          <>
            {loading ? (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="w-6 h-6 text-primary animate-spin" />
              </div>
            ) : notifications.length === 0 ? (
              <div className="text-center py-20 px-8">
                <Bell className="w-12 h-12 text-muted-foreground/30 mx-auto mb-4" />
                <h2 className="text-lg font-bold mb-1">No activity yet</h2>
                <p className="text-sm text-muted-foreground">Your notifications will appear here</p>
              </div>
            ) : (
              <>
                {todayNotifications.length > 0 && (
                  <NotificationSection title="Today" notifications={todayNotifications} getTimeAgo={getTimeAgo} />
                )}
                {earlierNotifications.length > 0 && (
                  <NotificationSection title="Earlier" notifications={earlierNotifications} getTimeAgo={getTimeAgo} />
                )}
              </>
            )}
          </>
        ) : (
          <div className="p-4 space-y-4">
            {alerts.length === 0 ? (
              <div className="text-center py-20 px-8">
                <CheckCircle2 className="w-12 h-12 text-green-500/30 mx-auto mb-4" />
                <h2 className="text-lg font-bold mb-1">No active alerts</h2>
                <p className="text-sm text-muted-foreground">No widespread deepfake content has been detected</p>
              </div>
            ) : (
              alerts.map((alert) => (
                <motion.div
                  key={alert.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`p-4 rounded-2xl border ${
                    alert.severity === 'CRITICAL' ? 'bg-red-500/10 border-red-500/30' :
                    alert.severity === 'HIGH' ? 'bg-orange-500/10 border-orange-500/30' :
                    'bg-yellow-500/10 border-yellow-500/30'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <AlertTriangle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                      alert.severity === 'CRITICAL' ? 'text-red-500' :
                      alert.severity === 'HIGH' ? 'text-orange-500' : 'text-yellow-500'
                    }`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <p className="text-sm font-bold">{alert.title}</p>
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                          alert.severity === 'CRITICAL' ? 'bg-red-500 text-white' :
                          alert.severity === 'HIGH' ? 'bg-orange-500 text-white' :
                          'bg-yellow-500 text-black'
                        }`}>
                          {alert.severity}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">{alert.description}</p>
                      <div className="flex items-center gap-3 mt-2 text-[10px] text-muted-foreground">
                        <span>Detected {alert.detection_count}x</span>
                        <span>{new Date(alert.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function NotificationSection({ title, notifications, getTimeAgo }: {
  title: string;
  notifications: Notification[];
  getTimeAgo: (d: string) => string;
}) {
  return (
    <div className="p-4">
      <h2 className="text-sm font-semibold text-muted-foreground mb-4">{title}</h2>
      <div className="space-y-2">
        {notifications.map((notification, index) => {
          const config = typeConfig[notification.type] || typeConfig.SYSTEM;
          const Icon = config.icon;

          return (
            <motion.div
              key={notification.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={`flex items-center gap-4 p-4 rounded-2xl shadow-soft transition-colors ${
                notification.is_read ? 'bg-card' : 'bg-primary/5 border border-primary/10'
              }`}
            >
              {notification.related_user?.avatar_url ? (
                <div className="relative flex-shrink-0">
                  <img
                    src={notification.related_user.avatar_url}
                    alt=""
                    className="w-12 h-12 rounded-full object-cover"
                  />
                  <div className={`absolute -bottom-1 -right-1 p-1 rounded-full ${config.bg}`}>
                    <Icon className={`w-3 h-3 ${config.color}`} />
                  </div>
                </div>
              ) : (
                <div className={`p-3 rounded-full flex-shrink-0 ${config.bg}`}>
                  <Icon className={`w-5 h-5 ${config.color}`} />
                </div>
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-foreground">
                  {notification.related_user && (
                    <Link
                      to={`/profile/${notification.related_user.username}`}
                      className="font-semibold hover:underline"
                    >
                      {notification.related_user.display_name || notification.related_user.username}
                    </Link>
                  )}
                  {notification.related_user ? ' ' : ''}
                  {notification.message || notification.title}
                </p>
                <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
                  <Clock className="w-3 h-3" />
                  {getTimeAgo(notification.created_at)}
                </p>
              </div>
              {!notification.is_read && (
                <div className="w-2 h-2 bg-primary rounded-full flex-shrink-0" />
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
