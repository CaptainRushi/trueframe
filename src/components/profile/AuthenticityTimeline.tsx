import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, XCircle, Edit3, BarChart3 } from "lucide-react";
import { BACKEND_URL } from "@/lib/api";

interface TimelineData {
  period: string;
  totalPosts: number;
  authentic: number;
  edited: number;
  blocked: number;
  authenticityRate: number;
  dailyBreakdown: { date: string; authentic: number; blocked: number }[];
}

interface AuthenticityTimelineProps {
  username: string;
}

export function AuthenticityTimeline({ username }: AuthenticityTimelineProps) {
  const [data, setData] = useState<TimelineData | null>(null);
  const [days, setDays] = useState(30);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTimeline();
  }, [username, days]);

  const fetchTimeline = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/transparency/timeline/${username}?days=${days}`);
      if (res.ok) setData(await res.json());
    } catch (e) {
      console.error("Failed to fetch timeline", e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return null;
  if (!data || data.totalPosts === 0) return null;

  return (
    <div className="bg-card rounded-2xl border border-border p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-black uppercase tracking-widest">Authenticity Timeline</h3>
        </div>
        <div className="flex gap-1">
          {[7, 30, 90].map((d) => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className={`px-2 py-1 text-[10px] font-bold rounded-lg transition-colors ${
                days === d ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
              }`}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center p-3 bg-green-500/5 rounded-xl border border-green-500/10">
          <div className="flex items-center justify-center gap-1 mb-1">
            <CheckCircle2 className="w-3 h-3 text-green-500" />
            <p className="text-xs font-bold text-green-500">Authentic</p>
          </div>
          <p className="text-xl font-black">{data.authentic}</p>
        </div>
        <div className="text-center p-3 bg-yellow-500/5 rounded-xl border border-yellow-500/10">
          <div className="flex items-center justify-center gap-1 mb-1">
            <Edit3 className="w-3 h-3 text-yellow-500" />
            <p className="text-xs font-bold text-yellow-500">Edited</p>
          </div>
          <p className="text-xl font-black">{data.edited}</p>
        </div>
        <div className="text-center p-3 bg-red-500/5 rounded-xl border border-red-500/10">
          <div className="flex items-center justify-center gap-1 mb-1">
            <XCircle className="w-3 h-3 text-red-500" />
            <p className="text-xs font-bold text-red-500">Blocked</p>
          </div>
          <p className="text-xl font-black">{data.blocked}</p>
        </div>
      </div>

      {/* Authenticity Rate Bar */}
      <div>
        <div className="flex justify-between items-center mb-1.5">
          <p className="text-[10px] font-bold text-muted-foreground uppercase">Authenticity Rate</p>
          <p className={`text-sm font-black ${data.authenticityRate >= 90 ? 'text-green-500' : data.authenticityRate >= 70 ? 'text-yellow-500' : 'text-red-500'}`}>
            {data.authenticityRate}%
          </p>
        </div>
        <div className="h-2 bg-muted rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${data.authenticityRate}%` }}
            transition={{ duration: 1 }}
            className={`h-full rounded-full ${
              data.authenticityRate >= 90 ? 'bg-green-500' :
              data.authenticityRate >= 70 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
          />
        </div>
      </div>

      {/* Mini Daily Chart */}
      {data.dailyBreakdown.length > 1 && (
        <div className="flex items-end gap-0.5 h-12 mt-2">
          {data.dailyBreakdown.slice(-14).map((day, i) => {
            const total = day.authentic + day.blocked;
            const height = total > 0 ? Math.max(4, (total / Math.max(...data.dailyBreakdown.map(d => d.authentic + d.blocked))) * 48) : 4;
            const greenPct = total > 0 ? (day.authentic / total) * 100 : 100;
            return (
              <div key={i} className="flex-1 flex flex-col justify-end" title={`${day.date}: ${day.authentic} real, ${day.blocked} blocked`}>
                <div
                  className="rounded-sm overflow-hidden"
                  style={{ height: `${height}px` }}
                >
                  <div className="bg-green-500" style={{ height: `${greenPct}%` }} />
                  <div className="bg-red-500" style={{ height: `${100 - greenPct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      <p className="text-[10px] text-muted-foreground text-center">{data.period} · {data.totalPosts} total uploads</p>
    </div>
  );
}
