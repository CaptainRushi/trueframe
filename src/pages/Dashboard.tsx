import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { supabase } from "@/lib/supabase";
import { useToast } from "@/components/ui/use-toast";
import {
    ShieldCheck,
    ShieldAlert,
    AlertCircle,
    BarChart3,
    History,
    TrendingUp,
    TrendingDown,
    Clock,
    Loader2,
    CheckCircle2,
    XCircle,
    AlertTriangle,
    Info,
    ShieldX
} from "lucide-react";

interface DashboardStats {
    totalUploads: number;
    verifiedUploads: number;
    rejectedUploads: number;
    realPercentage: number;
    fakePercentage: number;
    status: "TRUSTED" | "AT_RISK" | "RESTRICTED" | "NEW_USER";
}

interface UploadHistoryItem {
    id: string;
    created_at: string;
    verdict: "REAL" | "FAKE" | "REJECTED";
    score: number;
    reason: string | null;
    media_type: "image" | "video";
}

import { BACKEND_URL } from "@/lib/api";

export default function Dashboard() {
    const [stats, setStats] = useState<DashboardStats | null>(null);
    const [history, setHistory] = useState<UploadHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const { toast } = useToast();

    useEffect(() => {
        fetchDashboardData();
    }, []);

    const fetchDashboardData = async () => {
        setLoading(true);
        setError(null);

        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session) throw new Error("Not authenticated");

            // Fetch stats
            const statsResponse = await fetch(
                `${BACKEND_URL}/api/dashboard/stats`,
                {
                    headers: { 'Authorization': `Bearer ${session.access_token}` }
                }
            );

            if (!statsResponse.ok) {
                throw new Error("Failed to fetch dashboard statistics");
            }

            const statsData = await statsResponse.json();
            setStats(statsData);

            // Fetch history
            const historyResponse = await fetch(
                `${BACKEND_URL}/api/dashboard/history`,
                {
                    headers: { 'Authorization': `Bearer ${session.access_token}` }
                }
            );

            if (!historyResponse.ok) {
                throw new Error("Failed to fetch upload history");
            }

            const historyData = await historyResponse.json();
            setHistory(historyData.history || []);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error occurred");
            console.error("Dashboard fetch error:", err);
        } finally {
            setLoading(false);
        }
    };

    const getStatusConfig = (status: DashboardStats["status"]) => {
        switch (status) {
            case "TRUSTED":
                return {
                    icon: ShieldCheck,
                    color: "text-green-500",
                    bgColor: "bg-green-500/10",
                    borderColor: "border-green-500/30",
                    label: "Trusted Creator",
                    description: "Excellent track record. Keep up the great work!",
                };
            case "AT_RISK":
                return {
                    icon: ShieldAlert,
                    color: "text-yellow-500",
                    bgColor: "bg-yellow-500/10",
                    borderColor: "border-yellow-500/30",
                    label: "At Risk",
                    description: "Higher than average rejection rate. Be careful.",
                };
            case "RESTRICTED":
                return {
                    icon: ShieldX,
                    color: "text-red-500",
                    bgColor: "bg-red-500/10",
                    borderColor: "border-red-500/30",
                    label: "Restricted",
                    description: "Too many fake attempts. Exposure limited.",
                };
            default:
                return {
                    icon: Info,
                    color: "text-blue-500",
                    bgColor: "bg-blue-500/10",
                    borderColor: "border-blue-500/30",
                    label: "New Creator",
                    description: "Establish your trust by uploading real content.",
                };
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <div className="flex flex-col items-center gap-4">
                    <Loader2 className="w-10 h-10 text-primary animate-spin" />
                    <p className="text-muted-foreground animate-pulse">Computing Trust Data...</p>
                </div>
            </div>
        );
    }

    const statusConfig = stats ? getStatusConfig(stats.status) : getStatusConfig("NEW_USER");

    return (
        <div className="min-h-screen bg-background pb-20">
            <div className="max-w-4xl mx-auto p-6 md:p-8">
                {/* Header */}
                <header className="mb-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div>
                        <h1 className="text-4xl font-black tracking-tight mb-2">My Trust Dashboard</h1>
                        <p className="text-muted-foreground uppercase text-xs font-bold tracking-widest">
                            Real-time verification metrics & historical logs
                        </p>
                    </div>

                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className={`${statusConfig.bgColor} ${statusConfig.borderColor} border rounded-2xl p-4 flex items-center gap-4 max-w-sm`}
                    >
                        <div className={`p-3 rounded-xl bg-background shadow-sm ${statusConfig.color}`}>
                            <statusConfig.icon className="w-6 h-6" />
                        </div>
                        <div>
                            <p className={`text-sm font-black uppercase tracking-tight ${statusConfig.color}`}>{statusConfig.label}</p>
                            <p className="text-[10px] text-muted-foreground leading-tight">{statusConfig.description}</p>
                        </div>
                    </motion.div>
                </header>

                {error && (
                    <div className="mb-8 p-4 bg-destructive/10 border border-destructive/20 text-destructive rounded-xl flex items-center gap-3">
                        <AlertCircle className="w-5 h-5 flex-shrink-0" />
                        <p className="text-sm font-medium">{error}</p>
                    </div>
                )}

                {/* Main Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
                    <StatCard
                        title="Integrity Score"
                        value={`${stats?.realPercentage || 0}%`}
                        subtitle="Real content ratio"
                        icon={ShieldCheck}
                        trend={stats?.realPercentage && stats.realPercentage > 90 ? "UP" : "STABLE"}
                        color="primary"
                    />
                    <StatCard
                        title="Verified Real"
                        value={stats?.verifiedUploads.toString() || "0"}
                        subtitle="Successful uploads"
                        icon={CheckCircle2}
                        color="green"
                    />
                    <StatCard
                        title="Fake Uploads"
                        value={stats?.rejectedUploads.toString() || "0"}
                        subtitle="Deepfakes detected"
                        icon={ShieldAlert}
                        trend={stats?.rejectedUploads && stats.rejectedUploads > 0 ? "DOWN" : "STABLE"}
                        color="red"
                    />
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
                    {/* Distribution Chart / Visualization Area */}
                    <div className="lg:col-span-3 space-y-8">
                        <section className="bg-card rounded-[2rem] border border-border p-8">
                            <h3 className="text-xl font-black mb-6 flex items-center gap-2">
                                <BarChart3 className="w-5 h-5 text-primary" />
                                Content Authenticity
                            </h3>

                            <div className="space-y-8">
                                <div className="space-y-2">
                                    <div className="flex justify-between items-end">
                                        <p className="text-sm font-bold uppercase text-muted-foreground tracking-widest">Real Content ({stats?.verifiedUploads || 0})</p>
                                        <p className="text-2xl font-black">{stats?.realPercentage}%</p>
                                    </div>
                                    <div className="h-4 bg-muted rounded-full overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${stats?.realPercentage}%` }}
                                            className="h-full bg-green-500"
                                        />
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <div className="flex justify-between items-end">
                                        <p className="text-sm font-bold uppercase text-muted-foreground tracking-widest">Fake Content ({stats?.rejectedUploads || 0})</p>
                                        <p className="text-2xl font-black">{stats?.fakePercentage}%</p>
                                    </div>
                                    <div className="h-4 bg-muted rounded-full overflow-hidden">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${stats?.fakePercentage}%` }}
                                            className="h-full bg-destructive"
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="mt-8 pt-8 border-t border-border grid grid-cols-3 gap-4">
                                <div className="text-center p-4 bg-muted/30 rounded-2xl">
                                    <p className="text-2xl font-black">{stats?.totalUploads}</p>
                                    <p className="text-[10px] font-bold uppercase text-muted-foreground">Attempts</p>
                                </div>
                                <div className="text-center p-4 bg-muted/30 rounded-2xl">
                                    <p className="text-2xl font-black text-green-500">{stats?.verifiedUploads}</p>
                                    <p className="text-[10px] font-bold uppercase text-muted-foreground">Real</p>
                                </div>
                                <div className="text-center p-4 bg-muted/30 rounded-2xl">
                                    <p className="text-2xl font-black text-destructive">{stats?.rejectedUploads}</p>
                                    <p className="text-[10px] font-bold uppercase text-muted-foreground">Fake</p>
                                </div>
                            </div>
                        </section>
                    </div>

                    {/* History Sidebar */}
                    <div className="lg:col-span-2 space-y-6">
                        <section className="bg-card rounded-[2rem] border border-border p-6 h-full flex flex-col">
                            <h3 className="text-xl font-black mb-6 flex items-center gap-2">
                                <History className="w-5 h-5 text-primary" />
                                Recent Logs
                            </h3>

                            <div className="space-y-4 flex-grow">
                                {history.length > 0 ? (
                                    history.map((item) => (
                                        <div
                                            key={item.id}
                                            className="p-4 rounded-2xl border border-border/50 bg-muted/10 hover:bg-muted/30 transition-colors flex items-center gap-4"
                                        >
                                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${item.verdict === 'REAL' ? 'bg-green-500/10 text-green-500' : 'bg-destructive/10 text-destructive'
                                                }`}>
                                                {item.verdict === 'REAL' ? <CheckCircle2 className="w-5 h-5" /> : <ShieldX className="w-5 h-5" />}
                                            </div>
                                            <div className="min-w-0 flex-grow">
                                                <div className="flex justify-between items-start mb-0.5">
                                                    <p className="text-sm font-black uppercase tracking-tight truncate">
                                                        {item.verdict === 'REAL' ? 'Approved' : 'Blocked'}
                                                    </p>
                                                    <p className="text-[10px] font-medium text-muted-foreground whitespace-nowrap">
                                                        {new Date(item.created_at).toLocaleDateString()}
                                                    </p>
                                                </div>
                                                <p className="text-[10px] text-muted-foreground truncate italic">
                                                    {item.reason || `Score: ${item.score.toFixed(4)}`}
                                                </p>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="flex flex-col items-center justify-center py-20 text-center opacity-40">
                                        <Clock className="w-10 h-10 mb-2" />
                                        <p className="text-sm font-bold">No logs yet</p>
                                    </div>
                                )}
                            </div>

                            <button className="mt-6 w-full py-4 text-xs font-black uppercase tracking-widest text-muted-foreground hover:text-foreground transition-colors">
                                View Full Audit Trail
                            </button>
                        </section>
                    </div>
                </div>
            </div>
        </div>
    );
}

function StatCard({ title, value, subtitle, icon: Icon, trend, color }: any) {
    const colorClasses: any = {
        primary: "text-primary bg-primary/5",
        green: "text-green-500 bg-green-500/5",
        red: "text-destructive bg-destructive/5"
    };

    return (
        <motion.div
            whileHover={{ y: -4 }}
            className="bg-card rounded-[2rem] border border-border p-6 shadow-sm"
        >
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center mb-6 ${colorClasses[color]}`}>
                <Icon className="w-6 h-6" />
            </div>
            <div className="space-y-1">
                <p className="text-xs font-bold uppercase text-muted-foreground tracking-widest">{title}</p>
                <div className="flex items-end gap-2">
                    <p className="text-3xl font-black">{value}</p>
                    {trend && (
                        <div className={`mb-1 flex items-center text-[10px] font-bold ${trend === 'UP' ? 'text-green-500' : 'text-destructive'}`}>
                            {trend === 'UP' ? <TrendingUp className="w-3 h-3 mr-0.5" /> : <TrendingDown className="w-3 h-3 mr-0.5" />}
                            {trend}
                        </div>
                    )}
                </div>
                <p className="text-[10px] text-muted-foreground italic">{subtitle}</p>
            </div>
        </motion.div>
    );
}
