import { motion, AnimatePresence } from "framer-motion";
import { Copy, X, Link as LinkIcon, Share2, Smartphone } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

interface ShareModalProps {
    isOpen: boolean;
    onClose: () => void;
    post: {
        id: string;
        caption?: string;
        verification_status?: string;
    };
}

export function ShareModal({ isOpen, onClose, post }: ShareModalProps) {
    const [copied, setCopied] = useState(false);

    // Canonical Post URL
    const postUrl = `${window.location.origin}/post/${post.id}`;
    const encodedUrl = encodeURIComponent(postUrl);
    const encodedText = encodeURIComponent(`Verified content: ${post.caption ? post.caption.substring(0, 50) + "..." : "Check this out"}`);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(postUrl);
            setCopied(true);
            toast.success("Link copied to clipboard");
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            toast.error("Failed to copy link");
        }
    };

    const handleNativeShare = async () => {
        if (navigator.share) {
            try {
                await navigator.share({
                    title: "Verified Content via TrueFrame",
                    text: post.caption || "Verified Authentic Content",
                    url: postUrl
                });
                onClose();
            } catch (err) {
                // Ignore abort errors
            }
        } else {
            toast.error("System share not supported on this device");
        }
    };

    const shareLinks = [
        {
            name: "WhatsApp",
            icon: "💬", // Using emoji as icon placeholder to avoid specialized lib deps for MVP
            color: "bg-green-500",
            url: `https://wa.me/?text=${encodeURIComponent(`Verified post on TrueFrame: ${postUrl}`)}`
        },
        {
            name: "X (Twitter)",
            icon: "𝕏",
            color: "bg-black",
            url: `https://twitter.com/intent/tweet?text=${encodedText}&url=${encodedUrl}`
        },
        {
            name: "LinkedIn",
            icon: "in",
            color: "bg-blue-700",
            url: `https://www.linkedin.com/sharing/share-offsite/?url=${encodedUrl}`
        },
        {
            name: "Telegram",
            icon: "✈️",
            color: "bg-blue-400",
            url: `https://t.me/share/url?url=${encodedUrl}&text=${encodedText}`
        }
    ];

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={onClose}
                    className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                />
                <motion.div
                    initial={{ scale: 0.95, opacity: 0, y: 20 }}
                    animate={{ scale: 1, opacity: 1, y: 0 }}
                    exit={{ scale: 0.95, opacity: 0, y: 20 }}
                    className="relative w-full max-w-sm bg-card border border-border rounded-2xl shadow-2xl overflow-hidden p-6"
                >
                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center gap-2">
                            <Share2 className="w-5 h-5 text-primary" />
                            <h3 className="font-bold text-lg">Share Verified Post</h3>
                        </div>
                        <button onClick={onClose} className="p-1 hover:bg-muted rounded-full transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Copy Link Section (Primary) */}
                    <div
                        onClick={handleCopy}
                        className="flex items-center gap-3 p-3 bg-muted/50 hover:bg-muted border border-border rounded-xl cursor-pointer transition-colors mb-6 group"
                    >
                        <div className="p-2 bg-background rounded-full shadow-sm text-primary group-hover:scale-110 transition-transform">
                            <LinkIcon className="w-5 h-5" />
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium">Copy Link</p>
                            <p className="text-xs text-muted-foreground truncate">{postUrl}</p>
                        </div>
                        <div className="px-3 py-1 bg-primary/10 text-primary text-xs font-bold rounded-full">
                            {copied ? "COPIED" : "COPY"}
                        </div>
                    </div>

                    {/* Social Grid */}
                    <div className="grid grid-cols-4 gap-4 mb-6">
                        {shareLinks.map((link) => (
                            <a
                                key={link.name}
                                href={link.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex flex-col items-center gap-2 group"
                                onClick={onClose}
                            >
                                <div className={`w-12 h-12 flex items-center justify-center rounded-full text-white font-bold text-xl shadow-lg hover:scale-110 transition-transform ${link.color}`}>
                                    {link.icon}
                                </div>
                                <span className="text-xs text-muted-foreground font-medium">{link.name}</span>
                            </a>
                        ))}
                    </div>

                    {/* System Share (Mobile) */}
                    {navigator.share && (
                        <button
                            onClick={handleNativeShare}
                            className="w-full flex items-center justify-center gap-2 py-3 bg-muted hover:bg-muted/80 rounded-xl font-medium transition-colors"
                        >
                            <Smartphone className="w-4 h-4" />
                            Share via...
                        </button>
                    )}

                </motion.div>
            </div>
        </AnimatePresence>
    );
}
