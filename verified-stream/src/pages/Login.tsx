import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ShieldCheck, Chrome, Github, AlertCircle } from "lucide-react";
import { supabase } from "@/lib/supabase";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || "";
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || "";

export default function Login() {
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(
        (!supabaseUrl || !supabaseAnonKey)
            ? "Supabase configuration missing. Please check your .env file."
            : null
    );

    /**
     * Handle Google Login via Supabase Auth
     */
    const handleGoogleLogin = async () => {
        setLoading(true);
        setError(null);

        try {
            if (!supabase) {
                throw new Error("Supabase client not initialized. Check configuration.");
            }
            const { data, error } = await supabase.auth.signInWithOAuth({
                provider: "google",
                options: {
                    redirectTo: `${window.location.origin}/profile`,
                },
            });

            if (error) {
                throw error;
            }

            // OAuth redirect will happen automatically
        } catch (err: any) {
            setError(err.message || "Google login failed");
            setLoading(false);
        }
    };

    /**
     * Handle GitHub Login via Supabase Auth
     */
    const handleGitHubLogin = async () => {
        setLoading(true);
        setError(null);

        try {
            if (!supabase) {
                throw new Error("Supabase client not initialized. Check configuration.");
            }
            const { data, error } = await supabase.auth.signInWithOAuth({
                provider: "github",
                options: {
                    redirectTo: `${window.location.origin}/profile`,
                },
            });

            if (error) {
                throw error;
            }

            // OAuth redirect will happen automatically
        } catch (err: any) {
            setError(err.message || "GitHub login failed");
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted flex items-center justify-center p-4">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full max-w-md"
            >
                {/* Logo and Branding */}
                <div className="text-center mb-8">
                    <motion.div
                        initial={{ scale: 0.8 }}
                        animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 200 }}
                        className="inline-flex items-center justify-center p-4 bg-primary/10 rounded-3xl mb-4"
                    >
                        <ShieldCheck className="w-16 h-16 text-primary" />
                    </motion.div>
                    <h1 className="text-4xl font-bold text-foreground mb-2">
                        TrueFrame
                    </h1>
                    <p className="text-muted-foreground text-lg">
                        Upload only real content. Deepfakes are blocked.
                    </p>
                </div>

                {/* Login Card */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-card rounded-3xl shadow-2xl p-8 border border-border"
                >
                    <h2 className="text-2xl font-bold text-foreground mb-6 text-center">
                        Sign in to continue
                    </h2>

                    {/* Error Message */}
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mb-6 p-4 bg-destructive/10 border border-destructive/30 rounded-xl flex items-start gap-3"
                        >
                            <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-destructive">{error}</p>
                        </motion.div>
                    )}

                    {/* OAuth Buttons */}
                    <div className="space-y-4">
                        {/* Google Login Button */}
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={handleGoogleLogin}
                            disabled={loading}
                            className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-white hover:bg-gray-50 text-gray-900 rounded-xl font-semibold transition-colors border-2 border-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Chrome className="w-5 h-5" />
                            <span>Continue with Google</span>
                        </motion.button>

                        {/* GitHub Login Button */}
                        <motion.button
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            onClick={handleGitHubLogin}
                            disabled={loading}
                            className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-gray-900 hover:bg-gray-800 text-white rounded-xl font-semibold transition-colors border-2 border-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Github className="w-5 h-5" />
                            <span>Continue with GitHub</span>
                        </motion.button>
                    </div>

                    {/* Loading State */}
                    {loading && (
                        <div className="mt-6 text-center">
                            <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
                            <p className="text-sm text-muted-foreground mt-2">Redirecting to login...</p>
                        </div>
                    )}

                    {/* Trust Message */}
                    <div className="mt-8 p-4 bg-muted/50 rounded-xl">
                        <p className="text-xs text-muted-foreground text-center">
                            🔒 Only verified users can upload. All content is scanned for deepfakes.
                        </p>
                    </div>
                </motion.div>

                {/* Footer */}
                <p className="text-center text-sm text-muted-foreground mt-6">
                    By signing in, you agree to our terms and privacy policy
                </p>
            </motion.div>
        </div>
    );
}
