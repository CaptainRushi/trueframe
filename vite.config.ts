import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // Target modern browsers for smaller output
    target: 'es2020',
    // Enable CSS code splitting
    cssCodeSplit: true,
    // Optimize chunk size
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunk for framework essentials
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          // Separate chunk for heavy UI libraries
          'ui-vendor': ['framer-motion', 'lucide-react', 'recharts'],
          // Separate chunk for data layer
          'data-vendor': ['@supabase/supabase-js', '@tanstack/react-query'],
        },
      },
    },
    // Increase chunk size warning limit
    chunkSizeWarningLimit: 500,
  },
}));
