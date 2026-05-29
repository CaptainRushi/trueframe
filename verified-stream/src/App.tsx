import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import { ProtectedRoute } from "./components/auth/ProtectedRoute";
import { useAutoReload } from "./hooks/useAutoReload";
import { lazy, Suspense } from "react";
import { Loader2 } from "lucide-react";

// Lazy load all page-level components for code splitting
const Landing = lazy(() => import("./pages/Landing"));
const Feed = lazy(() => import("./pages/Feed"));
const Explore = lazy(() => import("./pages/Explore"));
const Upload = lazy(() => import("./pages/Upload"));
const Profile = lazy(() => import("./pages/Profile"));
const Notifications = lazy(() => import("./pages/Notifications"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Login = lazy(() => import("./pages/Login"));
const PostPage = lazy(() => import("./pages/PostPage"));
const CreatorProgram = lazy(() => import("./pages/CreatorProgram"));
const VerifyIdentity = lazy(() => import("./pages/VerifyIdentity"));
const Moderation = lazy(() => import("./pages/Moderation"));
const NotFound = lazy(() => import("./pages/NotFound"));

// Lightweight loading fallback
function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[50vh]">
      <Loader2 className="w-8 h-8 text-primary animate-spin" />
    </div>
  );
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 2, // 2 minutes
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const App = () => {
  useAutoReload();
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<Login />} />
              <Route path="/post/:postId" element={<PostPage />} />
              <Route element={<ProtectedRoute />}>
                <Route element={<AppLayout />}>
                  <Route path="/feed" element={<Feed />} />
                  <Route path="/explore" element={<Explore />} />
                  <Route path="/upload" element={<Upload />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route path="/profile/:username?" element={<Profile />} />
                  <Route path="/notifications" element={<Notifications />} />
                  <Route path="/creator" element={<CreatorProgram />} />
                  <Route path="/verify" element={<VerifyIdentity />} />
                  <Route path="/moderation" element={<Moderation />} />
                </Route>
              </Route>
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
