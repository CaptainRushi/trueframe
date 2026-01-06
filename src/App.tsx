import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import Landing from "./pages/Landing";
import Feed from "./pages/Feed";
import Explore from "./pages/Explore";
import Upload from "./pages/Upload";
import Profile from "./pages/Profile";
import Notifications from "./pages/Notifications";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";
import PostPage from "./pages/PostPage";
import { ProtectedRoute } from "./components/auth/ProtectedRoute";
import NotFound from "./pages/NotFound";
import { useAutoReload } from "./hooks/useAutoReload";

const queryClient = new QueryClient();

const App = () => {
  useAutoReload();
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
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
              </Route>
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
