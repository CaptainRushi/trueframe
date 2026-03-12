import { Outlet } from "react-router-dom";
import { BottomNav } from "./BottomNav";
import { SideNav } from "./SideNav";

export function AppLayout() {
  return (
    <div className="min-h-screen bg-background relative w-full overflow-x-hidden">
      {/* Sidebar for Desktop (md and above) - Fixed position */}
      <div className="hidden md:flex fixed top-0 left-0 h-screen w-64 z-40 shrink-0">
        <SideNav />
      </div>

      {/* Main Content Area - offset by sidebar width on desktop */}
      <main className="flex-1 min-h-screen pb-20 md:pb-0 md:ml-64 relative w-full">
        <Outlet />
      </main>

      {/* Bottom Nav for Mobile (below md) */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 z-40">
        <BottomNav />
      </div>
    </div>
  );
}
