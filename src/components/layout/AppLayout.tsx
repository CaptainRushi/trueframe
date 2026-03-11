import { Outlet } from "react-router-dom";
import { BottomNav } from "./BottomNav";
import { SideNav } from "./SideNav";

export function AppLayout() {
  return (
    <div className="flex min-h-screen bg-background relative w-full overflow-x-hidden">
      {/* Sidebar for Desktop (md and above) */}
      <div className="hidden md:flex h-screen sticky top-0 shrink-0">
        <SideNav />
      </div>

      {/* Main Content Area */}
      <main className="flex-1 min-h-screen pb-20 md:pb-0 relative w-full">
        <Outlet />
      </main>

      {/* Bottom Nav for Mobile (below md) */}
      <div className="md:hidden">
        <BottomNav />
      </div>
    </div>
  );
}
