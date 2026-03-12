import { Outlet } from "react-router-dom";
import { BottomNav } from "./BottomNav";
import { SideNav } from "./SideNav";

export function AppLayout() {
  return (
    <div className="min-h-screen bg-background relative w-full overflow-x-hidden flex justify-center">
      
      {/* Container: centers on ultra-wide screens, uses standard flex-row layout */}
      <div className="w-full max-w-[1200px] mx-auto flex flex-row relative justify-center">
        
        {/* Left Sidebar - sticky so it doesn't scroll, stays in flow */}
        <div className="hidden md:flex sticky top-0 h-screen w-[80px] lg:w-[275px] shrink-0 z-40">
          <SideNav />
        </div>

        {/* Center Main Content */}
        <main className="flex-1 min-w-0 min-h-screen max-w-[600px] w-full pb-20 md:pb-0 relative border-x border-border/50 bg-background">
          <Outlet />
        </main>

        {/* Right Sidebar Placeholder (Balance for desktop/XL screens) */}
        <div className="hidden xl:block sticky top-0 h-screen w-[320px] shrink-0 p-6 pt-10">
          <div className="bg-muted/30 rounded-2xl border border-border p-5 shadow-sm">
            <h3 className="font-bold mb-4 text-foreground/90">Trending in Trust</h3>
            <p className="text-xs text-muted-foreground italic mb-6">Coming soon...</p>
            <div className="space-y-4">
              {[1,2,3].map(i => (
                <div key={i} className="flex gap-3 items-center opacity-50">
                  <div className="w-10 h-10 rounded-full bg-border animate-pulse shrink-0" />
                  <div className="space-y-2 flex-1">
                    <div className="w-3/4 h-3 bg-border rounded animate-pulse" />
                    <div className="w-1/2 h-2 bg-border/50 rounded animate-pulse" />
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-8 pt-6 border-t border-border/50">
              <p className="text-[10px] text-muted-foreground uppercase font-black tracking-widest mb-2">Platform Stats</p>
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>Verified Content</span>
                <span className="font-bold text-foreground">99.9%</span>
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Bottom Nav for Mobile (below md) */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-background/95 backdrop-blur border-t border-border">
        <BottomNav />
      </div>

    </div>
  );
}
