import { useEffect } from "react";
import { Outlet, Link, useLocation } from "react-router-dom";
import { useUrlState } from "@/hooks/use-url-state";
import { useSetWorkspaceId, useWorkspaceId } from "@/context/workspace";
import { DateRangeSelector } from "./DateRangeSelector";
import { WorkspaceSwitcher } from "./WorkspaceSwitcher";
import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";

export function DashboardShell() {
  const [urlWsId, setUrlWsId] = useUrlState("workspace_id", "");
  const globalWsId = useWorkspaceId();
  const setGlobalWsId = useSetWorkspaceId();
  const [daysStr, setDays] = useUrlState("days", "30");

  // Sync: URL → global context (URL is source of truth when present)
  useEffect(() => {
    if (urlWsId && urlWsId !== globalWsId) {
      setGlobalWsId(urlWsId);
    }
  }, [urlWsId, globalWsId, setGlobalWsId]);

  // On mount: if no URL param but global has one, populate URL
  useEffect(() => {
    if (!urlWsId && globalWsId) {
      setUrlWsId(globalWsId);
    }
    // Only run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const workspaceId = urlWsId || globalWsId;

  function setWorkspaceId(id: string) {
    setUrlWsId(id);
    setGlobalWsId(id);
  }
  const days = parseInt(daysStr, 10) || 30;
  const location = useLocation();

  const wsParam = workspaceId ? `?workspace_id=${workspaceId}` : "";
  const isBacktests = location.pathname.startsWith("/dashboard/backtests");
  const isDashboard = !isBacktests;

  return (
    <div className="min-h-screen bg-background">
      {/* Top bar */}
      <header className="sticky top-0 z-[200] border-b border-border bg-bg-secondary/95 backdrop-blur supports-[backdrop-filter]:bg-bg-secondary/80">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between px-4 h-[56px]">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-accent" />
            <nav className="flex items-center gap-1">
              <Link
                to={`/${wsParam}`}
                className={cn(
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
                  isDashboard
                    ? "text-text-emphasis bg-bg-tertiary"
                    : "text-text-muted hover:text-foreground",
                )}
              >
                Dashboard
              </Link>
              <Link
                to={`/backtests${wsParam}`}
                className={cn(
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-colors",
                  isBacktests
                    ? "text-text-emphasis bg-bg-tertiary"
                    : "text-text-muted hover:text-foreground",
                )}
              >
                Backtests
              </Link>
            </nav>
            <WorkspaceSwitcher
              currentId={workspaceId}
              onSelect={(id) => setWorkspaceId(id)}
            />
          </div>

          {workspaceId && !isBacktests && (
            <DateRangeSelector
              value={days}
              onChange={(d) => setDays(String(d))}
            />
          )}
        </div>
      </header>

      {/* Content */}
      <main className="max-w-[1400px] mx-auto px-4 py-5">
        <Outlet context={{ workspaceId, days }} />
      </main>
    </div>
  );
}

export interface DashboardContext {
  workspaceId: string;
  days: number;
}
