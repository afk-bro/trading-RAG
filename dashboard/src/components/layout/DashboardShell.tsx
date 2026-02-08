import { Outlet } from "react-router-dom";
import { useUrlState } from "@/hooks/use-url-state";
import { DateRangeSelector } from "./DateRangeSelector";
import { Activity } from "lucide-react";

export function DashboardShell() {
  const [workspaceId] = useUrlState("workspace_id", "");
  const [daysStr, setDays] = useUrlState("days", "30");
  const days = parseInt(daysStr, 10) || 30;

  return (
    <div className="min-h-screen bg-background">
      {/* Top bar */}
      <header className="sticky top-0 z-[200] border-b border-border bg-bg-secondary/95 backdrop-blur supports-[backdrop-filter]:bg-bg-secondary/80">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between px-4 h-[56px]">
          <div className="flex items-center gap-3">
            <Activity className="w-5 h-5 text-accent" />
            <span className="text-base font-semibold text-text-emphasis">
              Trading Dashboard
            </span>
            {workspaceId && (
              <span className="text-xs text-text-muted font-mono bg-bg-tertiary px-2 py-1 rounded-md">
                {workspaceId.slice(0, 8)}...
              </span>
            )}
          </div>

          {workspaceId && (
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
