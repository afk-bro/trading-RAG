import { useOutletContext, useSearchParams } from "react-router-dom";
import type { DashboardContext } from "@/components/layout/DashboardShell";
import { useUrlState } from "@/hooks/use-url-state";
import { useBacktestRuns } from "@/hooks/use-backtest-runs";
import { BacktestsTable } from "@/components/backtests/BacktestsTable";
import { WorkspacePicker } from "@/components/layout/WorkspacePicker";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight } from "lucide-react";

const STATUSES = ["all", "completed", "failed", "running"] as const;
const PAGE_SIZE = 25;

export function BacktestsPage() {
  useOutletContext<DashboardContext>();
  const [workspaceId, setWorkspaceId] = useUrlState("workspace_id", "");
  const [searchParams, setSearchParams] = useSearchParams();

  const status = searchParams.get("status") ?? "all";
  const offset = parseInt(searchParams.get("offset") ?? "0", 10) || 0;

  const { data, isLoading } = useBacktestRuns(
    workspaceId || null,
    status,
    PAGE_SIZE,
    offset,
  );

  function setFilter(key: string, value: string) {
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (key === "status" && value === "all") {
          next.delete("status");
        } else {
          next.set(key, value);
        }
        // Reset offset when changing filter
        if (key === "status") next.delete("offset");
        return next;
      },
      { replace: true },
    );
  }

  if (!workspaceId) {
    return <WorkspacePicker onSelect={setWorkspaceId} />;
  }

  const total = data?.total ?? 0;
  const hasNext = offset + PAGE_SIZE < total;
  const hasPrev = offset > 0;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-text-emphasis">
          Backtest Runs
        </h2>

        {/* Status filter */}
        <div className="flex gap-1">
          {STATUSES.map((s) => (
            <button
              key={s}
              onClick={() => setFilter("status", s)}
              className={cn(
                "px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize",
                status === s
                  ? "bg-bg-tertiary text-text-emphasis"
                  : "text-text-muted hover:text-foreground",
              )}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
        {isLoading ? (
          <div className="p-4 space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={i}
                className="h-10 bg-bg-tertiary rounded animate-pulse"
              />
            ))}
          </div>
        ) : data && data.items.length > 0 ? (
          <BacktestsTable data={data.items} />
        ) : (
          <div className="p-8 text-center text-text-muted text-sm">
            No backtest runs found
          </div>
        )}
      </div>

      {/* Pagination */}
      {total > PAGE_SIZE && (
        <div className="flex items-center justify-between text-sm text-text-muted">
          <span>
            {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
          </span>
          <div className="flex gap-2">
            <button
              disabled={!hasPrev}
              onClick={() =>
                setFilter("offset", String(Math.max(0, offset - PAGE_SIZE)))
              }
              className="flex items-center gap-1 px-3 py-1.5 rounded-md border border-border
                         disabled:opacity-40 hover:bg-bg-tertiary transition-colors"
            >
              <ChevronLeft className="w-4 h-4" /> Prev
            </button>
            <button
              disabled={!hasNext}
              onClick={() => setFilter("offset", String(offset + PAGE_SIZE))}
              className="flex items-center gap-1 px-3 py-1.5 rounded-md border border-border
                         disabled:opacity-40 hover:bg-bg-tertiary transition-colors"
            >
              Next <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
