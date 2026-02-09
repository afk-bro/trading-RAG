import { useState } from "react";
import { useOutletContext, useSearchParams, useNavigate } from "react-router-dom";
import type { DashboardContext } from "@/components/layout/DashboardShell";
import { useUrlState } from "@/hooks/use-url-state";
import { useBacktestRuns } from "@/hooks/use-backtest-runs";
import { BacktestsTable } from "@/components/backtests/BacktestsTable";
import { WorkspacePicker } from "@/components/layout/WorkspacePicker";
import { Skeleton } from "@/components/Skeleton";
import { ErrorAlert } from "@/components/ErrorAlert";
import { cn } from "@/lib/utils";
import { ChevronLeft, ChevronRight, GitCompareArrows, Inbox } from "lucide-react";

const STATUSES = ["all", "completed", "failed", "running"] as const;
const PAGE_SIZE = 25;

export function BacktestsPage() {
  useOutletContext<DashboardContext>();
  const navigate = useNavigate();
  const [workspaceId, setWorkspaceId] = useUrlState("workspace_id", "");
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const status = searchParams.get("status") ?? "all";
  const offset = parseInt(searchParams.get("offset") ?? "0", 10) || 0;

  const { data, isLoading, isError, refetch } = useBacktestRuns(
    workspaceId || null,
    status,
    PAGE_SIZE,
    offset,
  );

  function setFilter(key: string, value: string) {
    setSelectedIds(new Set());
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

  function handleCompare() {
    const ids = Array.from(selectedIds);
    if (ids.length !== 2) return;
    navigate(`/backtests/compare?a=${ids[0]}&b=${ids[1]}`);
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
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-text-emphasis">
            Backtest Runs
          </h2>
          {selectedIds.size === 2 && (
            <button
              onClick={handleCompare}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium
                         bg-accent text-white rounded-md hover:bg-accent/90 transition-colors"
            >
              <GitCompareArrows className="w-3.5 h-3.5" /> Compare
            </button>
          )}
        </div>

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
              <Skeleton key={i} className="h-10" />
            ))}
          </div>
        ) : isError ? (
          <div className="p-4">
            <ErrorAlert
              message="Failed to load backtest runs"
              onRetry={() => refetch()}
            />
          </div>
        ) : data && data.items.length > 0 ? (
          <BacktestsTable
            data={data.items}
            selectedIds={selectedIds}
            onSelectionChange={setSelectedIds}
          />
        ) : (
          <div className="p-12 text-center space-y-2">
            <Inbox className="w-10 h-10 text-text-muted mx-auto" />
            <p className="text-sm font-medium text-text-muted">
              No backtest runs found
            </p>
            <p className="text-xs text-text-muted/70">
              Run a backtest via the API to see results here
            </p>
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
