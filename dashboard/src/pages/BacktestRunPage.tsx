import { useState, useMemo } from "react";
import { useParams, Link, useOutletContext } from "react-router-dom";
import type { DashboardContext } from "@/components/layout/DashboardShell";
import { useUrlState } from "@/hooks/use-url-state";
import { useRunDetail } from "@/hooks/use-run-detail";
import { useRunEvents } from "@/hooks/use-run-events";
import { useRunLineage } from "@/hooks/use-run-lineage";
import { DeltaKpiStrip } from "@/components/coaching/DeltaKpiStrip";
import { WhatYouLearnedCard } from "@/components/coaching/WhatYouLearnedCard";
import { ProcessScoreCard } from "@/components/coaching/ProcessScoreCard";
import { BaselineSelector } from "@/components/coaching/BaselineSelector";
import { LossAttributionPanel } from "@/components/coaching/LossAttributionPanel";
import { EquityChart, type TradeMarker } from "@/components/equity/EquityChart";
import { BacktestTradesTable } from "@/components/backtests/BacktestTradesTable";
import { BacktestTradeDrawer } from "@/components/backtests/BacktestTradeDrawer";
import { ReplayPanel } from "@/components/backtests/ReplayPanel";
import { ORBSummaryPanel } from "@/components/backtests/ORBSummaryPanel";
import { mapBacktestEquity } from "@/lib/chart-utils";
import { downloadFile } from "@/api/client";
import type { BacktestChartTradeRecord } from "@/api/types";
import { Skeleton } from "@/components/Skeleton";
import { ErrorAlert } from "@/components/ErrorAlert";
import { WorkspacePicker } from "@/components/layout/WorkspacePicker";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  Download,
  AlertTriangle,
  BarChart3,
  Film,
  Table2,
} from "lucide-react";

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-success/15 text-success",
  failed: "bg-danger/15 text-danger",
  running: "bg-accent/15 text-accent",
};

const PAGE_SIZE = 50;

type Tab = "results" | "replay";

export function BacktestRunPage() {
  const { runId } = useParams<{ runId: string }>();
  const { workspaceId: ctxWorkspaceId } = useOutletContext<DashboardContext>();
  const [urlWsId, setUrlWsId] = useUrlState("workspace_id", "");
  const workspaceId = ctxWorkspaceId || urlWsId;
  const [tradesPage, setTradesPage] = useState(1);
  const [activeTab, setActiveTab] = useState<Tab>("results");
  const [baselineRunId, setBaselineRunId] = useState<string | null>(null);
  const [drawerTrade, setDrawerTrade] = useState<BacktestChartTradeRecord | null>(null);

  const { data, isLoading, isError, refetch } = useRunDetail(
    workspaceId || null,
    runId ?? null,
    true, // include coaching
    baselineRunId,
  );
  const { data: eventsData } = useRunEvents(runId ?? null);
  const { data: lineageData } = useRunLineage(
    workspaceId || null,
    runId ?? null,
  );

  if (!workspaceId) {
    return <WorkspacePicker onSelect={setUrlWsId} />;
  }

  const equityData = useMemo(
    () => (data?.equity ? mapBacktestEquity(data.equity) : []),
    [data?.equity],
  );

  // Client-side pagination for trades
  const allTrades = data?.trades ?? [];
  const totalTrades = data?.trade_count ?? allTrades.length;
  const totalPages = Math.ceil(totalTrades / PAGE_SIZE);
  const pagedTrades = useMemo(() => {
    const start = (tradesPage - 1) * PAGE_SIZE;
    return allTrades.slice(start, start + PAGE_SIZE);
  }, [allTrades, tradesPage]);

  const tradeMarkers: TradeMarker[] = useMemo(() => {
    return pagedTrades.map((t) => ({
      time: t.t_entry,
      side: t.side,
    }));
  }, [pagedTrades]);

  const meta = data?.dataset;
  const strategyName = data?.strategy?.name;
  const symbol = meta?.symbol ?? "";
  const timeframe = meta?.timeframe ?? "";
  const dateRange =
    meta?.date_min && meta?.date_max
      ? `${String(meta.date_min).slice(0, 10)} — ${String(meta.date_max).slice(0, 10)}`
      : "";

  const regimeBadges: string[] = [];
  if (data?.regime_is) {
    const r = data.regime_is;
    const tags = [r.trend_tag, r.vol_tag, r.efficiency_tag].filter(Boolean);
    if (tags.length) regimeBadges.push(`IS: ${tags.join(", ")}`);
  }
  if (data?.regime_oos) {
    const r = data.regime_oos;
    const tags = [r.trend_tag, r.vol_tag, r.efficiency_tag].filter(Boolean);
    if (tags.length) regimeBadges.push(`OOS: ${tags.join(", ")}`);
  }

  // Coaching data
  const coaching = data?.coaching;
  const trajectory = data?.trajectory;

  // Lineage candidates for baseline selector
  const candidates = lineageData?.candidates ?? [];
  const autoBaselineId = candidates.find((c) => c.is_auto_baseline)?.run_id ?? null;

  function handleExportCsv() {
    if (!runId || totalTrades === 0) return;
    downloadFile(
      `/backtests/runs/${runId}/export/trades.csv`,
      `trades_${runId.slice(0, 8)}.csv`,
    );
  }

  function handleExportJson() {
    if (!runId) return;
    downloadFile(
      `/backtests/runs/${runId}/export/snapshot.json`,
      `snapshot_${runId.slice(0, 8)}.json`,
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
        <Skeleton className="h-[400px]" />
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorAlert
        message="Failed to load backtest run"
        onRetry={() => refetch()}
      />
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12 text-text-muted">
        Backtest run not found.
      </div>
    );
  }

  const title = strategyName
    ? `${strategyName}${symbol ? ` · ${symbol}` : ""}`
    : symbol || `Run ${runId?.slice(0, 8)}`;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <Link
            to={`/backtests${workspaceId ? `?workspace_id=${workspaceId}` : ""}`}
            className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-foreground mb-2"
          >
            <ArrowLeft className="w-3 h-3" /> Back to runs
          </Link>
          <div className="flex items-center gap-3 flex-wrap">
            <h2 className="text-lg font-semibold text-text-emphasis">
              {title}
            </h2>
            {timeframe && (
              <span className="text-sm text-text-muted">{timeframe}</span>
            )}
            {dateRange && (
              <span className="text-xs text-text-muted font-mono">
                {dateRange}
              </span>
            )}
            <span
              className={cn(
                "px-2 py-0.5 rounded-full text-xs font-medium",
                STATUS_COLORS[data.status] ?? "bg-bg-tertiary text-text-muted",
              )}
            >
              {data.status}
            </span>
            {eventsData && eventsData.event_count > 0 && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-amber-500/15 text-amber-400 border border-amber-500/30">
                ORB v1 · schema {String((eventsData.events[0] as Record<string, unknown>)?.schema_version ?? "1.0.0")}
              </span>
            )}
          </div>
          {regimeBadges.length > 0 && (
            <div className="flex gap-2 mt-1">
              {regimeBadges.map((badge) => (
                <span
                  key={badge}
                  className="text-[10px] px-2 py-0.5 bg-bg-tertiary text-text-muted rounded"
                >
                  {badge}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Export buttons */}
        <div className="flex gap-2 flex-shrink-0">
          {totalTrades > 0 && (
            <button
              onClick={handleExportCsv}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium
                         border border-border rounded-md text-text-muted
                         hover:text-foreground hover:bg-bg-tertiary transition-colors"
            >
              <Download className="w-3 h-3" /> CSV
            </button>
          )}
          <button
            onClick={handleExportJson}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium
                       border border-border rounded-md text-text-muted
                       hover:text-foreground hover:bg-bg-tertiary transition-colors"
          >
            <Download className="w-3 h-3" /> JSON
          </button>
        </div>
      </div>

      {/* Warnings bar */}
      {data.warnings.length > 0 && (
        <div className="flex items-start gap-2 p-3 bg-warning/10 border border-warning/20 rounded-lg">
          <AlertTriangle className="w-4 h-4 text-warning flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            {data.warnings.map((note, i) => (
              <p key={i} className="text-xs text-warning">
                {note}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Tab bar */}
      <div role="tablist" className="flex gap-1 border-b border-border">
        <button
          role="tab"
          aria-selected={activeTab === "results"}
          onClick={() => setActiveTab("results")}
          className={cn(
            "flex items-center gap-1.5 px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px",
            activeTab === "results"
              ? "border-accent text-foreground"
              : "border-transparent text-text-muted hover:text-foreground",
          )}
        >
          <Table2 className="w-3.5 h-3.5" /> Results
        </button>
        <button
          role="tab"
          aria-selected={activeTab === "replay"}
          onClick={() => setActiveTab("replay")}
          className={cn(
            "flex items-center gap-1.5 px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px",
            activeTab === "replay"
              ? "border-accent text-foreground"
              : "border-transparent text-text-muted hover:text-foreground",
          )}
        >
          <Film className="w-3.5 h-3.5" /> Replay
          {eventsData && eventsData.event_count > 0 && (
            <span className="ml-1 text-[10px] bg-accent/20 text-accent px-1.5 py-0.5 rounded-full">
              {eventsData.event_count}
            </span>
          )}
        </button>
      </div>

      {activeTab === "results" ? (
        <div role="tabpanel" className="space-y-4">
          {/* Coaching: What You Learned */}
          {coaching?.lineage && (
            <WhatYouLearnedCard lineage={coaching.lineage} />
          )}

          {/* Coaching: Process Score */}
          {coaching?.process_score && (
            <ProcessScoreCard score={coaching.process_score} />
          )}

          {/* Baseline selector + KPI strip */}
          <div className="space-y-2">
            {candidates.length > 0 && (
              <BaselineSelector
                candidates={candidates}
                currentBaselineId={baselineRunId}
                autoBaselineId={autoBaselineId}
                onSelect={setBaselineRunId}
              />
            )}
            <DeltaKpiStrip
              summary={data.summary}
              deltas={coaching?.lineage?.deltas}
              trajectory={trajectory?.runs}
              isLoading={isLoading}
            />
          </div>

          {/* ORB engine summary (renders only for ORB runs) */}
          {eventsData && eventsData.event_count > 0 && (
            <ORBSummaryPanel
              params={data.params}
              events={eventsData.events}
              trades={allTrades.map((t) => ({
                pnl: t.pnl,
                side: t.side,
              }))}
            />
          )}

          {/* Equity chart */}
          {data.equity.length === 0 ? (
            <div className="bg-bg-secondary border border-border rounded-lg p-8 text-center text-text-muted text-sm">
              Equity curve not available for this run
            </div>
          ) : (
            <EquityChart data={equityData} tradeMarkers={tradeMarkers} />
          )}

          {/* Loss Attribution */}
          {coaching?.loss_attribution &&
            !("timed_out" in coaching.loss_attribution) && (
              <LossAttributionPanel attribution={coaching.loss_attribution} />
            )}

          {/* Trades table */}
          <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <h3 className="text-sm font-medium text-text-emphasis">
                Trades ({totalTrades})
              </h3>
              {totalPages > 1 && (
                <div className="flex items-center gap-2 text-xs text-text-muted">
                  <button
                    disabled={tradesPage <= 1}
                    onClick={() => setTradesPage((p) => p - 1)}
                    className="px-2 py-1 rounded border border-border disabled:opacity-40
                               hover:bg-bg-tertiary transition-colors"
                  >
                    Prev
                  </button>
                  <span>
                    {tradesPage} / {totalPages}
                  </span>
                  <button
                    disabled={tradesPage >= totalPages}
                    onClick={() => setTradesPage((p) => p + 1)}
                    className="px-2 py-1 rounded border border-border disabled:opacity-40
                               hover:bg-bg-tertiary transition-colors"
                  >
                    Next
                  </button>
                </div>
              )}
            </div>
            {pagedTrades.length > 0 ? (
              <BacktestTradesTable
                data={pagedTrades}
                onTradeClick={setDrawerTrade}
              />
            ) : (
              <div className="p-12 text-center space-y-2">
                <BarChart3 className="w-10 h-10 text-text-muted mx-auto" />
                <p className="text-sm font-medium text-text-muted">
                  No trades executed
                </p>
                <p className="text-xs text-text-muted/70">
                  Strategy did not generate entry signals for this run
                </p>
              </div>
            )}
          </div>

          {/* Trade drawer */}
          <BacktestTradeDrawer
            open={!!drawerTrade}
            onClose={() => setDrawerTrade(null)}
            trade={drawerTrade}
            symbol={symbol}
            workspaceId={workspaceId}
          />
        </div>
      ) : (
        <div role="tabpanel" className="space-y-4">
          {/* Replay tab content */}
          {data.equity.length > 0 && (
            <EquityChart data={equityData} tradeMarkers={tradeMarkers} />
          )}
          <ReplayPanel
            events={eventsData?.events ?? []}
            maxBarIndex={data.equity.length > 0 ? data.equity.length - 1 : 0}
          />
        </div>
      )}
    </div>
  );
}
