import { useState, useMemo } from "react";
import { useParams, Link, useOutletContext } from "react-router-dom";
import type { DashboardContext } from "@/components/layout/DashboardShell";
import { useBacktestChart } from "@/hooks/use-backtest-chart";
import { RunKpiStrip } from "@/components/backtests/RunKpiStrip";
import { EquityChart, type TradeMarker } from "@/components/equity/EquityChart";
import { BacktestTradesTable } from "@/components/backtests/BacktestTradesTable";
import { BacktestTradeDrawer } from "@/components/backtests/BacktestTradeDrawer";
import { mapBacktestEquity } from "@/lib/chart-utils";
import { downloadFile } from "@/api/client";
import type { BacktestChartTradeRecord } from "@/api/types";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  Download,
  AlertTriangle,
} from "lucide-react";

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-success/15 text-success",
  failed: "bg-danger/15 text-danger",
  running: "bg-accent/15 text-accent",
};

export function BacktestRunPage() {
  const { runId } = useParams<{ runId: string }>();
  const { workspaceId } = useOutletContext<DashboardContext>();
  const [tradesPage, setTradesPage] = useState(1);
  const { data, isLoading } = useBacktestChart(runId ?? null, tradesPage);

  const [drawerTrade, setDrawerTrade] = useState<BacktestChartTradeRecord | null>(null);

  const equityData = useMemo(
    () => (data?.equity ? mapBacktestEquity(data.equity) : []),
    [data?.equity],
  );

  const tradeMarkers: TradeMarker[] = useMemo(() => {
    if (!data?.trades_page) return [];
    return data.trades_page.map((t) => ({
      time: t.t_entry,
      side: t.side,
    }));
  }, [data?.trades_page]);

  const meta = data?.dataset_meta;
  const symbol = meta?.symbol ?? "";
  const timeframe = meta?.timeframe ?? "";
  const dateRange =
    meta?.date_min && meta?.date_max
      ? `${meta.date_min.slice(0, 10)} — ${meta.date_max.slice(0, 10)}`
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

  function handleExportCsv() {
    if (!data?.exports?.trades_csv) return;
    downloadFile(data.exports.trades_csv, `trades_${runId?.slice(0, 8)}.csv`);
  }

  function handleExportJson() {
    if (!data?.exports?.json_snapshot) return;
    downloadFile(
      data.exports.json_snapshot,
      `snapshot_${runId?.slice(0, 8)}.json`,
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-64 bg-bg-tertiary rounded animate-pulse" />
        <div className="grid grid-cols-6 gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-20 bg-bg-tertiary rounded animate-pulse" />
          ))}
        </div>
        <div className="h-[400px] bg-bg-tertiary rounded animate-pulse" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-center py-12 text-text-muted">
        Backtest run not found.
      </div>
    );
  }

  const pagination = data.trades_pagination;
  const totalPages = Math.ceil(pagination.total / pagination.page_size);

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
              {meta?.symbol ?? `Run ${runId?.slice(0, 8)}`}
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
          {data.exports?.trades_csv && (
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

      {/* Notes bar */}
      {data.notes.length > 0 && (
        <div className="flex items-start gap-2 p-3 bg-warning/10 border border-warning/20 rounded-lg">
          <AlertTriangle className="w-4 h-4 text-warning flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            {data.notes.map((note, i) => (
              <p key={i} className="text-xs text-warning">
                {note}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* KPI strip */}
      <RunKpiStrip summary={data.summary} />

      {/* Equity chart */}
      {data.equity_source === "missing" ? (
        <div className="bg-bg-secondary border border-border rounded-lg p-8 text-center text-text-muted text-sm">
          Equity curve not available for this run
        </div>
      ) : (
        <EquityChart data={equityData} tradeMarkers={tradeMarkers} />
      )}

      {/* Trades table */}
      <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h3 className="text-sm font-medium text-text-emphasis">
            Trades ({pagination.total})
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
        {data.trades_page.length > 0 ? (
          <BacktestTradesTable
            data={data.trades_page}
            onTradeClick={setDrawerTrade}
          />
        ) : (
          <div className="p-8 text-center text-text-muted text-sm">
            No trades recorded
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
  );
}
