import { useState, useEffect } from "react";
import type { BacktestChartTradeRecord } from "@/api/types";
import { useRagContext } from "@/hooks/use-rag-context";
import { useQueryClient } from "@tanstack/react-query";
import { X, RefreshCw, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import { fmtPct, fmtPnl } from "@/lib/chart-utils";
import { Field } from "@/components/ui/Field";
import { format } from "date-fns";

interface Props {
  open: boolean;
  onClose: () => void;
  trade: BacktestChartTradeRecord | null;
  symbol: string;
  workspaceId: string;
}

type Tab = "details" | "context";

function fmtPrice(n: number | undefined | null): string {
  if (n == null) return "—";
  return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 5 });
}

function ContextTab({
  symbol,
  entryTime,
  workspaceId,
  active,
}: {
  symbol: string;
  entryTime: string;
  workspaceId: string;
  active: boolean;
}) {
  const queryClient = useQueryClient();
  const { data, isLoading, isFetching } = useRagContext(
    workspaceId || null,
    symbol || null,
    entryTime || null,
    active,
  );

  function rerun() {
    queryClient.invalidateQueries({
      queryKey: ["rag-context", workspaceId, symbol, entryTime],
    });
  }

  if (!symbol) {
    return (
      <p className="text-sm text-text-muted py-4">
        No symbol — RAG context unavailable.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-text-muted">
          Relevant knowledge for{" "}
          <span className="text-foreground font-medium">{symbol}</span>
        </p>
        <button
          onClick={rerun}
          disabled={isFetching}
          className="flex items-center gap-1 px-2 py-1 text-xs text-text-muted
                     hover:text-foreground disabled:opacity-40 transition-colors"
        >
          <RefreshCw
            className={`w-3 h-3 ${isFetching ? "animate-spin" : ""}`}
          />
          Re-run
        </button>
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-24 bg-bg-tertiary rounded animate-pulse"
            />
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {(data?.results ?? []).map((chunk) => (
            <div
              key={chunk.chunk_id}
              className="bg-background border border-border-subtle rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-text-emphasis truncate">
                  {chunk.source_title ?? "Untitled"}
                </span>
                <div className="flex items-center gap-2">
                  <div className="w-12 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent rounded-full"
                      style={{
                        width: `${Math.min(chunk.score * 100, 100)}%`,
                      }}
                    />
                  </div>
                  <span className="text-[10px] text-text-muted">
                    {(chunk.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <p className="text-xs text-foreground line-clamp-4 leading-relaxed">
                {chunk.content}
              </p>
              {chunk.source_url && (
                <a
                  href={chunk.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 mt-2 text-[10px] text-accent hover:underline"
                >
                  Source <ExternalLink className="w-2.5 h-2.5" />
                </a>
              )}
            </div>
          ))}

          {data?.results?.length === 0 && (
            <p className="text-xs text-text-muted py-4 text-center">
              No relevant context found
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export function BacktestTradeDrawer({
  open,
  onClose,
  trade,
  symbol,
  workspaceId,
}: Props) {
  const [tab, setTab] = useState<Tab>("details");

  useEffect(() => {
    setTab("details");
  }, [trade?.t_entry]);

  if (!open || !trade) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-[500] bg-black/50"
        onClick={onClose}
      />

      {/* Drawer */}
      <div role="dialog" aria-modal="true" className="fixed top-0 right-0 z-[501] h-full w-full max-w-[480px] bg-bg-secondary border-l border-border overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-bg-secondary border-b border-border px-4 py-3 z-10">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-text-emphasis">
              Trade
              {symbol && (
                <span className="ml-2 text-text-muted">{symbol}</span>
              )}
            </h3>
            <button
              onClick={onClose}
              className="text-text-muted hover:text-foreground p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex gap-1">
            {(["details", "context"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize",
                  tab === t
                    ? "bg-bg-tertiary text-text-emphasis"
                    : "text-text-muted hover:text-foreground",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-4">
          {tab === "details" ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <Field
                  label="Side"
                  value={trade.side}
                  className={
                    trade.side === "long" ? "text-success capitalize" : "text-danger capitalize"
                  }
                />
                <Field
                  label="PnL"
                  value={fmtPnl(trade.pnl)}
                  className={trade.pnl >= 0 ? "text-success" : "text-danger"}
                />
                <Field label="Entry Price" value={fmtPrice(trade.entry_price)} />
                <Field label="Exit Price" value={fmtPrice(trade.exit_price)} />
                <Field
                  label="Entry Time"
                  value={
                    trade.t_entry
                      ? format(new Date(trade.t_entry), "yyyy-MM-dd HH:mm")
                      : "—"
                  }
                />
                <Field
                  label="Exit Time"
                  value={
                    trade.t_exit
                      ? format(new Date(trade.t_exit), "yyyy-MM-dd HH:mm")
                      : "—"
                  }
                />
                <Field label="Return" value={fmtPct(trade.return_pct)} />
                <Field
                  label="Size"
                  value={trade.size != null ? trade.size.toFixed(4) : "—"}
                />
              </div>
            </div>
          ) : (
            <ContextTab
              symbol={symbol}
              entryTime={trade.t_entry}
              workspaceId={workspaceId}
              active={tab === "context"}
            />
          )}
        </div>
      </div>
    </>
  );
}
