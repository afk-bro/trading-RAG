import { useState, useEffect, useRef } from "react";
import type { BacktestChartTradeRecord } from "@/api/types";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { fmtPct, fmtPnl, fmtPrice } from "@/lib/chart-utils";
import { Field } from "@/components/ui/Field";
import { TradeContext } from "@/components/trades/TradeContext";
import { format } from "date-fns";

interface Props {
  open: boolean;
  onClose: () => void;
  trade: BacktestChartTradeRecord | null;
  symbol: string;
  workspaceId: string;
}

type Tab = "details" | "context";

export function BacktestTradeDrawer({
  open,
  onClose,
  trade,
  symbol,
  workspaceId,
}: Props) {
  const [tab, setTab] = useState<Tab>("details");
  const drawerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTab("details");
  }, [trade?.t_entry]);

  // Trap focus and handle Escape
  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") { onClose(); return; }
      if (e.key !== "Tab" || !drawerRef.current) return;
      const focusable = drawerRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable.length) return;
      const first = focusable[0]!;
      const last = focusable[focusable.length - 1]!;
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  if (!open || !trade) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-[500] bg-black/50"
        onClick={onClose}
      />

      {/* Drawer */}
      <div ref={drawerRef} role="dialog" aria-modal="true" className="fixed top-0 right-0 z-[501] h-full w-full max-w-[480px] bg-bg-secondary border-l border-border overflow-y-auto">
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
              aria-label="Close trade drawer"
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
            <TradeContext
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
