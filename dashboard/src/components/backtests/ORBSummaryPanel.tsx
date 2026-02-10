import type { RunEvent } from "@/api/types";
import { cn } from "@/lib/utils";
import { Lock, Target, ShieldAlert, ArrowUpRight, ArrowDownRight } from "lucide-react";

interface ORBSummaryPanelProps {
  params: Record<string, unknown>;
  events: RunEvent[];
  trades: { pnl?: number; side?: string }[];
}

function deriveNoTradeReason(
  events: RunEvent[],
  trades: { pnl?: number; side?: string }[],
  confirmMode: string,
): string {
  const hasLock = events.some((e) => e.type === "orb_range_locked");
  const hasSetup = events.some((e) => e.type === "setup_valid");
  const hasEntry = events.some((e) => e.type === "entry_signal");

  if (!hasLock) return "Opening range did not form";
  if (!hasSetup) return "No breakout beyond opening range";
  if (!hasEntry) {
    if (confirmMode === "retest") return "Retest confirmation expired";
    return "Entry conditions not met";
  }
  if (trades.length === 0) return "Position size was zero (insufficient equity)";
  return "No trade";
}

/**
 * Compact ORB engine summary card for the Results tab.
 * Shows OR range, lock time, confirm/stop mode, and trade outcome.
 */
export function ORBSummaryPanel({ params, events, trades }: ORBSummaryPanelProps) {
  const orMinutes = Number(params.or_minutes ?? 30);
  const confirmMode = String(params.confirm_mode ?? "close-beyond");
  const stopMode = String(params.stop_mode ?? "or-opposite");
  const targetR = Number(params.target_r ?? 1.5);
  const maxTrades = Number(params.max_trades ?? 1);
  const session = String(params.session ?? "NY AM");

  // Derive from events
  const locked = events.find((e) => e.type === "orb_range_locked");
  const setup = events.find((e) => e.type === "setup_valid");
  const orHigh = locked ? Number(locked.high ?? 0) : null;
  const orLow = locked ? Number(locked.low ?? 0) : null;
  const orRange = locked ? Number(locked.range ?? 0) : null;
  const lockBar = locked ? Number(locked.bar_index) : null;

  const direction = setup ? String(setup.direction ?? "") : null;
  const noTradeReason = !trades.length ? deriveNoTradeReason(events, trades, confirmMode) : null;

  // Trade outcome
  const trade = trades.length > 0 ? trades[0] : null;
  const pnl = trade?.pnl ?? null;

  // No ORB events at all — not an ORB run, don't render
  if (events.length === 0 && !params.or_minutes) return null;

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <div className="px-4 py-2.5 border-b border-border flex items-center gap-2">
        <Target className="w-3.5 h-3.5 text-amber-400" />
        <h3 className="text-sm font-medium text-text-emphasis">
          ORB Engine Summary
        </h3>
        <span className="text-[10px] text-text-muted ml-auto">{session}</span>
      </div>

      <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* OR Range */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">
            Opening Range
          </div>
          {orHigh !== null && orLow !== null ? (
            <div className="space-y-0.5">
              <div className="text-sm font-mono text-foreground">
                {orHigh.toFixed(2)} — {orLow.toFixed(2)}
              </div>
              <div className="text-xs text-text-muted">
                Range: <span className="font-mono">{orRange?.toFixed(2)}</span>
              </div>
            </div>
          ) : (
            <div className="text-sm text-text-muted">Not locked</div>
          )}
        </div>

        {/* Lock Info */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">
            Lock
          </div>
          <div className="flex items-center gap-1.5">
            <Lock className="w-3 h-3 text-amber-400" />
            <span className="text-sm font-mono text-foreground">
              {orMinutes}m
            </span>
            {lockBar !== null && (
              <span className="text-xs text-text-muted">
                (bar {lockBar})
              </span>
            )}
          </div>
        </div>

        {/* Confirm + Stop Mode */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">
            Modes
          </div>
          <div className="space-y-0.5 text-xs">
            <div>
              <span className="text-text-muted">Confirm: </span>
              <span className="font-mono text-foreground">{confirmMode}</span>
            </div>
            <div>
              <span className="text-text-muted">Stop: </span>
              <span className="font-mono text-foreground">{stopMode}</span>
            </div>
          </div>
        </div>

        {/* Trade Outcome */}
        <div>
          <div className="text-[10px] text-text-muted uppercase tracking-wide mb-1">
            Outcome
          </div>
          {trade ? (
            <div className="space-y-0.5">
              <div className="flex items-center gap-1">
                {direction === "long" ? (
                  <ArrowUpRight className="w-3 h-3 text-emerald-400" />
                ) : direction === "short" ? (
                  <ArrowDownRight className="w-3 h-3 text-red-400" />
                ) : null}
                <span
                  className={cn(
                    "text-sm font-mono",
                    pnl !== null && pnl > 0 ? "text-emerald-400" : "text-red-400",
                  )}
                >
                  {pnl !== null ? (pnl > 0 ? "+" : "") + pnl.toFixed(2) : "—"}
                </span>
              </div>
              <div className="text-xs text-text-muted flex items-center gap-1">
                {pnl !== null && pnl > 0 ? (
                  <><Target className="w-2.5 h-2.5" /> Winner</>
                ) : pnl !== null && pnl < 0 ? (
                  <><ShieldAlert className="w-2.5 h-2.5" /> Loser</>
                ) : (
                  "Flat"
                )}
              </div>
            </div>
          ) : (
            <div className="text-sm text-text-muted">{noTradeReason}</div>
          )}
        </div>
      </div>

      {/* Footer: params */}
      <div className="px-4 py-2 border-t border-border/50 flex gap-4 text-[10px] text-text-muted">
        <span>Target R: <span className="font-mono">{targetR}</span></span>
        <span>Max trades: <span className="font-mono">{maxTrades}</span></span>
        {trades.length > 1 && (
          <span>Trades: <span className="font-mono">{trades.length}</span></span>
        )}
      </div>
    </div>
  );
}
