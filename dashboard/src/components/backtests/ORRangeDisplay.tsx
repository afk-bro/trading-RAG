import { useMemo } from "react";
import type { RunEvent } from "@/api/types";
import { cn } from "@/lib/utils";
import { Lock, Unlock, TrendingUp, TrendingDown } from "lucide-react";

interface ORRangeDisplayProps {
  events: RunEvent[];
  barIndex: number;
}

/**
 * Visual display of the Opening Range state derived from replay events.
 *
 * - OR_BUILD: pulsing high/low lines with current values
 * - Locked: solid shaded band with range stats
 * - After setup_valid: breakout direction arrow
 */
export function ORRangeDisplay({ events, barIndex }: ORRangeDisplayProps) {
  const state = useMemo(() => deriveState(events, barIndex), [events, barIndex]);

  if (!state) return null;

  const { high, low, range, phase, direction, lockBarIndex, entryPrice } = state;

  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide">
          Opening Range
        </h4>
        <span
          className={cn(
            "flex items-center gap-1 px-2 py-0.5 text-[10px] rounded-full border",
            phase === "forming"
              ? "bg-blue-500/15 text-blue-400 border-blue-500/30"
              : phase === "locked"
                ? "bg-amber-500/15 text-amber-400 border-amber-500/30"
                : phase === "breakout"
                  ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                  : "bg-purple-500/15 text-purple-400 border-purple-500/30",
          )}
        >
          {phase === "forming" ? (
            <>
              <Unlock className="w-2.5 h-2.5" /> Forming
            </>
          ) : phase === "locked" ? (
            <>
              <Lock className="w-2.5 h-2.5" /> Locked
            </>
          ) : (
            <>
              {direction === "long" ? (
                <TrendingUp className="w-2.5 h-2.5" />
              ) : (
                <TrendingDown className="w-2.5 h-2.5" />
              )}
              {phase === "breakout" ? "Breakout" : "Entry"}
            </>
          )}
        </span>
      </div>

      {/* Visual range bar */}
      <div className="relative h-16 flex items-center">
        {/* Background track */}
        <div className="absolute inset-x-0 h-0.5 bg-bg-tertiary top-1/2 -translate-y-1/2" />

        {/* Shaded range band */}
        <div
          className={cn(
            "absolute inset-x-[10%] right-[10%] rounded transition-all duration-300",
            phase === "forming"
              ? "bg-blue-500/10 border border-dashed border-blue-500/30"
              : "bg-amber-500/10 border border-amber-500/30",
          )}
          style={{ top: "15%", bottom: "15%" }}
        />

        {/* High line */}
        <div className="absolute inset-x-[8%] right-[8%]" style={{ top: "15%" }}>
          <div
            className={cn(
              "h-px w-full",
              phase === "forming" ? "bg-blue-400/60" : "bg-amber-400",
            )}
          />
          <span className="absolute -top-4 left-0 text-[10px] font-mono text-text-muted">
            H: {high.toFixed(2)}
          </span>
        </div>

        {/* Low line */}
        <div className="absolute inset-x-[8%] right-[8%]" style={{ bottom: "15%" }}>
          <div
            className={cn(
              "h-px w-full",
              phase === "forming" ? "bg-blue-400/60" : "bg-amber-400",
            )}
          />
          <span className="absolute -bottom-4 left-0 text-[10px] font-mono text-text-muted">
            L: {low.toFixed(2)}
          </span>
        </div>

        {/* Lock marker */}
        {lockBarIndex !== null && (
          <div className="absolute top-1/2 -translate-y-1/2 right-[10%]">
            <Lock className="w-3 h-3 text-amber-400" />
          </div>
        )}

        {/* Breakout arrow */}
        {direction && (
          <div
            className={cn(
              "absolute right-[5%]",
              direction === "long" ? "top-[5%]" : "bottom-[5%]",
            )}
          >
            {direction === "long" ? (
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
          </div>
        )}

        {/* Entry price tick */}
        {entryPrice !== null && (
          <div className="absolute right-[3%] top-1/2 -translate-y-1/2">
            <span className="text-[9px] font-mono text-purple-400">
              {entryPrice.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-4 mt-3 text-xs">
        <div>
          <span className="text-text-muted">Range: </span>
          <span className="font-mono text-foreground">{range.toFixed(2)}</span>
        </div>
        {lockBarIndex !== null && (
          <div>
            <span className="text-text-muted">Locked bar: </span>
            <span className="font-mono text-foreground">{lockBarIndex}</span>
          </div>
        )}
        {direction && (
          <div>
            <span className="text-text-muted">Direction: </span>
            <span
              className={cn(
                "font-mono",
                direction === "long" ? "text-emerald-400" : "text-red-400",
              )}
            >
              {direction}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

interface ORState {
  high: number;
  low: number;
  range: number;
  phase: "forming" | "locked" | "breakout" | "entry";
  direction: string | null;
  lockBarIndex: number | null;
  entryPrice: number | null;
}

function deriveState(events: RunEvent[], barIndex: number): ORState | null {
  let high = 0;
  let low = 0;
  let lockBarIndex: number | null = null;
  let direction: string | null = null;
  let entryPrice: number | null = null;
  let hasRange = false;

  for (const evt of events) {
    if (evt.bar_index > barIndex) break;

    if (evt.type === "orb_range_update") {
      high = Number(evt.orb_high ?? 0);
      low = Number(evt.orb_low ?? 0);
      hasRange = true;
    } else if (evt.type === "orb_range_locked") {
      high = Number(evt.high ?? high);
      low = Number(evt.low ?? low);
      lockBarIndex = Number(evt.bar_index);
      hasRange = true;
    } else if (evt.type === "setup_valid") {
      direction = String(evt.direction ?? "");
    } else if (evt.type === "entry_signal") {
      entryPrice = Number(evt.price ?? 0);
      // Map side to direction for display
      const side = String(evt.side ?? "");
      if (!direction) {
        direction = side === "buy" ? "long" : side === "sell" ? "short" : null;
      }
    }
  }

  if (!hasRange) return null;

  let phase: ORState["phase"] = "forming";
  if (entryPrice !== null) {
    phase = "entry";
  } else if (direction) {
    phase = "breakout";
  } else if (lockBarIndex !== null) {
    phase = "locked";
  }

  return {
    high,
    low,
    range: high - low,
    phase,
    direction,
    lockBarIndex,
    entryPrice,
  };
}
