import { useState, useEffect, useCallback, useRef } from "react";
import type { RunEvent } from "@/api/types";
import { cn } from "@/lib/utils";
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Rewind,
  Clock,
} from "lucide-react";

const EVENT_COLORS: Record<string, string> = {
  orb_range_update: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  orb_range_locked: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  setup_valid: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  entry_signal: "bg-purple-500/20 text-purple-400 border-purple-500/30",
};

const EVENT_LABELS: Record<string, string> = {
  orb_range_update: "OR Forming",
  orb_range_locked: "OR Locked",
  setup_valid: "Setup Valid",
  entry_signal: "Entry Signal",
};

const EVENT_DESCRIPTIONS: Record<string, string> = {
  orb_range_update: "Opening range is still forming. High/low being tracked.",
  orb_range_locked: "Opening range window closed. Range is now fixed.",
  setup_valid: "Confirmed breakout of opening range level.",
  entry_signal: "Trade entry triggered.",
};

const SPEEDS = [0.5, 1, 2, 4] as const;

interface ReplayPanelProps {
  events: RunEvent[];
  maxBarIndex: number;
  onBarIndexChange?: (barIndex: number) => void;
}

export function ReplayPanel({
  events,
  maxBarIndex,
  onBarIndexChange,
}: ReplayPanelProps) {
  const [barIndex, setBarIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1); // Default 1x
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const speed = SPEEDS[speedIdx] ?? 1;

  // Events up to current bar
  const visibleEvents = events.filter((e) => e.bar_index <= barIndex);
  const currentEvent = visibleEvents.length > 0 ? visibleEvents[visibleEvents.length - 1] : null;

  // Next event (for "what's coming" hint)
  const nextEvent = events.find((e) => e.bar_index > barIndex) ?? null;

  const updateBar = useCallback(
    (idx: number) => {
      const clamped = Math.max(0, Math.min(idx, maxBarIndex));
      setBarIndex(clamped);
      onBarIndexChange?.(clamped);
    },
    [maxBarIndex, onBarIndexChange],
  );

  // Play/pause timer
  useEffect(() => {
    if (playing) {
      const ms = Math.round(200 / speed);
      intervalRef.current = setInterval(() => {
        setBarIndex((prev) => {
          const next = prev + 1;
          if (next > maxBarIndex) {
            setPlaying(false);
            return prev;
          }
          onBarIndexChange?.(next);
          return next;
        });
      }, ms);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, speed, maxBarIndex, onBarIndexChange]);

  function handleStep(delta: number) {
    setPlaying(false);
    updateBar(barIndex + delta);
  }

  function handleJumpToEvent(dir: "prev" | "next") {
    setPlaying(false);
    if (dir === "next" && nextEvent) {
      updateBar(nextEvent.bar_index);
    } else if (dir === "prev") {
      const prev = [...events]
        .reverse()
        .find((e) => e.bar_index < barIndex);
      if (prev) updateBar(prev.bar_index);
    }
  }

  function handleReset() {
    setPlaying(false);
    updateBar(0);
  }

  function cycleSpeed() {
    setSpeedIdx((i) => (i + 1) % SPEEDS.length);
  }

  return (
    <div className="space-y-4">
      {/* Transport controls */}
      <div className="flex items-center gap-2 bg-bg-secondary border border-border rounded-lg px-4 py-3">
        <button
          onClick={handleReset}
          className="p-1.5 rounded hover:bg-bg-tertiary text-text-muted hover:text-foreground transition-colors"
          title="Reset to start"
        >
          <Rewind className="w-4 h-4" />
        </button>
        <button
          onClick={() => handleJumpToEvent("prev")}
          className="p-1.5 rounded hover:bg-bg-tertiary text-text-muted hover:text-foreground transition-colors"
          title="Previous event"
        >
          <SkipBack className="w-4 h-4" />
        </button>
        <button
          onClick={() => handleStep(-1)}
          className="px-2 py-1 text-xs rounded border border-border text-text-muted hover:text-foreground hover:bg-bg-tertiary transition-colors"
        >
          -1
        </button>
        <button
          onClick={() => setPlaying((p) => !p)}
          className={cn(
            "p-2 rounded-full transition-colors",
            playing
              ? "bg-accent/20 text-accent hover:bg-accent/30"
              : "bg-bg-tertiary text-foreground hover:bg-accent/20 hover:text-accent",
          )}
          title={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </button>
        <button
          onClick={() => handleStep(1)}
          className="px-2 py-1 text-xs rounded border border-border text-text-muted hover:text-foreground hover:bg-bg-tertiary transition-colors"
        >
          +1
        </button>
        <button
          onClick={() => handleJumpToEvent("next")}
          className="p-1.5 rounded hover:bg-bg-tertiary text-text-muted hover:text-foreground transition-colors"
          title="Next event"
        >
          <SkipForward className="w-4 h-4" />
        </button>

        {/* Speed */}
        <button
          onClick={cycleSpeed}
          className="flex items-center gap-1 px-2 py-1 text-xs rounded border border-border text-text-muted hover:text-foreground hover:bg-bg-tertiary transition-colors ml-2"
          title="Cycle speed"
        >
          <Clock className="w-3 h-3" />
          {speed}x
        </button>

        {/* Bar slider */}
        <div className="flex-1 mx-3">
          <input
            type="range"
            min={0}
            max={maxBarIndex}
            value={barIndex}
            onChange={(e) => {
              setPlaying(false);
              updateBar(Number(e.target.value));
            }}
            className="w-full h-1.5 bg-bg-tertiary rounded-lg appearance-none cursor-pointer accent-accent"
          />
        </div>

        {/* Bar counter */}
        <span className="text-xs font-mono text-text-muted tabular-nums min-w-[80px] text-right">
          Bar {barIndex} / {maxBarIndex}
        </span>
      </div>

      {/* Coach panel: current + next event */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Current state */}
        <div className="bg-bg-secondary border border-border rounded-lg p-4">
          <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
            Current State
          </h4>
          {currentEvent ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "px-2 py-0.5 text-xs rounded-full border",
                    EVENT_COLORS[currentEvent.type] ?? "bg-bg-tertiary text-text-muted",
                  )}
                >
                  {EVENT_LABELS[currentEvent.type] ?? currentEvent.type}
                </span>
                <span className="text-xs text-text-muted font-mono">
                  bar {currentEvent.bar_index}
                </span>
              </div>
              <p className="text-sm text-foreground">
                {EVENT_DESCRIPTIONS[currentEvent.type] ?? ""}
              </p>
              {/* Event details */}
              <EventDetails event={currentEvent} />
            </div>
          ) : (
            <p className="text-sm text-text-muted">
              No events yet. Press play or advance bars.
            </p>
          )}
        </div>

        {/* What's next */}
        <div className="bg-bg-secondary border border-border rounded-lg p-4">
          <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide mb-2">
            What's Next
          </h4>
          {nextEvent ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span
                  className={cn(
                    "px-2 py-0.5 text-xs rounded-full border opacity-60",
                    EVENT_COLORS[nextEvent.type] ?? "bg-bg-tertiary text-text-muted",
                  )}
                >
                  {EVENT_LABELS[nextEvent.type] ?? nextEvent.type}
                </span>
                <span className="text-xs text-text-muted font-mono">
                  in {nextEvent.bar_index - barIndex} bar{nextEvent.bar_index - barIndex !== 1 ? "s" : ""}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-sm text-text-muted">
              No more events.
            </p>
          )}
        </div>
      </div>

      {/* Event timeline */}
      <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-2 border-b border-border">
          <h4 className="text-xs font-medium text-text-muted uppercase tracking-wide">
            Event Timeline ({events.length} events)
          </h4>
        </div>
        <div className="max-h-48 overflow-y-auto">
          {events.length === 0 ? (
            <p className="p-4 text-sm text-text-muted text-center">
              No events recorded for this run.
            </p>
          ) : (
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-bg-secondary">
                <tr className="border-b border-border text-text-muted">
                  <th className="text-left px-4 py-1.5 font-medium">Bar</th>
                  <th className="text-left px-4 py-1.5 font-medium">Time</th>
                  <th className="text-left px-4 py-1.5 font-medium">Event</th>
                </tr>
              </thead>
              <tbody>
                {events.map((event, i) => {
                  const isActive = event.bar_index <= barIndex;
                  const isCurrent =
                    currentEvent && event.bar_index === currentEvent.bar_index &&
                    event.type === currentEvent.type;
                  return (
                    <tr
                      key={i}
                      onClick={() => updateBar(event.bar_index)}
                      className={cn(
                        "border-b border-border/50 cursor-pointer transition-colors",
                        isCurrent
                          ? "bg-accent/10"
                          : isActive
                          ? "hover:bg-bg-tertiary"
                          : "opacity-40 hover:opacity-70",
                      )}
                    >
                      <td className="px-4 py-1.5 font-mono">{event.bar_index}</td>
                      <td className="px-4 py-1.5 font-mono text-text-muted">
                        {event.ts ? String(event.ts).slice(11, 19) : "—"}
                      </td>
                      <td className="px-4 py-1.5">
                        <span
                          className={cn(
                            "px-1.5 py-0.5 rounded text-[10px] border",
                            EVENT_COLORS[event.type] ?? "bg-bg-tertiary text-text-muted",
                          )}
                        >
                          {EVENT_LABELS[event.type] ?? event.type}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

function EventDetails({ event }: { event: RunEvent }) {
  // Display relevant fields based on event type
  const fields: [string, unknown][] = Object.entries(event).filter(
    ([k]) => !["type", "bar_index", "ts"].includes(k),
  );

  if (fields.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2">
      {fields.map(([key, val]) => (
        <div key={key} className="flex justify-between text-xs">
          <span className="text-text-muted">{key}</span>
          <span className="text-foreground font-mono">
            {typeof val === "number" ? val.toFixed(2) : String(val)}
          </span>
        </div>
      ))}
    </div>
  );
}
