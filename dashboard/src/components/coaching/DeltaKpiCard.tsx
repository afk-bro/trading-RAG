import { cn } from "@/lib/utils";
import type { KpiDelta } from "@/api/types";
import { TrajectorySparkline } from "./TrajectorySparkline";

interface Props {
  label: string;
  value: string;
  delta?: KpiDelta;
  sparkline?: number[];
  color?: "default" | "success" | "danger" | "warning" | "accent";
  isLoading?: boolean;
}

const colorMap = {
  default: "text-foreground",
  success: "text-success",
  danger: "text-danger",
  warning: "text-warning",
  accent: "text-accent",
};

function formatDeltaValue(d: KpiDelta): string {
  if (d.delta == null) return "";
  const abs = Math.abs(d.delta);
  const sign = d.delta > 0 ? "+" : "-";
  // Format as percentage for pct metrics
  if (d.metric.includes("pct") || d.metric === "win_rate") {
    return `${sign}${(abs * 100).toFixed(2)}%`;
  }
  return `${sign}${abs.toFixed(2)}`;
}

function deltaColorClass(d: KpiDelta): string {
  if (d.improved === null || d.delta === null || d.delta === 0) {
    return "text-text-muted";
  }
  return d.improved ? "text-success" : "text-warning";
}

function deltaArrow(d: KpiDelta): string {
  if (d.delta === null || d.delta === 0) return "";
  return d.delta > 0 ? "\u2191" : "\u2193";
}

export function DeltaKpiCard({
  label,
  value,
  delta,
  sparkline,
  color = "default",
  isLoading,
}: Props) {
  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-4">
      <p className="text-xs text-text-muted mb-1">{label}</p>
      {isLoading ? (
        <div className="h-7 w-20 bg-bg-tertiary rounded animate-pulse" />
      ) : (
        <>
          <p className={cn("text-2xl font-semibold", colorMap[color])}>
            {value}
          </p>
          {delta && delta.delta !== null && delta.delta !== 0 && (
            <p className={cn("text-xs mt-0.5", deltaColorClass(delta))}>
              {deltaArrow(delta)} {formatDeltaValue(delta)}
            </p>
          )}
          {sparkline && sparkline.length >= 2 && (
            <div className="mt-1.5">
              <TrajectorySparkline
                values={sparkline}
                width={56}
                height={16}
                color="var(--color-text-muted)"
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
