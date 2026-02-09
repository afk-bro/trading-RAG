import { useMemo } from "react";
import type {
  BacktestChartSummary,
  KpiDelta,
  TrajectoryRun,
} from "@/api/types";
import { DeltaKpiCard } from "./DeltaKpiCard";

interface Props {
  summary: BacktestChartSummary;
  deltas?: KpiDelta[];
  trajectory?: TrajectoryRun[];
  isLoading?: boolean;
}

function fmtPct(n: number | undefined | null): string {
  if (n == null) return "\u2014";
  return `${(n * 100).toFixed(2)}%`;
}

function fmtNum(n: number | undefined | null, decimals = 2): string {
  if (n == null) return "\u2014";
  return n.toFixed(decimals);
}

interface MetricConfig {
  label: string;
  key: keyof BacktestChartSummary;
  trajectoryKey?: keyof TrajectoryRun;
  format: (v: number | undefined | null) => string;
  color?: (v: number | undefined | null) => "default" | "success" | "danger";
}

const METRICS: MetricConfig[] = [
  {
    label: "Net Return",
    key: "return_pct",
    trajectoryKey: "return_pct",
    format: fmtPct,
    color: (v) => (v != null ? (v >= 0 ? "success" : "danger") : "default"),
  },
  {
    label: "Max Drawdown",
    key: "max_drawdown_pct",
    trajectoryKey: "max_drawdown_pct",
    format: fmtPct,
    color: () => "danger",
  },
  {
    label: "Sharpe",
    key: "sharpe",
    trajectoryKey: "sharpe",
    format: (v) => fmtNum(v),
  },
  {
    label: "Win Rate",
    key: "win_rate",
    trajectoryKey: "win_rate",
    format: fmtPct,
  },
  {
    label: "Profit Factor",
    key: "profit_factor",
    format: (v) => fmtNum(v),
  },
  {
    label: "Trades",
    key: "trades",
    trajectoryKey: "trades",
    format: (v) => fmtNum(v, 0),
  },
];

export function DeltaKpiStrip({ summary, deltas, trajectory, isLoading }: Props) {
  const deltaMap = useMemo(() => {
    const map = new Map<string, KpiDelta>();
    if (deltas) {
      for (const d of deltas) {
        map.set(d.metric, d);
      }
    }
    return map;
  }, [deltas]);

  const sparklineMap = useMemo(() => {
    const map = new Map<string, number[]>();
    if (!trajectory || trajectory.length < 2) return map;

    for (const m of METRICS) {
      if (!m.trajectoryKey) continue;
      const values = trajectory
        .map((r) => r[m.trajectoryKey!] as number | undefined)
        .filter((v): v is number => v != null);
      if (values.length >= 2) {
        map.set(m.key as string, values);
      }
    }
    return map;
  }, [trajectory]);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      {METRICS.map((m) => {
        const val = summary[m.key] as number | undefined | null;
        const color = m.color ? m.color(val) : "default";
        return (
          <DeltaKpiCard
            key={m.key}
            label={m.label}
            value={m.format(val)}
            delta={deltaMap.get(m.key as string)}
            sparkline={sparklineMap.get(m.key as string)}
            color={color}
            isLoading={isLoading}
          />
        );
      })}
    </div>
  );
}
