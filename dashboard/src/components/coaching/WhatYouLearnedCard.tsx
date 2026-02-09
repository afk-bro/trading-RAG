import { useState } from "react";
import { cn } from "@/lib/utils";
import type { LineageData, KpiDelta } from "@/api/types";
import { AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";

interface Props {
  lineage: LineageData;
}

const METRIC_LABELS: Record<string, string> = {
  return_pct: "Net Return",
  max_drawdown_pct: "Max Drawdown",
  sharpe: "Sharpe",
  win_rate: "Win Rate",
  profit_factor: "Profit Factor",
  trades: "Trades",
  avg_trade_pct: "Avg Trade",
  buy_hold_return_pct: "Buy & Hold",
};

function formatDelta(d: KpiDelta): string {
  if (d.delta == null) return "";
  const abs = Math.abs(d.delta);
  const sign = d.delta > 0 ? "+" : "-";
  if (d.metric.includes("pct") || d.metric === "win_rate") {
    return `${sign}${(abs * 100).toFixed(2)}%`;
  }
  return `${sign}${abs.toFixed(2)}`;
}

function DeltaItem({ delta }: { delta: KpiDelta }) {
  const label = METRIC_LABELS[delta.metric] ?? delta.metric;
  return (
    <div className="flex items-center justify-between text-xs py-0.5">
      <span className="text-text-muted">{label}</span>
      <span className="font-mono">{formatDelta(delta)}</span>
    </div>
  );
}

function DeltaColumn({
  title,
  deltas,
  borderColor,
}: {
  title: string;
  deltas: KpiDelta[];
  borderColor: string;
}) {
  if (deltas.length === 0) return null;

  return (
    <div className={cn("border-l-2 pl-3 min-w-0", borderColor)}>
      <p className="text-xs font-medium text-text-emphasis mb-1">{title}</p>
      {deltas.map((d) => (
        <DeltaItem key={d.metric} delta={d} />
      ))}
    </div>
  );
}

export function WhatYouLearnedCard({ lineage }: Props) {
  const [showParamDiff, setShowParamDiff] = useState(false);

  // First-run state: no previous run
  if (!lineage.previous_run_id) {
    return (
      <div className="bg-bg-secondary border border-border rounded-lg p-4">
        <h3 className="text-sm font-medium text-text-emphasis mb-1">
          What You Learned
        </h3>
        <p className="text-xs text-text-muted">
          Baseline captured. Your next run will show progress deltas
          automatically.
        </p>
      </div>
    );
  }

  // Categorize deltas
  const improved: KpiDelta[] = [];
  const tradeoffs: KpiDelta[] = [];
  const unchanged: KpiDelta[] = [];

  for (const d of lineage.deltas) {
    if (d.delta == null || d.improved == null) {
      unchanged.push(d);
    } else if (d.improved) {
      improved.push(d);
    } else {
      tradeoffs.push(d);
    }
  }

  const paramDiffEntries = Object.entries(lineage.param_diffs);

  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-text-emphasis">
          What You Learned
        </h3>
        {lineage.params_changed && (
          <button
            onClick={() => setShowParamDiff(!showParamDiff)}
            className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium
                       bg-accent/10 text-accent rounded-full hover:bg-accent/20
                       transition-colors"
          >
            Config changed
            {showParamDiff ? (
              <ChevronUp className="w-3 h-3" />
            ) : (
              <ChevronDown className="w-3 h-3" />
            )}
          </button>
        )}
      </div>

      {/* Comparison warnings */}
      {lineage.comparison_warnings.length > 0 && (
        <div className="flex items-start gap-1.5 mb-3 p-2 bg-warning/5 border border-warning/10 rounded">
          <AlertTriangle className="w-3 h-3 text-warning flex-shrink-0 mt-0.5" />
          <p className="text-[10px] text-warning">
            Not directly comparable: {lineage.comparison_warnings.join("; ")}
          </p>
        </div>
      )}

      {/* Three-column delta display */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <DeltaColumn
          title="Improved"
          deltas={improved}
          borderColor="border-success"
        />
        <DeltaColumn
          title="Tradeoffs"
          deltas={tradeoffs}
          borderColor="border-warning"
        />
        <DeltaColumn
          title="Unchanged"
          deltas={unchanged}
          borderColor="border-border"
        />
      </div>

      {/* Expandable param diff */}
      {showParamDiff && paramDiffEntries.length > 0 && (
        <div className="mt-3 pt-2 border-t border-border-subtle">
          <p className="text-[10px] font-medium text-text-muted mb-1">
            Parameter changes
          </p>
          <div className="space-y-0.5">
            {paramDiffEntries.map(([key, [oldVal, newVal]]) => (
              <div key={key} className="flex gap-2 text-[10px] font-mono">
                <span className="text-text-muted">{key}:</span>
                <span className="text-danger line-through">{oldVal}</span>
                <span className="text-success">{newVal}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
