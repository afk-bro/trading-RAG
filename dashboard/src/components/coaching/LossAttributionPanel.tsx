import { useState } from "react";
import { cn } from "@/lib/utils";
import type { LossAttribution } from "@/api/types";
import { ChevronDown, ChevronUp } from "lucide-react";

interface Props {
  attribution: LossAttribution;
}

function HBar({
  label,
  value,
  maxValue,
  count,
}: {
  label: string;
  value: number;
  maxValue: number;
  count: number;
}) {
  const pct = maxValue > 0 ? (Math.abs(value) / maxValue) * 100 : 0;
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-text-muted">{label}</span>
        <span className="text-text-muted font-mono">
          {count} trades · ${Math.abs(value).toFixed(0)}
        </span>
      </div>
      <div className="h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
        <div
          className="h-full bg-danger/60 rounded-full"
          style={{ width: `${Math.min(100, pct)}%` }}
        />
      </div>
    </div>
  );
}

export function LossAttributionPanel({ attribution }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (attribution.total_losses === 0) {
    return null;
  }

  const maxTimeLoss = Math.max(
    ...attribution.time_clusters.map((c) => Math.abs(c.total_loss)),
    1,
  );
  const maxSizeLoss = Math.max(
    ...attribution.size_clusters.map((c) => Math.abs(c.total_loss)),
    1,
  );

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm
                   font-medium text-text-emphasis hover:bg-bg-tertiary/50
                   transition-colors"
      >
        <span>
          Loss Analysis ({attribution.total_losses} losing trade
          {attribution.total_losses !== 1 ? "s" : ""})
        </span>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-text-muted" />
        ) : (
          <ChevronDown className="w-4 h-4 text-text-muted" />
        )}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Time-of-day clusters */}
          {attribution.time_clusters.length > 0 && (
            <div>
              <p className="text-xs font-medium text-text-muted mb-2">
                Loss by hour of day
              </p>
              <div className="space-y-1.5">
                {attribution.time_clusters.map((c) => (
                  <HBar
                    key={c.label}
                    label={c.label}
                    value={c.total_loss}
                    maxValue={maxTimeLoss}
                    count={c.trade_count}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Size clusters */}
          {attribution.size_clusters.length > 0 && (
            <div>
              <p className="text-xs font-medium text-text-muted mb-2">
                Loss by trade size
              </p>
              <div className="space-y-1.5">
                {attribution.size_clusters.map((c) => (
                  <HBar
                    key={c.label}
                    label={c.label}
                    value={c.total_loss}
                    maxValue={maxSizeLoss}
                    count={c.trade_count}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Regime context */}
          {attribution.regime_summary && (
            <div>
              <p className="text-xs font-medium text-text-muted mb-1">
                Regime context
              </p>
              <p className="text-xs text-text-muted">
                {attribution.regime_summary.context}
              </p>
            </div>
          )}

          {/* Policy counterfactuals */}
          {attribution.counterfactuals.length > 0 && (
            <div>
              <p className="text-xs font-medium text-text-muted mb-2">
                What-if scenarios
              </p>
              <div className="space-y-2">
                {attribution.counterfactuals.map((cf, i) => (
                  <div
                    key={i}
                    className={cn(
                      "p-2.5 rounded border text-xs",
                      cf.delta > 0
                        ? "border-success/20 bg-success/5"
                        : "border-border bg-bg-tertiary/50",
                    )}
                  >
                    <p className="text-text-emphasis">{cf.description}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] px-1.5 py-0.5 bg-bg-tertiary rounded text-text-muted">
                        Hypothetical
                      </span>
                      <span className="font-mono text-text-muted">
                        {cf.metric_name}: {cf.actual.toFixed(2)} →{" "}
                        {cf.hypothetical.toFixed(2)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
