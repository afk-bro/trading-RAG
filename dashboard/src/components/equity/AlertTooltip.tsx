import type { AlertItem } from "@/api/types";
import { SEVERITY_COLORS } from "@/lib/chart-utils";
import { format } from "date-fns";
import { X } from "lucide-react";

interface Props {
  alert: AlertItem;
  x: number;
  y: number;
  onClose: () => void;
}

export function AlertTooltip({ alert, x, y, onClose }: Props) {
  return (
    <div
      className="absolute z-[1100] bg-bg-secondary border border-border rounded-lg shadow-lg p-3 w-64"
      style={{ left: x, top: y }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2 rounded-full"
            style={{ backgroundColor: SEVERITY_COLORS[alert.severity] }}
          />
          <span className="text-xs font-medium text-text-emphasis capitalize">
            {alert.severity}
          </span>
        </div>
        <button
          onClick={onClose}
          aria-label="Close alert tooltip"
          className="text-text-muted hover:text-foreground"
        >
          <X className="w-3 h-3" />
        </button>
      </div>

      <p className="text-sm text-foreground mb-1">{alert.rule_type}</p>

      {alert.last_triggered_at && (
        <p className="text-xs text-text-muted mb-1">
          Triggered: {format(new Date(alert.last_triggered_at), "MMM d, HH:mm")}
        </p>
      )}

      <p className="text-xs text-text-muted mb-1">
        Occurrences: {alert.occurrence_count}
      </p>

      {alert.payload && (
        <p className="text-xs text-text-muted truncate">
          {JSON.stringify(alert.payload).slice(0, 100)}
        </p>
      )}
    </div>
  );
}
