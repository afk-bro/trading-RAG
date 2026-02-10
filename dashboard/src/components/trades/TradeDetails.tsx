import { useQuery } from "@tanstack/react-query";
import { getTradeEventDetail } from "@/api/endpoints";
import { format } from "date-fns";
import { formatCurrency } from "@/lib/chart-utils";
import type { TradeEventItem } from "@/api/types";
import { Field } from "@/components/ui/Field";

interface Props {
  event: TradeEventItem;
  workspaceId: string;
}

export function TradeDetails({ event, workspaceId }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ["trade-event-detail", workspaceId, event.id],
    queryFn: () => getTradeEventDetail(workspaceId, event.id),
    staleTime: 60_000,
  });

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="grid grid-cols-2 gap-3">
        <Field label="Type" value={event.event_type} />
        <Field label="Symbol" value={event.symbol ?? "—"} />
        <Field
          label="Side"
          value={event.side ?? "—"}
          className={
            event.side === "long"
              ? "text-success"
              : event.side === "short"
                ? "text-danger"
                : ""
          }
        />
        <Field
          label="PnL"
          value={event.pnl != null ? formatCurrency(event.pnl) : "—"}
          className={
            event.pnl != null
              ? event.pnl >= 0
                ? "text-success"
                : "text-danger"
              : ""
          }
        />
        <Field
          label="Entry"
          value={
            event.entry_price != null ? formatCurrency(event.entry_price) : "—"
          }
        />
        <Field
          label="Exit"
          value={
            event.exit_price != null ? formatCurrency(event.exit_price) : "—"
          }
        />
      </div>

      {/* Related events timeline */}
      <div>
        <h4 className="text-xs font-medium text-text-muted mb-2">
          Related Events
        </h4>
        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="h-10 bg-bg-tertiary rounded animate-pulse"
              />
            ))}
          </div>
        ) : (
          <div className="space-y-1">
            {(data?.related_events ?? []).map((re) => (
              <div
                key={re.id}
                className="flex items-center gap-2 px-3 py-2 rounded-md bg-background border border-border-subtle"
              >
                <div className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <span className="text-xs font-mono text-foreground">
                    {re.event_type}
                  </span>
                  {re.symbol && (
                    <span className="text-xs text-text-muted ml-2">
                      {re.symbol}
                    </span>
                  )}
                </div>
                <span className="text-[10px] text-text-muted flex-shrink-0">
                  {format(new Date(re.event_time), "HH:mm:ss")}
                </span>
              </div>
            ))}
            {(data?.related_events ?? []).length === 0 && (
              <p className="text-xs text-text-muted py-2">
                No related events
              </p>
            )}
          </div>
        )}
      </div>

      {/* Raw metadata */}
      {event.metadata && Object.keys(event.metadata).length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-text-muted mb-2">
            Metadata
          </h4>
          <pre className="text-[10px] text-text-muted bg-background border border-border-subtle rounded-md p-3 overflow-auto max-h-40">
            {JSON.stringify(event.metadata, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
