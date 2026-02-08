import { useState } from "react";
import { useTradeEvents } from "@/hooks/use-trade-events";
import { TradeTable } from "./TradeTable";
import { TradeDrawer } from "./TradeDrawer";
import type { TradeEventItem } from "@/api/types";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface Props {
  workspaceId: string;
  days: number;
}

const PAGE_SIZE = 20;

export function TradeExplorer({ workspaceId, days }: Props) {
  const [offset, setOffset] = useState(0);
  const [selectedEvent, setSelectedEvent] = useState<TradeEventItem | null>(
    null,
  );

  const { data, isLoading } = useTradeEvents(workspaceId, {
    days,
    limit: PAGE_SIZE,
    offset,
  });

  const total = data?.total ?? 0;
  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-text-emphasis">
          Trade Events
        </h3>
        <span className="text-xs text-text-muted">
          {total} event{total !== 1 ? "s" : ""}
        </span>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-48">
          <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          <TradeTable
            data={data?.items ?? []}
            onRowClick={setSelectedEvent}
          />

          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-border">
              <button
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0}
                className="flex items-center gap-1 px-2 py-1 text-xs text-text-muted
                           hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-3 h-3" /> Prev
              </button>
              <span className="text-xs text-text-muted">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total}
                className="flex items-center gap-1 px-2 py-1 text-xs text-text-muted
                           hover:text-foreground disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Next <ChevronRight className="w-3 h-3" />
              </button>
            </div>
          )}
        </>
      )}

      <TradeDrawer
        open={!!selectedEvent}
        onClose={() => setSelectedEvent(null)}
        event={selectedEvent}
        workspaceId={workspaceId}
      />
    </div>
  );
}
