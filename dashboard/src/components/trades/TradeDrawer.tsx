import { useState, useEffect } from "react";
import type { TradeEventItem } from "@/api/types";
import { TradeDetails } from "./TradeDetails";
import { TradeContext } from "./TradeContext";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  open: boolean;
  onClose: () => void;
  event: TradeEventItem | null;
  workspaceId: string;
}

type Tab = "details" | "context";

export function TradeDrawer({ open, onClose, event, workspaceId }: Props) {
  const [tab, setTab] = useState<Tab>("details");

  // Reset tab when event changes
  useEffect(() => {
    setTab("details");
  }, [event?.id]);

  if (!open || !event) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-[500] bg-black/50"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed top-0 right-0 z-[501] h-full w-full max-w-[480px] bg-bg-secondary border-l border-border overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-bg-secondary border-b border-border px-4 py-3 z-10">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-text-emphasis">
              {event.event_type}
              {event.symbol && (
                <span className="ml-2 text-text-muted">{event.symbol}</span>
              )}
            </h3>
            <button
              onClick={onClose}
              className="text-text-muted hover:text-foreground p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex gap-1">
            {(["details", "context"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-md transition-colors capitalize",
                  tab === t
                    ? "bg-bg-tertiary text-text-emphasis"
                    : "text-text-muted hover:text-foreground",
                )}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-4">
          {tab === "details" ? (
            <TradeDetails event={event} workspaceId={workspaceId} />
          ) : (
            <TradeContext
              event={event}
              workspaceId={workspaceId}
              active={tab === "context"}
            />
          )}
        </div>
      </div>
    </>
  );
}
