import { useRagContext } from "@/hooks/use-rag-context";
import { useQueryClient } from "@tanstack/react-query";
import type { TradeEventItem } from "@/api/types";
import { RefreshCw, ExternalLink } from "lucide-react";

interface Props {
  event: TradeEventItem;
  workspaceId: string;
  active: boolean;
}

export function TradeContext({ event, workspaceId, active }: Props) {
  const queryClient = useQueryClient();
  const { data, isLoading, isFetching } = useRagContext(
    workspaceId,
    event.symbol,
    event.event_time,
    active,
  );

  function rerun() {
    queryClient.invalidateQueries({
      queryKey: [
        "rag-context",
        workspaceId,
        event.symbol,
        event.event_time,
      ],
    });
  }

  if (!event.symbol) {
    return (
      <p className="text-sm text-text-muted py-4">
        No symbol associated with this event — RAG context unavailable.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-text-muted">
          Relevant knowledge for{" "}
          <span className="text-foreground font-medium">{event.symbol}</span>
        </p>
        <button
          onClick={rerun}
          disabled={isFetching}
          className="flex items-center gap-1 px-2 py-1 text-xs text-text-muted
                     hover:text-foreground disabled:opacity-40 transition-colors"
        >
          <RefreshCw
            className={`w-3 h-3 ${isFetching ? "animate-spin" : ""}`}
          />
          Re-run
        </button>
      </div>

      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-24 bg-bg-tertiary rounded animate-pulse"
            />
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {(data?.results ?? []).map((chunk) => (
            <div
              key={chunk.chunk_id}
              className="bg-background border border-border-subtle rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-text-emphasis truncate">
                  {chunk.source_title ?? "Untitled"}
                </span>
                <div className="flex items-center gap-2">
                  {/* Score bar */}
                  <div className="w-12 h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-accent rounded-full"
                      style={{ width: `${Math.min(chunk.score * 100, 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-text-muted">
                    {(chunk.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <p className="text-xs text-foreground line-clamp-4 leading-relaxed">
                {chunk.content}
              </p>
              {chunk.source_url && (
                <a
                  href={chunk.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 mt-2 text-[10px] text-accent hover:underline"
                >
                  Source <ExternalLink className="w-2.5 h-2.5" />
                </a>
              )}
            </div>
          ))}

          {data?.results?.length === 0 && (
            <p className="text-xs text-text-muted py-4 text-center">
              No relevant context found
            </p>
          )}
        </div>
      )}
    </div>
  );
}
