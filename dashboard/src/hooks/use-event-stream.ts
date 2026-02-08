import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { apiPost } from "@/api/client";
import type { SSETicketResponse } from "@/api/types";

/**
 * SSE hook: obtains a ticket, sets cookie, opens EventSource,
 * invalidates alerts/summary on events, reconnects with backoff.
 */
export function useEventStream(workspaceId: string | null) {
  const queryClient = useQueryClient();
  const esRef = useRef<EventSource | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryDelay = useRef(5_000);

  useEffect(() => {
    if (!workspaceId) return;

    let cancelled = false;

    async function connect() {
      try {
        const { ticket } = await apiPost<SSETicketResponse>(
          "/admin/events/ticket",
          undefined,
        );

        // Set cookie for EventSource auth
        document.cookie = `sse_ticket=${ticket}; path=/; SameSite=Lax`;

        if (cancelled) return;

        const url = `/admin/events/stream?workspace_id=${workspaceId}&topics=alerts,coverage,backtests`;
        const es = new EventSource(url);
        esRef.current = es;

        es.onopen = () => {
          retryDelay.current = 5_000; // reset backoff
        };

        es.onmessage = () => {
          queryClient.invalidateQueries({ queryKey: ["alerts", workspaceId] });
          queryClient.invalidateQueries({ queryKey: ["summary", workspaceId] });
        };

        es.onerror = () => {
          es.close();
          esRef.current = null;
          if (!cancelled) {
            retryRef.current = setTimeout(() => {
              retryDelay.current = Math.min(retryDelay.current * 2, 60_000);
              connect();
            }, retryDelay.current);
          }
        };
      } catch {
        // ticket fetch failed — retry
        if (!cancelled) {
          retryRef.current = setTimeout(() => {
            retryDelay.current = Math.min(retryDelay.current * 2, 60_000);
            connect();
          }, retryDelay.current);
        }
      }
    }

    connect();

    return () => {
      cancelled = true;
      esRef.current?.close();
      esRef.current = null;
      if (retryRef.current) clearTimeout(retryRef.current);
    };
  }, [workspaceId, queryClient]);
}
