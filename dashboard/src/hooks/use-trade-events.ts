import { useQuery } from "@tanstack/react-query";
import { getTradeEvents } from "@/api/endpoints";

export function useTradeEvents(
  workspaceId: string | null,
  params: {
    days?: number;
    limit?: number;
    offset?: number;
    event_type?: string;
    symbol?: string;
  } = {},
) {
  return useQuery({
    queryKey: ["trade-events", workspaceId, params],
    queryFn: () => getTradeEvents(workspaceId!, params),
    enabled: !!workspaceId,
    staleTime: 30_000,
  });
}
