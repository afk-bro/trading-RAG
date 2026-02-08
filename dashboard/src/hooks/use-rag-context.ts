import { useQuery } from "@tanstack/react-query";
import { postQuery } from "@/api/endpoints";
import { format } from "date-fns";

export function useRagContext(
  workspaceId: string | null,
  symbol: string | null,
  eventTime: string | null,
  enabled: boolean = false,
) {
  const question = symbol
    ? `What trading context and analysis is relevant to ${symbol} around ${eventTime ? format(new Date(eventTime), "yyyy-MM-dd HH:mm") : "now"}?`
    : null;

  return useQuery({
    queryKey: ["rag-context", workspaceId, symbol, eventTime],
    queryFn: () =>
      postQuery(workspaceId!, question!, symbol ? { symbols: [symbol] } : undefined),
    enabled: !!workspaceId && !!question && enabled,
    staleTime: 60_000,
  });
}
