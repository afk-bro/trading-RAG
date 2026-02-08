import { useQuery } from "@tanstack/react-query";
import { getEquity } from "@/api/endpoints";

export function useEquity(workspaceId: string | null, days: number) {
  return useQuery({
    queryKey: ["equity", workspaceId, days],
    queryFn: () => getEquity(workspaceId!, days),
    enabled: !!workspaceId,
    staleTime: 30_000,
  });
}
