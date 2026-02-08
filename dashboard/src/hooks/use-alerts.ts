import { useQuery } from "@tanstack/react-query";
import { getAlerts } from "@/api/endpoints";

export function useAlerts(
  workspaceId: string | null,
  days: number,
  includeResolved: boolean = true,
) {
  return useQuery({
    queryKey: ["alerts", workspaceId, days, includeResolved],
    queryFn: () => getAlerts(workspaceId!, days, includeResolved),
    enabled: !!workspaceId,
    staleTime: 30_000,
  });
}
