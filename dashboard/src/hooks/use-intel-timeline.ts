import { useQuery } from "@tanstack/react-query";
import { getIntelTimeline } from "@/api/endpoints";

export function useIntelTimeline(workspaceId: string | null, days: number) {
  return useQuery({
    queryKey: ["intel-timeline", workspaceId, days],
    queryFn: () => getIntelTimeline(workspaceId!, days),
    enabled: !!workspaceId,
    staleTime: 30_000,
  });
}
