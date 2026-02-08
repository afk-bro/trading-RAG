import { useQuery } from "@tanstack/react-query";
import { getSummary } from "@/api/endpoints";

export function useSummary(workspaceId: string | null) {
  return useQuery({
    queryKey: ["summary", workspaceId],
    queryFn: () => getSummary(workspaceId!),
    enabled: !!workspaceId,
    staleTime: 30_000,
  });
}
