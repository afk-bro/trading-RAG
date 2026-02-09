import { useQuery } from "@tanstack/react-query";
import { getRunDetail } from "@/api/endpoints";

export function useRunDetail(workspaceId: string | null, runId: string | null) {
  return useQuery({
    queryKey: ["run-detail", workspaceId, runId],
    queryFn: () => getRunDetail(workspaceId!, runId!),
    enabled: !!workspaceId && !!runId,
    staleTime: 60_000,
  });
}
