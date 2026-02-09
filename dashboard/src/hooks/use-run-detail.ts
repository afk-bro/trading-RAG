import { useQuery } from "@tanstack/react-query";
import { getRunDetail } from "@/api/endpoints";

export function useRunDetail(
  workspaceId: string | null,
  runId: string | null,
  includeCoaching?: boolean,
  baselineRunId?: string | null,
) {
  return useQuery({
    queryKey: ["run-detail", workspaceId, runId, includeCoaching, baselineRunId],
    queryFn: () => getRunDetail(workspaceId!, runId!, includeCoaching, baselineRunId),
    enabled: !!workspaceId && !!runId,
    staleTime: 60_000,
  });
}
