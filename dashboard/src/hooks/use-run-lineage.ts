import { useQuery } from "@tanstack/react-query";
import { getRunLineage } from "@/api/endpoints";

export function useRunLineage(
  workspaceId: string | null,
  runId: string | null,
) {
  return useQuery({
    queryKey: ["run-lineage", workspaceId, runId],
    queryFn: () => getRunLineage(workspaceId!, runId!),
    enabled: !!workspaceId && !!runId,
    staleTime: 60_000,
  });
}
