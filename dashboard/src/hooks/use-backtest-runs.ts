import { useQuery } from "@tanstack/react-query";
import { getBacktestRuns } from "@/api/endpoints";

export function useBacktestRuns(
  workspaceId: string | null,
  status?: string,
  limit: number = 25,
  offset: number = 0,
) {
  return useQuery({
    queryKey: ["backtest-runs", workspaceId, status, limit, offset],
    queryFn: () =>
      getBacktestRuns(workspaceId!, {
        status: status && status !== "all" ? status : undefined,
        limit,
        offset,
      }),
    enabled: !!workspaceId,
    staleTime: 30_000,
  });
}
