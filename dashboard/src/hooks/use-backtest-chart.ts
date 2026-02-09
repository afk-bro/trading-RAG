import { useQuery } from "@tanstack/react-query";
import { getBacktestChartData } from "@/api/endpoints";

export function useBacktestChart(
  runId: string | null,
  page: number = 1,
  pageSize: number = 50,
) {
  return useQuery({
    queryKey: ["backtest-chart", runId, page, pageSize],
    queryFn: () => getBacktestChartData(runId!, page, pageSize),
    enabled: !!runId,
    staleTime: 60_000,
  });
}
