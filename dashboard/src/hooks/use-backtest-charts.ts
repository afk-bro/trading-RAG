import { useQueries } from "@tanstack/react-query";
import { getBacktestChartData } from "@/api/endpoints";
import type { BacktestChartData } from "@/api/types";

export function useBacktestCharts(idA: string | null, idB: string | null) {
  const results = useQueries({
    queries: [idA, idB].map((id) => ({
      queryKey: ["backtest-chart", id, 1, 50],
      queryFn: () => getBacktestChartData(id!, 1, 50),
      enabled: !!id,
      staleTime: 60_000,
    })),
  });

  return {
    a: results[0]?.data as BacktestChartData | undefined,
    b: results[1]?.data as BacktestChartData | undefined,
    isLoading: results.some((r) => r.isLoading),
    isError: results.some((r) => r.isError),
  };
}
