import { KpiCard } from "@/components/kpi/KpiCard";
import type { BacktestChartSummary } from "@/api/types";
import { fmtPct, fmtNum } from "@/lib/chart-utils";

interface Props {
  summary: BacktestChartSummary;
  isLoading?: boolean;
}

export function RunKpiStrip({ summary, isLoading }: Props) {
  const returnPct = summary.return_pct;
  const returnColor =
    returnPct != null ? (returnPct >= 0 ? "success" : "danger") : "default";

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
      <KpiCard
        label="Net Return"
        value={fmtPct(returnPct)}
        color={returnColor}
        isLoading={isLoading}
      />
      <KpiCard
        label="Max Drawdown"
        value={fmtPct(summary.max_drawdown_pct)}
        color="danger"
        isLoading={isLoading}
      />
      <KpiCard
        label="Sharpe"
        value={fmtNum(summary.sharpe)}
        isLoading={isLoading}
      />
      <KpiCard
        label="Win Rate"
        value={fmtPct(summary.win_rate)}
        isLoading={isLoading}
      />
      <KpiCard
        label="Profit Factor"
        value={fmtNum(summary.profit_factor)}
        isLoading={isLoading}
      />
      <KpiCard
        label="Trades"
        value={fmtNum(summary.trades, 0)}
        isLoading={isLoading}
      />
    </div>
  );
}
