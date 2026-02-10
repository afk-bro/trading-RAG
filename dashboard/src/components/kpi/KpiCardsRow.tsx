import { KpiCard } from "./KpiCard";
import { formatCurrency, formatPercent } from "@/lib/chart-utils";
import type { DashboardSummary } from "@/api/types";

interface Props {
  summary: DashboardSummary | null | undefined;
  isLoading: boolean;
}

export function KpiCardsRow({ summary, isLoading }: Props) {
  const eq = summary?.equity;
  const denom = eq ? 1 - eq.drawdown_pct : 0;
  const startingEquity = eq && denom > 0 ? eq.peak_equity / denom : null;
  const netPnl =
    eq && startingEquity ? eq.equity - startingEquity : null;
  const totalReturn = eq
    ? startingEquity && startingEquity > 0
      ? (eq.equity - startingEquity) / startingEquity
      : null
    : null;

  const activeAlerts = summary?.alerts.total_active ?? 0;
  const worstSeverity =
    (summary?.alerts.by_severity.critical ?? 0) > 0
      ? "danger"
      : (summary?.alerts.by_severity.high ?? 0) > 0
        ? "danger"
        : (summary?.alerts.by_severity.medium ?? 0) > 0
          ? "warning"
          : "default";

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <KpiCard
        label="Net PnL"
        value={netPnl != null ? formatCurrency(netPnl) : "—"}
        color={netPnl != null ? (netPnl >= 0 ? "success" : "danger") : "default"}
        isLoading={isLoading}
      />
      <KpiCard
        label="Drawdown"
        value={eq ? formatPercent(eq.drawdown_pct) : "—"}
        color={eq && eq.drawdown_pct > 0.1 ? "danger" : "warning"}
        isLoading={isLoading}
      />
      <KpiCard
        label="Total Return"
        value={totalReturn != null ? formatPercent(totalReturn) : "—"}
        color={
          totalReturn != null
            ? totalReturn >= 0
              ? "success"
              : "danger"
            : "default"
        }
        isLoading={isLoading}
      />
      <KpiCard
        label="Active Alerts"
        value={String(activeAlerts)}
        color={activeAlerts > 0 ? (worstSeverity as "danger" | "warning" | "default") : "default"}
        isLoading={isLoading}
      />
    </div>
  );
}
