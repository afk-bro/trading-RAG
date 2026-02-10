import type { BacktestChartSummary } from "@/api/types";
import { cn } from "@/lib/utils";

interface Props {
  summaryA: BacktestChartSummary;
  summaryB: BacktestChartSummary;
  labelA: string;
  labelB: string;
}

interface KpiRow {
  label: string;
  key: keyof BacktestChartSummary;
  format: (v: unknown) => string;
  higherIsBetter: boolean;
}

function fmtPct(v: unknown): string {
  if (v == null) return "—";
  return `${((v as number) * 100).toFixed(2)}%`;
}

function fmtNum(v: unknown, decimals = 2): string {
  if (v == null) return "—";
  return (v as number).toFixed(decimals);
}

function fmtInt(v: unknown): string {
  if (v == null) return "—";
  return (v as number).toFixed(0);
}

const ROWS: KpiRow[] = [
  { label: "Net Return", key: "return_pct", format: fmtPct, higherIsBetter: true },
  { label: "Max Drawdown", key: "max_drawdown_pct", format: fmtPct, higherIsBetter: false },
  { label: "Sharpe", key: "sharpe", format: (v) => fmtNum(v), higherIsBetter: true },
  { label: "Win Rate", key: "win_rate", format: fmtPct, higherIsBetter: true },
  { label: "Profit Factor", key: "profit_factor", format: (v) => fmtNum(v), higherIsBetter: true },
  { label: "Trades", key: "trades", format: fmtInt, higherIsBetter: false },
  { label: "Avg Trade", key: "avg_trade_pct", format: fmtPct, higherIsBetter: true },
  { label: "Buy & Hold", key: "buy_hold_return_pct", format: fmtPct, higherIsBetter: true },
];

function deltaColor(a: unknown, b: unknown, higherIsBetter: boolean): string {
  if (a == null || b == null) return "";
  const diff = (a as number) - (b as number);
  if (diff === 0) return "";
  const aWins = higherIsBetter ? diff > 0 : diff < 0;
  return aWins ? "text-success" : "text-danger";
}

function formatDelta(a: unknown, b: unknown, format: (v: unknown) => string): string {
  if (a == null || b == null) return "—";
  const diff = (a as number) - (b as number);
  const sign = diff > 0 ? "+" : "";
  return sign + format(diff);
}

export function CompareKpiTable({ summaryA, summaryB, labelA, labelB }: Props) {
  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-text-emphasis">
          KPI Comparison
        </h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th scope="col" className="text-left px-3 py-2 text-xs font-medium text-text-muted">
                Metric
              </th>
              <th scope="col" className="text-right px-3 py-2 text-xs font-medium text-text-muted">
                {labelA}
              </th>
              <th scope="col" className="text-right px-3 py-2 text-xs font-medium text-text-muted">
                {labelB}
              </th>
              <th scope="col" className="text-right px-3 py-2 text-xs font-medium text-text-muted">
                Delta (A−B)
              </th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => {
              const valA = summaryA[row.key];
              const valB = summaryB[row.key];
              return (
                <tr
                  key={row.key}
                  className="border-b border-border-subtle"
                >
                  <td className="px-3 py-2 font-medium text-text-emphasis">
                    {row.label}
                  </td>
                  <td className="px-3 py-2 text-right font-mono">
                    {row.format(valA)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono">
                    {row.format(valB)}
                  </td>
                  <td
                    className={cn(
                      "px-3 py-2 text-right font-mono",
                      deltaColor(valA, valB, row.higherIsBetter),
                    )}
                  >
                    {formatDelta(valA, valB, row.format)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
