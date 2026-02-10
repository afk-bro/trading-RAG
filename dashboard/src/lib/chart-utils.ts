export function formatCurrency(n: number | null | undefined): string {
  if (n == null) return "—";
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });
}

export function formatPercent(n: number | null | undefined): string {
  if (n == null) return "—";
  return `${(n * 100).toFixed(2)}%`;
}

export function formatPnl(n: number | null | undefined): string {
  if (n == null) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}${formatCurrency(n)}`;
}

/** Alias for formatPercent — used by KPI strips and tables. */
export const fmtPct = formatPercent;

/** Short number formatter used by KPI strips and tables. */
export function fmtNum(n: number | undefined | null, decimals = 2): string {
  if (n == null) return "—";
  return n.toFixed(decimals);
}

/** Alias for formatPnl — used by KPI strips and tables. */
export const fmtPnl = formatPnl;

export function fmtPrice(n: number | undefined | null): string {
  if (n == null) return "—";
  return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 5 });
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

/**
 * Convert ISO timestamp to UTC unix seconds for Lightweight Charts.
 * LWC uses seconds, not milliseconds — this is the #1 silent bug.
 */
export function toUnixSeconds(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000);
}

export const REGIME_COLORS: Record<string, string> = {
  bullish: "#3fb95040",
  trending_up: "#3fb95040",
  bearish: "#f8514940",
  trending_down: "#f8514940",
  ranging: "#d2992240",
  volatile: "#a371f740",
};

export const REGIME_COLORS_SOLID: Record<string, string> = {
  bullish: "#3fb950",
  trending_up: "#3fb950",
  bearish: "#f85149",
  trending_down: "#f85149",
  ranging: "#d29922",
  volatile: "#a371f7",
};

export const SEVERITY_COLORS: Record<string, string> = {
  critical: "#f85149",
  high: "#f85149",
  medium: "#d29922",
  low: "#8b949e",
};

export function regimeColor(regime: string): string {
  return REGIME_COLORS[regime.toLowerCase()] ?? "#8b949e20";
}

export function regimeColorSolid(regime: string): string {
  return REGIME_COLORS_SOLID[regime.toLowerCase()] ?? "#8b949e";
}

/**
 * Map backtest equity points to EquityDataPoint format for EquityChart.
 * Computes drawdown and peak from raw equity values client-side.
 */
export function mapBacktestEquity(
  points: { t: string; equity: number }[],
): {
  snapshot_ts: string;
  computed_at: string;
  equity: number;
  cash: number;
  positions_value: number;
  realized_pnl: number;
  peak_equity: number;
  drawdown_pct: number;
}[] {
  let peak = 0;
  return points.map((p) => {
    if (p.equity > peak) peak = p.equity;
    const dd = peak > 0 ? (peak - p.equity) / peak : 0;
    return {
      snapshot_ts: p.t,
      computed_at: p.t,
      equity: p.equity,
      cash: 0,
      positions_value: 0,
      realized_pnl: 0,
      peak_equity: peak,
      drawdown_pct: dd,
    };
  });
}
