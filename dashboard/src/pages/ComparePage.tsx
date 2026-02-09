import { useSearchParams, Link } from "react-router-dom";
import { useBacktestCharts } from "@/hooks/use-backtest-charts";
import { CompareKpiTable } from "@/components/compare/CompareKpiTable";
import { CompareEquityChart } from "@/components/compare/CompareEquityChart";
import { ConfigDiffPanel } from "@/components/compare/ConfigDiffPanel";
import { ArrowLeft } from "lucide-react";

function buildLabel(meta?: Record<string, unknown>): string {
  const symbol = (meta?.symbol as string) ?? "";
  const tf = (meta?.timeframe as string) ?? "";
  if (symbol && tf) return `${symbol} ${tf}`;
  if (symbol) return symbol;
  return "Run";
}

export function ComparePage() {
  const [searchParams] = useSearchParams();
  const idA = searchParams.get("a");
  const idB = searchParams.get("b");

  const { a, b, isLoading, isError } = useBacktestCharts(idA, idB);

  if (!idA || !idB) {
    return (
      <div className="text-center py-12 text-text-muted">
        Select two runs to compare.{" "}
        <Link to="/backtests" className="text-accent hover:underline">
          Back to runs
        </Link>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-64 bg-bg-tertiary rounded animate-pulse" />
        <div className="h-48 bg-bg-tertiary rounded animate-pulse" />
        <div className="h-[400px] bg-bg-tertiary rounded animate-pulse" />
        <div className="h-12 bg-bg-tertiary rounded animate-pulse" />
      </div>
    );
  }

  if (isError || !a || !b) {
    return (
      <div className="text-center py-12 text-text-muted">
        Failed to load one or both runs.{" "}
        <Link to="/backtests" className="text-accent hover:underline">
          Back to runs
        </Link>
      </div>
    );
  }

  const labelA = `A: ${buildLabel(a.dataset_meta)}`;
  const labelB = `B: ${buildLabel(b.dataset_meta)}`;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <Link
          to="/backtests"
          className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-foreground mb-2"
        >
          <ArrowLeft className="w-3 h-3" /> Back to runs
        </Link>
        <h2 className="text-lg font-semibold text-text-emphasis">
          Compare Runs
        </h2>
        <p className="text-xs text-text-muted mt-1">
          <span className="text-[#58a6ff]">{labelA}</span>
          {" vs "}
          <span className="text-[#a371f7]">{labelB}</span>
        </p>
      </div>

      {/* KPI comparison */}
      <CompareKpiTable
        summaryA={a.summary}
        summaryB={b.summary}
        labelA={labelA}
        labelB={labelB}
      />

      {/* Equity overlay */}
      <CompareEquityChart
        equityA={a.equity}
        equityB={b.equity}
        labelA={labelA}
        labelB={labelB}
      />

      {/* Config diff */}
      <ConfigDiffPanel
        paramsA={a.params}
        paramsB={b.params}
        labelA={labelA}
        labelB={labelB}
      />
    </div>
  );
}
