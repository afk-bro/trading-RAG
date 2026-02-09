import { useSearchParams, Link } from "react-router-dom";
import { useBacktestCharts } from "@/hooks/use-backtest-charts";
import { CompareKpiTable } from "@/components/compare/CompareKpiTable";
import { CompareEquityChart } from "@/components/compare/CompareEquityChart";
import { ConfigDiffPanel } from "@/components/compare/ConfigDiffPanel";
import { Skeleton } from "@/components/Skeleton";
import { ErrorAlert } from "@/components/ErrorAlert";
import { ArrowLeft, GitCompareArrows, ArrowRight } from "lucide-react";

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

  const { a, b, isLoading, isError, refetch } = useBacktestCharts(idA, idB);

  if (!idA || !idB) {
    return (
      <div className="py-16 text-center space-y-3">
        <GitCompareArrows className="w-10 h-10 text-text-muted mx-auto" />
        <p className="text-sm font-medium text-text-muted">
          Select two runs to compare
        </p>
        <Link
          to="/backtests"
          className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
        >
          Go to backtests <ArrowRight className="w-3 h-3" />
        </Link>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-48" />
        <Skeleton className="h-[400px]" />
        <Skeleton className="h-12" />
      </div>
    );
  }

  if (isError || !a || !b) {
    return (
      <ErrorAlert
        message="Failed to load one or both runs"
        onRetry={() => refetch()}
      />
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
