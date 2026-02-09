import { useRef, useEffect } from "react";
import { createChart, type IChartApi, type Time } from "lightweight-charts";
import { toUnixSeconds } from "@/lib/chart-utils";
import type { BacktestChartEquityPoint } from "@/api/types";

interface Props {
  equityA: BacktestChartEquityPoint[];
  equityB: BacktestChartEquityPoint[];
  labelA: string;
  labelB: string;
}

function normalize(points: BacktestChartEquityPoint[]) {
  if (!points.length) return [];
  const first = points[0]!.equity;
  if (first === 0) return points.map((p) => ({ time: toUnixSeconds(p.t) as Time, value: 0 }));
  return points.map((p) => ({
    time: toUnixSeconds(p.t) as Time,
    value: (p.equity / first) * 100,
  }));
}

export function CompareEquityChart({ equityA, equityB, labelA, labelB }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 360,
      layout: {
        background: { color: "#0d1117" },
        textColor: "#8b949e",
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#21262d" },
        horzLines: { color: "#21262d" },
      },
      crosshair: { mode: 0 },
      rightPriceScale: { borderColor: "#30363d" },
      timeScale: { borderColor: "#30363d", timeVisible: true },
    });

    const seriesA = chart.addLineSeries({
      color: "#58a6ff",
      lineWidth: 2,
      priceScaleId: "right",
      title: labelA,
    });

    const seriesB = chart.addLineSeries({
      color: "#a371f7",
      lineWidth: 2,
      priceScaleId: "right",
      title: labelB,
    });

    seriesA.setData(normalize(equityA));
    seriesB.setData(normalize(equityB));
    chart.timeScale().fitContent();

    chartRef.current = chart;

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [equityA, equityB, labelA, labelB]);

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-text-emphasis">
          Equity Comparison (Normalized)
        </h3>
      </div>
      <div ref={containerRef} />
    </div>
  );
}
