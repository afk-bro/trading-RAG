import {
  useRef,
  useEffect,
  useState,
  useCallback,
  type RefObject,
} from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { toUnixSeconds } from "@/lib/chart-utils";
import type { EquityDataPoint } from "@/api/types";
import type { AlertItem } from "@/api/types";
import type { IntelSnapshot } from "@/api/types";
import { DrawdownToggle } from "./DrawdownToggle";
import { RegimeStrip } from "./RegimeStrip";
import { AlertTooltip } from "./AlertTooltip";
import { SEVERITY_COLORS } from "@/lib/chart-utils";

export interface TradeMarker {
  time: string;
  side: "long" | "short" | string;
}

interface Props {
  data: EquityDataPoint[];
  alerts?: AlertItem[];
  regimeSnapshots?: IntelSnapshot[];
  tradeMarkers?: TradeMarker[];
}

export function EquityChart({ data, alerts, regimeSnapshots, tradeMarkers }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const equitySeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const drawdownSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const [showDrawdown, setShowDrawdown] = useState(false);
  const [tooltipAlert, setTooltipAlert] = useState<{
    alert: AlertItem;
    x: number;
    y: number;
  } | null>(null);

  // Create chart once
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
      rightPriceScale: {
        borderColor: "#30363d",
      },
      timeScale: {
        borderColor: "#30363d",
        timeVisible: true,
      },
    });

    const equitySeries = chart.addLineSeries({
      color: "#58a6ff",
      lineWidth: 2,
      priceScaleId: "right",
    });

    const drawdownSeries = chart.addAreaSeries({
      topColor: "rgba(248, 81, 73, 0.35)",
      bottomColor: "rgba(248, 81, 73, 0.0)",
      lineColor: "#f85149",
      lineWidth: 1,
      priceScaleId: "drawdown",
      visible: false,
    });

    chart.priceScale("drawdown").applyOptions({
      scaleMargins: { top: 0.7, bottom: 0 },
      borderVisible: false,
    });

    chartRef.current = chart;
    equitySeriesRef.current = equitySeries;
    drawdownSeriesRef.current = drawdownSeries;

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
      equitySeriesRef.current = null;
      drawdownSeriesRef.current = null;
    };
  }, []);

  // Update equity data
  useEffect(() => {
    if (!equitySeriesRef.current || !data.length) return;

    const equityData = data.map((d) => ({
      time: toUnixSeconds(d.snapshot_ts) as Time,
      value: d.equity,
    }));

    equitySeriesRef.current.setData(equityData);

    const drawdownData = data.map((d) => ({
      time: toUnixSeconds(d.snapshot_ts) as Time,
      value: d.drawdown_pct * 100,
    }));

    drawdownSeriesRef.current?.setData(drawdownData);

    chartRef.current?.timeScale().fitContent();
  }, [data]);

  // Toggle drawdown visibility
  useEffect(() => {
    drawdownSeriesRef.current?.applyOptions({ visible: showDrawdown });
  }, [showDrawdown]);

  // Alert + trade markers
  useEffect(() => {
    if (!equitySeriesRef.current || !data.length) {
      equitySeriesRef.current?.setMarkers([]);
      return;
    }

    const allMarkers: SeriesMarker<Time>[] = [];
    const equityTimes = data.map((d) => toUnixSeconds(d.snapshot_ts));

    // Alert markers
    if (alerts?.length) {
      for (const a of alerts) {
        if (!a.last_triggered_at) continue;
        const alertTime = toUnixSeconds(a.last_triggered_at);
        let closest = equityTimes[0]!;
        let minDist = Math.abs(alertTime - closest);
        for (const t of equityTimes) {
          const dist = Math.abs(alertTime - t);
          if (dist < minDist) {
            minDist = dist;
            closest = t;
          }
        }

        const isCritical =
          a.severity === "critical" || a.severity === "high";

        allMarkers.push({
          time: closest as Time,
          position: "aboveBar" as const,
          shape: isCritical
            ? ("arrowDown" as const)
            : ("circle" as const),
          color: SEVERITY_COLORS[a.severity] ?? "#8b949e",
          text: a.rule_type,
          id: a.id,
        });
      }
    }

    // Trade entry markers (backtest)
    if (tradeMarkers?.length) {
      for (const tm of tradeMarkers) {
        const tmTime = toUnixSeconds(tm.time);
        let closest = equityTimes[0]!;
        let minDist = Math.abs(tmTime - closest);
        for (const t of equityTimes) {
          const dist = Math.abs(tmTime - t);
          if (dist < minDist) {
            minDist = dist;
            closest = t;
          }
        }

        const isLong = tm.side === "long";
        allMarkers.push({
          time: closest as Time,
          position: isLong ? ("belowBar" as const) : ("aboveBar" as const),
          shape: isLong
            ? ("arrowUp" as const)
            : ("arrowDown" as const),
          color: isLong ? "#3fb950" : "#f85149",
          text: isLong ? "L" : "S",
        });
      }
    }

    allMarkers.sort((a, b) => (a.time as number) - (b.time as number));
    equitySeriesRef.current.setMarkers(allMarkers);
  }, [alerts, data, tradeMarkers]);

  const handleMarkerClick = useCallback(
    (_e: React.MouseEvent) => {
      if (tooltipAlert) {
        setTooltipAlert(null);
      }
    },
    [tooltipAlert],
  );

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-text-emphasis">
          Equity Curve
        </h3>
        <DrawdownToggle
          active={showDrawdown}
          onToggle={() => setShowDrawdown((s) => !s)}
        />
      </div>

      <div className="relative" onClick={handleMarkerClick}>
        {regimeSnapshots && regimeSnapshots.length > 0 && chartRef.current && (
          <RegimeStrip
            snapshots={regimeSnapshots}
            chart={chartRef.current}
            chartContainer={containerRef as RefObject<HTMLDivElement>}
          />
        )}
        <div ref={containerRef} />
        {tooltipAlert && (
          <AlertTooltip
            alert={tooltipAlert.alert}
            x={tooltipAlert.x}
            y={tooltipAlert.y}
            onClose={() => setTooltipAlert(null)}
          />
        )}
      </div>
    </div>
  );
}
