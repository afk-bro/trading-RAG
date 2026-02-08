import { useEffect, useState, useCallback, type RefObject } from "react";
import type { IChartApi, Time } from "lightweight-charts";
import type { IntelSnapshot } from "@/api/types";
import { toUnixSeconds, regimeColor, regimeColorSolid } from "@/lib/chart-utils";

interface Segment {
  regime: string;
  left: number;
  width: number;
}

interface Props {
  snapshots: IntelSnapshot[];
  chart: IChartApi;
  chartContainer: RefObject<HTMLDivElement>;
}

export function RegimeStrip({ snapshots, chart, chartContainer }: Props) {
  const [segments, setSegments] = useState<Segment[]>([]);

  const recompute = useCallback(() => {
    if (!snapshots.length || !chartContainer.current) return;

    const timeScale = chart.timeScale();
    const sorted = [...snapshots].sort(
      (a, b) => toUnixSeconds(a.as_of_ts) - toUnixSeconds(b.as_of_ts),
    );

    const newSegments: Segment[] = [];

    for (let i = 0; i < sorted.length; i++) {
      const snap = sorted[i]!;
      const startTime = toUnixSeconds(snap.as_of_ts) as Time;
      const endTime =
        i < sorted.length - 1
          ? (toUnixSeconds(sorted[i + 1]!.as_of_ts) as Time)
          : null;

      const startX = timeScale.timeToCoordinate(startTime);
      if (startX === null) continue;

      let endX: number;
      if (endTime) {
        const ex = timeScale.timeToCoordinate(endTime);
        if (ex === null) continue;
        endX = ex;
      } else {
        endX = chartContainer.current.clientWidth;
      }

      const width = endX - startX;
      if (width > 0) {
        newSegments.push({
          regime: snap.regime,
          left: startX,
          width,
        });
      }
    }

    setSegments(newSegments);
  }, [snapshots, chart, chartContainer]);

  useEffect(() => {
    recompute();

    const timeScale = chart.timeScale();
    timeScale.subscribeVisibleTimeRangeChange(recompute);
    timeScale.subscribeSizeChange(recompute);

    return () => {
      timeScale.unsubscribeVisibleTimeRangeChange(recompute);
      timeScale.unsubscribeSizeChange(recompute);
    };
  }, [chart, recompute]);

  return (
    <div className="absolute inset-0 pointer-events-none" style={{ zIndex: 1 }}>
      {/* Legend */}
      <div className="absolute top-2 right-12 flex gap-2 z-10">
        {Array.from(new Set(segments.map((s) => s.regime))).map((regime) => (
          <div key={regime} className="flex items-center gap-1">
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: regimeColorSolid(regime) }}
            />
            <span className="text-[10px] text-text-muted capitalize">
              {regime}
            </span>
          </div>
        ))}
      </div>

      {/* Colored bands */}
      {segments.map((seg, i) => (
        <div
          key={i}
          className="absolute top-0 bottom-0"
          style={{
            left: seg.left,
            width: seg.width,
            backgroundColor: regimeColor(seg.regime),
          }}
        />
      ))}
    </div>
  );
}
