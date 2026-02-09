import { cn } from "@/lib/utils";
import type { ProcessScore } from "@/api/types";

interface Props {
  score: ProcessScore;
}

const GRADE_COLORS: Record<string, string> = {
  A: "text-success",
  B: "text-accent",
  C: "text-warning",
  D: "text-orange-400",
  F: "text-text-muted",
  unavailable: "text-text-muted",
  timed_out: "text-text-muted",
};

const GRADE_STROKE: Record<string, string> = {
  A: "stroke-success",
  B: "stroke-accent",
  C: "stroke-warning",
  D: "stroke-orange-400",
  F: "stroke-text-muted",
  unavailable: "stroke-text-muted",
  timed_out: "stroke-text-muted",
};

function GaugeArc({ value, grade }: { value: number; grade: string }) {
  // SVG arc from ~225 degrees to ~-45 degrees (270 degree sweep)
  const radius = 40;
  const cx = 50;
  const cy = 50;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degrees
  const filledLength = (value / 100) * arcLength;

  return (
    <svg viewBox="0 0 100 100" className="w-24 h-24">
      {/* Background arc */}
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill="none"
        className="stroke-bg-tertiary"
        strokeWidth={6}
        strokeDasharray={`${arcLength} ${circumference}`}
        strokeDashoffset={0}
        strokeLinecap="round"
        transform={`rotate(135 ${cx} ${cy})`}
      />
      {/* Filled arc */}
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill="none"
        className={GRADE_STROKE[grade] ?? "stroke-text-muted"}
        strokeWidth={6}
        strokeDasharray={`${filledLength} ${circumference}`}
        strokeDashoffset={0}
        strokeLinecap="round"
        transform={`rotate(135 ${cx} ${cy})`}
      />
      {/* Grade letter */}
      <text
        x={cx}
        y={cy + 2}
        textAnchor="middle"
        dominantBaseline="middle"
        className={cn("text-2xl font-bold fill-current", GRADE_COLORS[grade] ?? "text-text-muted")}
        fontSize="24"
      >
        {grade === "unavailable" || grade === "timed_out" ? "?" : grade}
      </text>
      {/* Score number */}
      <text
        x={cx}
        y={cy + 18}
        textAnchor="middle"
        className="fill-current text-text-muted"
        fontSize="10"
      >
        {value > 0 ? `${Math.round(value)}` : ""}
      </text>
    </svg>
  );
}

function ComponentBar({
  name,
  score,
  weight,
  detail,
  available,
}: {
  name: string;
  score: number;
  weight: number;
  detail: string;
  available: boolean;
}) {
  const label = name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-xs">
        <span className={cn("font-medium", available ? "text-text-emphasis" : "text-text-muted")}>
          {label}
        </span>
        <span className="text-text-muted">
          {available ? `${Math.round(score)}` : ""} · {(weight * 100).toFixed(0)}%
        </span>
      </div>
      {available ? (
        <div className="h-1.5 bg-bg-tertiary rounded-full overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-all"
            style={{ width: `${Math.min(100, score)}%` }}
          />
        </div>
      ) : (
        <p className="text-[10px] text-text-muted italic">{detail}</p>
      )}
    </div>
  );
}

export function ProcessScoreCard({ score }: Props) {
  const isUnavailable =
    score.grade === "unavailable" || score.grade === "timed_out";

  // Find weakest available component for focus area
  const weakest = score.components
    .filter((c) => c.available)
    .sort((a, b) => a.score - b.score)[0];

  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-4">
      <h3 className="text-sm font-medium text-text-emphasis mb-3">
        Process Quality
      </h3>
      {isUnavailable ? (
        <p className="text-xs text-text-muted">
          {score.grade === "timed_out"
            ? "Score computation timed out"
            : "Not enough data to compute process score"}
        </p>
      ) : (
        <div className="flex gap-4">
          {/* Gauge */}
          <div className="flex-shrink-0">
            <GaugeArc value={score.total ?? 0} grade={score.grade} />
          </div>

          {/* Component bars */}
          <div className="flex-1 space-y-2 min-w-0">
            {score.components.map((c) => (
              <ComponentBar key={c.name} {...c} />
            ))}
          </div>
        </div>
      )}

      {/* Focus area */}
      {weakest && !isUnavailable && (
        <p className="text-xs text-text-muted mt-3 pt-2 border-t border-border-subtle">
          Focus area: {weakest.detail}
        </p>
      )}
    </div>
  );
}
