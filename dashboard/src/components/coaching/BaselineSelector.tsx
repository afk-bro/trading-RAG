import type { LineageCandidate } from "@/api/types";

interface Props {
  candidates: LineageCandidate[];
  currentBaselineId: string | null;
  autoBaselineId: string | null;
  onSelect: (runId: string | null) => void;
}

function fmtDate(iso: string): string {
  if (!iso) return "";
  return iso.slice(0, 16).replace("T", " ");
}

function fmtSummary(c: LineageCandidate): string {
  const parts: string[] = [];
  if (c.return_pct != null) parts.push(`${(c.return_pct * 100).toFixed(1)}%`);
  if (c.sharpe != null) parts.push(`S:${c.sharpe.toFixed(2)}`);
  return parts.length ? ` (${parts.join(", ")})` : "";
}

export function BaselineSelector({
  candidates,
  currentBaselineId,
  autoBaselineId,
  onSelect,
}: Props) {
  if (candidates.length === 0) return null;

  return (
    <div className="flex items-center gap-2 text-xs text-text-muted">
      <span>vs.</span>
      <select
        value={currentBaselineId ?? "auto"}
        onChange={(e) => {
          const val = e.target.value;
          onSelect(val === "auto" ? null : val);
        }}
        className="bg-bg-secondary border border-border rounded px-2 py-1 text-xs
                   text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
      >
        <option value="auto">
          previous run (auto)
        </option>
        {candidates.map((c) => (
          <option key={c.run_id} value={c.run_id}>
            {fmtDate(c.completed_at)}
            {fmtSummary(c)}
            {c.run_id === autoBaselineId ? " (auto)" : ""}
          </option>
        ))}
      </select>
    </div>
  );
}
