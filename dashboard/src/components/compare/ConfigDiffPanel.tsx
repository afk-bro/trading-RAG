import { useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";

interface Props {
  paramsA: Record<string, unknown>;
  paramsB: Record<string, unknown>;
  labelA: string;
  labelB: string;
}

function stringify(v: unknown): string {
  if (v === undefined) return "—";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

export function ConfigDiffPanel({ paramsA, paramsB, labelA, labelB }: Props) {
  const [open, setOpen] = useState(false);

  const allKeys = Array.from(
    new Set([...Object.keys(paramsA), ...Object.keys(paramsB)]),
  ).sort();

  const changed: string[] = [];
  const same: string[] = [];

  for (const key of allKeys) {
    if (JSON.stringify(paramsA[key]) !== JSON.stringify(paramsB[key])) {
      changed.push(key);
    } else {
      same.push(key);
    }
  }

  return (
    <div className="bg-bg-secondary border border-border rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-bg-tertiary/50 transition-colors"
      >
        <h3 className="text-sm font-medium text-text-emphasis">
          Config Diff ({changed.length} difference{changed.length !== 1 ? "s" : ""})
        </h3>
        <ChevronDown
          className={cn(
            "w-4 h-4 text-text-muted transition-transform",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <div className="border-t border-border overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th scope="col" className="text-left px-3 py-2 text-xs font-medium text-text-muted">
                  Param
                </th>
                <th scope="col" className="text-right px-3 py-2 text-xs font-medium text-text-muted">
                  {labelA}
                </th>
                <th scope="col" className="text-right px-3 py-2 text-xs font-medium text-text-muted">
                  {labelB}
                </th>
              </tr>
            </thead>
            <tbody>
              {changed.map((key) => (
                <tr
                  key={key}
                  className="border-b border-border-subtle border-l-2 border-l-accent"
                >
                  <td className="px-3 py-2 font-medium text-text-emphasis font-mono text-xs">
                    {key}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {stringify(paramsA[key])}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {stringify(paramsB[key])}
                  </td>
                </tr>
              ))}
              {same.map((key) => (
                <tr
                  key={key}
                  className="border-b border-border-subtle"
                >
                  <td className="px-3 py-2 font-mono text-xs text-text-muted">
                    {key}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-text-muted">
                    {stringify(paramsA[key])}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-text-muted">
                    {stringify(paramsB[key])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
