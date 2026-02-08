import { cn } from "@/lib/utils";

const OPTIONS = [7, 14, 30, 90] as const;

interface Props {
  value: number;
  onChange: (days: number) => void;
}

export function DateRangeSelector({ value, onChange }: Props) {
  return (
    <div className="flex gap-1">
      {OPTIONS.map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          className={cn(
            "px-3 py-1 rounded-md text-xs font-medium transition-colors",
            d === value
              ? "bg-accent text-background"
              : "bg-bg-tertiary text-text-muted hover:text-foreground",
          )}
        >
          {d}d
        </button>
      ))}
    </div>
  );
}
