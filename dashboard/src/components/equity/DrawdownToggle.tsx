import { cn } from "@/lib/utils";
import { TrendingDown } from "lucide-react";

interface Props {
  active: boolean;
  onToggle: () => void;
}

export function DrawdownToggle({ active, onToggle }: Props) {
  return (
    <button
      onClick={onToggle}
      aria-label={active ? "Hide drawdown overlay" : "Show drawdown overlay"}
      aria-pressed={active}
      className={cn(
        "flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-colors",
        active
          ? "bg-danger/20 text-danger"
          : "bg-bg-tertiary text-text-muted hover:text-foreground",
      )}
    >
      <TrendingDown className="w-3 h-3" />
      Drawdown
    </button>
  );
}
