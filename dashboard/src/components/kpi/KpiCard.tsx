import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: string;
  color?: "default" | "success" | "danger" | "warning" | "accent";
  isLoading?: boolean;
}

const colorMap = {
  default: "text-foreground",
  success: "text-success",
  danger: "text-danger",
  warning: "text-warning",
  accent: "text-accent",
};

export function KpiCard({ label, value, color = "default", isLoading }: Props) {
  return (
    <div className="bg-bg-secondary border border-border rounded-lg p-4">
      <p className="text-xs text-text-muted mb-1">{label}</p>
      {isLoading ? (
        <div className="h-7 w-20 bg-bg-tertiary rounded animate-pulse" />
      ) : (
        <p className={cn("text-2xl font-semibold", colorMap[color])}>
          {value}
        </p>
      )}
    </div>
  );
}
