import { cn } from "@/lib/utils";

export function Field({
  label,
  value,
  className,
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div>
      <p className="text-[10px] text-text-muted">{label}</p>
      <p className={cn("text-sm font-medium text-foreground", className)}>
        {value}
      </p>
    </div>
  );
}
