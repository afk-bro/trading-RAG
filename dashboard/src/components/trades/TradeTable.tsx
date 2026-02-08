import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { formatCurrency, formatDuration } from "@/lib/chart-utils";
import type { TradeEventItem } from "@/api/types";
import { ArrowUpDown } from "lucide-react";

const col = createColumnHelper<TradeEventItem>();

const columns = [
  col.accessor("event_time", {
    header: "Time",
    cell: (info) => (
      <span className="text-xs text-text-muted whitespace-nowrap">
        {format(new Date(info.getValue()), "MMM d, HH:mm")}
      </span>
    ),
  }),
  col.accessor("event_type", {
    header: "Type",
    cell: (info) => (
      <span className="text-xs font-mono">{info.getValue()}</span>
    ),
  }),
  col.accessor("symbol", {
    header: "Symbol",
    cell: (info) => (
      <span className="text-xs font-medium text-text-emphasis">
        {info.getValue() ?? "—"}
      </span>
    ),
  }),
  col.accessor("side", {
    header: "Side",
    cell: (info) => {
      const v = info.getValue();
      return (
        <span
          className={cn(
            "text-xs font-medium capitalize",
            v === "long" ? "text-success" : v === "short" ? "text-danger" : "text-text-muted",
          )}
        >
          {v ?? "—"}
        </span>
      );
    },
  }),
  col.accessor("entry_price", {
    header: "Entry",
    cell: (info) => (
      <span className="text-xs">{info.getValue() != null ? formatCurrency(info.getValue()!) : "—"}</span>
    ),
  }),
  col.accessor("exit_price", {
    header: "Exit",
    cell: (info) => (
      <span className="text-xs">{info.getValue() != null ? formatCurrency(info.getValue()!) : "—"}</span>
    ),
  }),
  col.accessor("pnl", {
    header: "PnL",
    cell: (info) => {
      const v = info.getValue();
      return (
        <span
          className={cn(
            "text-xs font-medium",
            v != null ? (v >= 0 ? "text-success" : "text-danger") : "text-text-muted",
          )}
        >
          {v != null ? `${v >= 0 ? "+" : ""}${formatCurrency(v)}` : "—"}
        </span>
      );
    },
  }),
  col.accessor("duration_s", {
    header: "Duration",
    cell: (info) => (
      <span className="text-xs text-text-muted">
        {formatDuration(info.getValue())}
      </span>
    ),
  }),
];

interface Props {
  data: TradeEventItem[];
  onRowClick: (event: TradeEventItem) => void;
}

export function TradeTable({ data, onRowClick }: Props) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b border-border">
              {hg.headers.map((header) => (
                <th
                  key={header.id}
                  onClick={header.column.getToggleSortingHandler()}
                  className="text-left px-3 py-2 text-xs font-medium text-text-muted cursor-pointer
                             hover:text-foreground select-none whitespace-nowrap"
                >
                  <div className="flex items-center gap-1">
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext(),
                    )}
                    <ArrowUpDown className="w-3 h-3 opacity-40" />
                  </div>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row, i) => (
            <tr
              key={row.id}
              onClick={() => onRowClick(row.original)}
              className={cn(
                "border-b border-border-subtle cursor-pointer transition-colors hover:bg-bg-tertiary",
                i % 2 === 1 && "bg-bg-secondary",
              )}
            >
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-3 py-2">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {data.length === 0 && (
        <p className="text-center py-8 text-text-muted text-sm">
          No trade events found
        </p>
      )}
    </div>
  );
}
