import { useMemo, useState } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import type { BacktestChartTradeRecord } from "@/api/types";
import { cn } from "@/lib/utils";
import { fmtPct, fmtPnl, fmtPrice } from "@/lib/chart-utils";
import { format } from "date-fns";
import { ArrowUpDown } from "lucide-react";

interface Props {
  data: BacktestChartTradeRecord[];
  onTradeClick: (trade: BacktestChartTradeRecord) => void;
}

export function BacktestTradesTable({ data, onTradeClick }: Props) {
  const [sorting, setSorting] = useState<SortingState>([]);

  const columns = useMemo<ColumnDef<BacktestChartTradeRecord>[]>(
    () => [
      {
        accessorKey: "t_entry",
        header: "Entry",
        cell: ({ getValue }) => {
          const v = getValue<string>();
          return v ? format(new Date(v), "yyyy-MM-dd HH:mm") : "—";
        },
      },
      {
        accessorKey: "t_exit",
        header: "Exit",
        cell: ({ getValue }) => {
          const v = getValue<string>();
          return v ? format(new Date(v), "yyyy-MM-dd HH:mm") : "—";
        },
      },
      {
        accessorKey: "side",
        header: "Side",
        cell: ({ getValue }) => {
          const s = getValue<string>();
          return (
            <span
              className={cn(
                "font-medium capitalize",
                s === "long" ? "text-success" : "text-danger",
              )}
            >
              {s}
            </span>
          );
        },
      },
      {
        accessorKey: "entry_price",
        header: "Entry Price",
        cell: ({ getValue }) => fmtPrice(getValue<number | undefined>()),
      },
      {
        accessorKey: "exit_price",
        header: "Exit Price",
        cell: ({ getValue }) => fmtPrice(getValue<number | undefined>()),
      },
      {
        accessorKey: "pnl",
        header: "PnL",
        cell: ({ getValue }) => {
          const v = getValue<number>();
          return (
            <span className={v >= 0 ? "text-success" : "text-danger"}>
              {fmtPnl(v)}
            </span>
          );
        },
      },
      {
        accessorKey: "return_pct",
        header: "Return",
        cell: ({ getValue }) => {
          const v = getValue<number>();
          return (
            <span className={v >= 0 ? "text-success" : "text-danger"}>
              {fmtPct(v)}
            </span>
          );
        },
      },
      {
        accessorKey: "size",
        header: "Size",
        cell: ({ getValue }) => {
          const v = getValue<number | undefined>();
          return v != null ? v.toFixed(4) : "—";
        },
      },
    ],
    [],
  );

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
      <table className="w-full text-sm">
        <thead>
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b border-border">
              {hg.headers.map((header) => (
                <th
                  key={header.id}
                  scope="col"
                  className="text-left px-3 py-2 text-xs font-medium text-text-muted cursor-pointer select-none hover:text-foreground"
                  onClick={header.column.getToggleSortingHandler()}
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
              onClick={() => onTradeClick(row.original)}
              className={cn(
                "border-b border-border-subtle hover:bg-bg-tertiary/50 cursor-pointer transition-colors",
                i % 2 === 1 && "bg-bg-secondary/50",
              )}
            >
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-3 py-2.5">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
