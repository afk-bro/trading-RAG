import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import type { BacktestRunListItem } from "@/api/types";
import { cn } from "@/lib/utils";
import { fmtPct, fmtNum } from "@/lib/chart-utils";
import { format } from "date-fns";
import { ArrowUpDown } from "lucide-react";

interface Props {
  data: BacktestRunListItem[];
  selectedIds?: Set<string>;
  onSelectionChange?: (ids: Set<string>) => void;
}

const STATUS_COLORS: Record<string, string> = {
  completed: "bg-success/15 text-success",
  failed: "bg-danger/15 text-danger",
  running: "bg-accent/15 text-accent",
};

export function BacktestsTable({ data, selectedIds, onSelectionChange }: Props) {
  const navigate = useNavigate();
  const [sorting, setSorting] = useState<SortingState>([]);

  const hasSelection = selectedIds !== undefined && onSelectionChange !== undefined;

  const columns = useMemo<ColumnDef<BacktestRunListItem>[]>(
    () => [
      ...(hasSelection
        ? [
            {
              id: "select",
              enableSorting: false,
              header: () => null,
              cell: ({ row }: { row: { original: BacktestRunListItem } }) => {
                const id = row.original.id;
                const checked = selectedIds?.has(id) ?? false;
                const disabled = !checked && (selectedIds?.size ?? 0) >= 2;
                return (
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={disabled}
                    className="accent-accent cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                    onClick={(e: React.MouseEvent) => e.stopPropagation()}
                    onChange={() => {
                      const next = new Set(selectedIds);
                      if (next.has(id)) {
                        next.delete(id);
                      } else {
                        next.add(id);
                      }
                      onSelectionChange?.(next);
                    }}
                  />
                );
              },
            } satisfies ColumnDef<BacktestRunListItem>,
          ]
        : []),
      {
        accessorKey: "created_at",
        header: "Date",
        cell: ({ getValue }) =>
          format(new Date(getValue<string>()), "yyyy-MM-dd HH:mm"),
        sortingFn: "datetime",
      },
      {
        id: "strategy_name",
        header: "Strategy",
        accessorFn: (row) =>
          row.strategy_name || row.strategy_entity_id.slice(0, 12),
        cell: ({ getValue }) => (
          <span className="font-medium">{getValue<string>()}</span>
        ),
      },
      {
        id: "symbol",
        header: "Symbol",
        accessorFn: (row) => (row.dataset_meta?.symbol as string) ?? "—",
      },
      {
        id: "timeframe",
        header: "TF",
        accessorFn: (row) => (row.dataset_meta?.timeframe as string) ?? "—",
      },
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ getValue }) => {
          const s = getValue<string>();
          return (
            <span
              className={cn(
                "px-2 py-0.5 rounded-full text-xs font-medium",
                STATUS_COLORS[s] ?? "bg-bg-tertiary text-text-muted",
              )}
            >
              {s}
            </span>
          );
        },
      },
      {
        id: "trades",
        header: "Trades",
        accessorFn: (row) => row.summary?.trades,
        cell: ({ getValue }) => fmtNum(getValue<number | undefined>(), 0),
      },
      {
        id: "win_rate",
        header: "Win Rate",
        accessorFn: (row) => row.summary?.win_rate,
        cell: ({ getValue }) => fmtPct(getValue<number | undefined>()),
      },
      {
        id: "profit_factor",
        header: "PF",
        accessorFn: (row) => row.summary?.profit_factor,
        cell: ({ getValue }) => fmtNum(getValue<number | undefined>()),
      },
      {
        id: "max_drawdown",
        header: "Max DD",
        accessorFn: (row) => row.summary?.max_drawdown_pct,
        cell: ({ getValue }) => {
          const v = getValue<number | undefined>();
          return (
            <span className={v != null ? "text-danger" : ""}>
              {fmtPct(v)}
            </span>
          );
        },
      },
      {
        id: "return_pct",
        header: "Return",
        accessorFn: (row) => row.summary?.return_pct,
        cell: ({ getValue }) => {
          const v = getValue<number | undefined>();
          return (
            <span
              className={
                v != null ? (v >= 0 ? "text-success" : "text-danger") : ""
              }
            >
              {fmtPct(v)}
            </span>
          );
        },
      },
    ],
    [hasSelection, selectedIds, onSelectionChange],
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
                  className={cn(
                    "text-left px-3 py-2 text-xs font-medium text-text-muted select-none",
                    header.column.getCanSort() && "cursor-pointer hover:text-foreground",
                    header.id === "select" && "w-8",
                  )}
                  onClick={header.column.getToggleSortingHandler()}
                >
                  <div className="flex items-center gap-1">
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext(),
                    )}
                    {header.column.getCanSort() && (
                      <ArrowUpDown className="w-3 h-3 opacity-40" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr
              key={row.id}
              onClick={() => navigate(`/backtests/${row.original.id}`)}
              className="border-b border-border-subtle hover:bg-bg-tertiary/50 cursor-pointer transition-colors"
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
