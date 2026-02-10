import { useOutletContext } from "react-router-dom";
import type { DashboardContext } from "@/components/layout/DashboardShell";
import { useUrlState } from "@/hooks/use-url-state";
import { useSummary } from "@/hooks/use-summary";
import { useEquity } from "@/hooks/use-equity";
import { Skeleton } from "@/components/Skeleton";
import { ErrorAlert } from "@/components/ErrorAlert";
import { useIntelTimeline } from "@/hooks/use-intel-timeline";
import { useAlerts } from "@/hooks/use-alerts";
import { useEventStream } from "@/hooks/use-event-stream";
import { KpiCardsRow } from "@/components/kpi/KpiCardsRow";
import { EquityChart } from "@/components/equity/EquityChart";
import { TradeExplorer } from "@/components/trades/TradeExplorer";
import { WorkspacePicker } from "@/components/layout/WorkspacePicker";

export function DashboardPage() {
  const { workspaceId: ctxWorkspaceId, days } = useOutletContext<DashboardContext>();
  const [urlWsId, setWorkspaceId] = useUrlState("workspace_id", "");
  const workspaceId = ctxWorkspaceId || urlWsId;

  // Data hooks
  const { data: summary, isLoading: summaryLoading } = useSummary(
    workspaceId || null,
  );
  const {
    data: equity,
    isLoading: equityLoading,
    isError: equityError,
    refetch: refetchEquity,
  } = useEquity(workspaceId || null, days);
  const { data: intel } = useIntelTimeline(workspaceId || null, days);
  const { data: alerts } = useAlerts(workspaceId || null, days, true);

  // SSE connection
  useEventStream(workspaceId || null);

  // Flatten regime snapshots from all versions
  const regimeSnapshots =
    intel?.versions.flatMap((v) => v.snapshots) ?? [];

  if (!workspaceId) {
    return <WorkspacePicker onSelect={setWorkspaceId} />;
  }

  return (
    <div className="space-y-4">
      <KpiCardsRow summary={summary} isLoading={summaryLoading} />

      {equityLoading ? (
        <div className="bg-bg-secondary border border-border rounded-lg p-4 space-y-3">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-[360px]" />
        </div>
      ) : equityError ? (
        <ErrorAlert
          message="Failed to load equity data"
          onRetry={() => refetchEquity()}
        />
      ) : equity?.data?.length === 0 ? (
        <div className="bg-bg-secondary border border-border rounded-lg p-8 text-center text-text-muted text-sm">
          No equity data available yet
        </div>
      ) : (
        <EquityChart
          data={equity?.data ?? []}
          alerts={alerts?.alerts}
          regimeSnapshots={regimeSnapshots}
        />
      )}

      <TradeExplorer workspaceId={workspaceId} days={days} />
    </div>
  );
}
