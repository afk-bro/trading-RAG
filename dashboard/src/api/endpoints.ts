import { apiGet, apiPost } from "./client";
import type {
  Workspace,
  WorkspaceListResponse,
  EquityResponse,
  IntelTimelineResponse,
  AlertsResponse,
  DashboardSummary,
  TradeEventsResponse,
  TradeEventDetail,
  RagQueryResponse,
  SSETicketResponse,
  BacktestRunListResponse,
  BacktestChartData,
  RunDetailResponse,
  RunEventsResponse,
  LineageCandidatesResponse,
} from "./types";

export function getWorkspaces() {
  return apiGet<WorkspaceListResponse>("/workspaces");
}

export function createWorkspace(body: {
  name: string;
  slug?: string;
  description?: string;
}) {
  return apiPost<Workspace>("/workspaces", body);
}

export function getEquity(ws: string, days: number) {
  return apiGet<EquityResponse>(`/dashboards/${ws}/equity`, { days });
}

export function getIntelTimeline(ws: string, days: number) {
  return apiGet<IntelTimelineResponse>(`/dashboards/${ws}/intel-timeline`, {
    days,
  });
}

export function getAlerts(
  ws: string,
  days: number,
  includeResolved: boolean = false,
) {
  return apiGet<AlertsResponse>(`/dashboards/${ws}/alerts`, {
    days,
    include_resolved: includeResolved,
  });
}

export function getSummary(ws: string) {
  return apiGet<DashboardSummary>(`/dashboards/${ws}/summary`);
}

export function getTradeEvents(
  ws: string,
  params: {
    days?: number;
    limit?: number;
    offset?: number;
    event_type?: string;
    symbol?: string;
    correlation_id?: string;
  } = {},
) {
  return apiGet<TradeEventsResponse>(
    `/dashboards/${ws}/trade-events`,
    params as Record<string, string | number | boolean>,
  );
}

export function getTradeEventDetail(ws: string, eventId: string) {
  return apiGet<TradeEventDetail>(
    `/dashboards/${ws}/trade-events/${eventId}`,
  );
}

export function postQuery(
  ws: string,
  question: string,
  filters?: { symbols?: string[] },
) {
  return apiPost<RagQueryResponse>("/query", {
    workspace_id: ws,
    question,
    mode: "retrieve",
    top_k: 5,
    filters,
  });
}

export function getSSETicket() {
  return apiPost<SSETicketResponse>("/admin/events/ticket", undefined);
}

export function getBacktestRuns(
  ws: string,
  params: { status?: string; limit?: number; offset?: number } = {},
) {
  return apiGet<BacktestRunListResponse>("/backtests/", {
    workspace_id: ws,
    ...params,
  } as Record<string, string | number | boolean>);
}

export function getBacktestChartData(
  runId: string,
  page?: number,
  pageSize?: number,
) {
  const params: Record<string, string | number | boolean> = {};
  if (page != null) params.page = page;
  if (pageSize != null) params.page_size = pageSize;
  return apiGet<BacktestChartData>(`/backtests/runs/${runId}/chart-data`, params);
}

export function getRunDetail(
  ws: string,
  runId: string,
  includeCoaching?: boolean,
  baselineRunId?: string | null,
) {
  const params: Record<string, string | number | boolean> = {};
  if (includeCoaching) params.include_coaching = true;
  if (baselineRunId) params.baseline_run_id = baselineRunId;
  return apiGet<RunDetailResponse>(`/dashboards/${ws}/backtests/${runId}`, params);
}

export function getRunLineage(ws: string, runId: string) {
  return apiGet<LineageCandidatesResponse>(
    `/dashboards/${ws}/backtests/${runId}/lineage`,
  );
}

export function getRunEvents(runId: string) {
  return apiGet<RunEventsResponse>(`/backtests/runs/${runId}/events`);
}
