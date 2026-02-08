import { apiGet, apiPost } from "./client";
import type {
  EquityResponse,
  IntelTimelineResponse,
  AlertsResponse,
  DashboardSummary,
  TradeEventsResponse,
  TradeEventDetail,
  RagQueryResponse,
  SSETicketResponse,
} from "./types";

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
