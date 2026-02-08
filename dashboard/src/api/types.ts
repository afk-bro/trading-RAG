/* ------------------------------------------------------------------ */
/*  Equity                                                             */
/* ------------------------------------------------------------------ */

export interface EquityDataPoint {
  snapshot_ts: string;
  computed_at: string;
  equity: number;
  cash: number;
  positions_value: number;
  realized_pnl: number;
  peak_equity: number;
  drawdown_pct: number;
}

export interface EquitySummary {
  current_equity: number | null;
  current_drawdown_pct: number | null;
  max_drawdown_pct: number;
  total_return_pct: number;
  latest_ts: string | null;
}

export interface EquityResponse {
  workspace_id: string;
  window_days: number;
  snapshot_count: number;
  data: EquityDataPoint[];
  summary: EquitySummary;
}

/* ------------------------------------------------------------------ */
/*  Intel Timeline                                                     */
/* ------------------------------------------------------------------ */

export interface IntelSnapshot {
  snapshot_id: string;
  as_of_ts: string;
  computed_at: string;
  regime: string;
  confidence_score: number;
  confidence_components: Record<string, number> | null;
}

export interface IntelVersion {
  version_id: string;
  version_number: number;
  version_tag: string | null;
  strategy_name: string;
  snapshot_count: number;
  latest_confidence: number | null;
  latest_regime: string | null;
  snapshots: IntelSnapshot[];
}

export interface IntelTimelineResponse {
  workspace_id: string;
  version_filter: string;
  window_days: number;
  version_count: number;
  total_snapshots: number;
  versions: IntelVersion[];
}

/* ------------------------------------------------------------------ */
/*  Alerts                                                             */
/* ------------------------------------------------------------------ */

export interface AlertItem {
  id: string;
  rule_type: string;
  severity: "critical" | "high" | "medium" | "low";
  status: "active" | "acknowledged" | "resolved";
  dedupe_key: string;
  payload: Record<string, unknown> | null;
  occurrence_count: number;
  first_triggered_at: string | null;
  last_triggered_at: string | null;
  acknowledged_at: string | null;
  acknowledged_by: string | null;
  resolved_at: string | null;
  resolved_by: string | null;
  resolution_note: string | null;
}

export interface AlertsSummary {
  by_status: Record<string, number>;
  by_severity: Record<string, number>;
  by_rule_type: Record<string, number>;
}

export interface AlertsResponse {
  workspace_id: string;
  include_resolved: boolean;
  window_days: number;
  total_alerts: number;
  summary: AlertsSummary;
  alerts: AlertItem[];
}

/* ------------------------------------------------------------------ */
/*  Dashboard Summary                                                  */
/* ------------------------------------------------------------------ */

export interface DashboardSummaryEquity {
  equity: number;
  cash: number;
  positions_value: number;
  drawdown_pct: number;
  peak_equity: number;
  as_of: string;
}

export interface DashboardSummaryIntelVersion {
  version_id: string;
  version_number: number;
  strategy_name: string;
  regime: string;
  confidence_score: number;
  as_of: string;
}

export interface DashboardSummary {
  workspace_id: string;
  generated_at: string;
  equity: DashboardSummaryEquity | null;
  intel: {
    active_versions: number;
    versions: DashboardSummaryIntelVersion[];
  };
  alerts: {
    total_active: number;
    by_severity: Record<string, number>;
  };
}

/* ------------------------------------------------------------------ */
/*  Trade Events                                                       */
/* ------------------------------------------------------------------ */

export interface TradeEventItem {
  id: string;
  correlation_id: string;
  event_type: string;
  event_time: string;
  symbol: string | null;
  side: string | null;
  entry_price: number | null;
  exit_price: number | null;
  pnl: number | null;
  duration_s: number | null;
  strategy_entity_id: string | null;
  payload: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface TradeEventsResponse {
  items: TradeEventItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface TradeEventDetail {
  event: TradeEventItem;
  related_events: TradeEventItem[];
}

/* ------------------------------------------------------------------ */
/*  RAG Query                                                          */
/* ------------------------------------------------------------------ */

export interface RagChunkResult {
  chunk_id: string;
  score: number;
  content: string;
  metadata: Record<string, unknown>;
  source_title?: string;
  source_url?: string;
}

export interface RagQueryResponse {
  workspace_id: string;
  question: string;
  mode: string;
  results: RagChunkResult[];
  answer?: string;
}

/* ------------------------------------------------------------------ */
/*  SSE Ticket                                                         */
/* ------------------------------------------------------------------ */

export interface SSETicketResponse {
  ticket: string;
  expires_in_seconds: number;
}
