/* ------------------------------------------------------------------ */
/*  Workspaces                                                         */
/* ------------------------------------------------------------------ */

export interface Workspace {
  id: string;
  name: string;
  slug: string;
  is_active: boolean;
  created_at: string;
}

export interface WorkspaceListResponse {
  workspaces: Workspace[];
}

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

/* ------------------------------------------------------------------ */
/*  Backtest Types                                                     */
/* ------------------------------------------------------------------ */

export interface BacktestSummary {
  return_pct: number;
  max_drawdown_pct: number;
  sharpe?: number;
  win_rate: number;
  trades: number;
  profit_factor?: number;
  avg_trade_pct?: number;
  buy_hold_return_pct?: number;
}

export interface BacktestRunListItem {
  id: string;
  created_at: string;
  strategy_entity_id: string;
  strategy_name?: string;
  status: string;
  summary?: BacktestSummary;
  dataset_meta: Record<string, unknown>;
}

export interface BacktestRunListResponse {
  items: BacktestRunListItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface BacktestChartEquityPoint {
  t: string;
  equity: number;
}

export interface BacktestChartTradeRecord {
  t_entry: string;
  t_exit: string;
  side: string;
  size?: number;
  entry_price?: number;
  exit_price?: number;
  pnl: number;
  return_pct: number;
}

export interface BacktestChartSummary {
  return_pct?: number;
  max_drawdown_pct?: number;
  sharpe?: number;
  trades?: number;
  win_rate?: number;
  profit_factor?: number;
  avg_trade_pct?: number;
  buy_hold_return_pct?: number;
  [k: string]: unknown;
}

export interface DatasetMeta {
  symbol?: string;
  timeframe?: string;
  date_min?: string;
  date_max?: string;
  row_count?: number;
  [k: string]: unknown;
}

export interface RegimeInfo {
  trend_tag?: string;
  vol_tag?: string;
  efficiency_tag?: string;
  ts_start?: string;
  ts_end?: string;
}

export interface TradesPagination {
  page: number;
  page_size: number;
  total: number;
}

export interface ExportLinks {
  trades_csv?: string;
  json_snapshot: string;
}

export interface BacktestChartData {
  run_id: string;
  status: string;
  dataset_meta?: DatasetMeta;
  params: Record<string, unknown>;
  summary: BacktestChartSummary;
  equity: BacktestChartEquityPoint[];
  equity_source: string;
  trades_page: BacktestChartTradeRecord[];
  trades_pagination: TradesPagination;
  exports: ExportLinks;
  notes: string[];
  regime_is?: RegimeInfo;
  regime_oos?: RegimeInfo;
}

/* ------------------------------------------------------------------ */
/*  Run Detail DTO (workspace-scoped, UI-shaped)                       */
/* ------------------------------------------------------------------ */

export interface RunDetailStrategy {
  entity_id: string | null;
  version_id: string | null;
  name: string | null;
}

export interface DrawdownPoint {
  t: string;
  drawdown_pct: number;
}

export interface RunDetailRegime {
  trend_tag: string | null;
  vol_tag: string | null;
  efficiency_tag: string | null;
  tags: string[];
}

export interface RunDetailTrade {
  t_entry: string;
  t_exit: string;
  side: string;
  entry_price?: number;
  exit_price?: number;
  pnl: number;
  return_pct: number;
}

export interface RunDetailResponse {
  run_id: string;
  workspace_id: string;
  status: string;
  run_kind: string;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  strategy: RunDetailStrategy;
  dataset: DatasetMeta;
  params: Record<string, unknown>;
  summary: BacktestChartSummary;
  equity: BacktestChartEquityPoint[];
  drawdown: DrawdownPoint[];
  trades: RunDetailTrade[];
  trade_count: number;
  warnings: string[];
  regime_is: RunDetailRegime | null;
  regime_oos: RunDetailRegime | null;
}

/* ------------------------------------------------------------------ */
/*  Run Events (Replay)                                               */
/* ------------------------------------------------------------------ */

export interface RunEvent {
  type: "orb_range_update" | "orb_range_locked" | "setup_valid" | "entry_signal";
  bar_index: number;
  ts: string;
  [key: string]: unknown;
}

export interface RunEventsResponse {
  run_id: string;
  events: RunEvent[];
  event_count: number;
}
