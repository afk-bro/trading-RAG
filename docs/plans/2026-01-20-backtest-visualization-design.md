# Backtest Visualization MVP Design

**Date**: 2026-01-20
**Status**: Draft
**Authors**: Human + Claude

## Goal and Scope

Implement minimal backtest result visualization to enable visual inspection of strategy performance. Aligned with PRD Section 7 (Visualization & Reporting).

### In Scope (MVP)
- Single backtest run detail page with:
  - Equity curve chart (strategy equity over time)
  - Drawdown overlay (percentage drawdown)
  - Trade list table (entry/exit, side, PnL)
  - Summary metrics card (return, sharpe, max DD, win rate)
- JSON data endpoint for chart consumption
- Self-contained HTML page (no build step, embedded Plotly)

### Deferred
- Candlestick chart with entry/exit markers (Phase +1)
- Benchmark comparison line (buy & hold)
- Tune heatmap / parameter scatter plots
- WFO fold visualization
- PDF/image export
- Dark mode

---

## Data Sources

All data already exists in `backtest_runs` table:

| Column | Type | Chart Use |
|--------|------|-----------|
| `equity_curve` | JSONB | `[{"t": "ISO", "equity": float}, ...]` |
| `trades` | JSONB | `[{"t_entry", "t_exit", "side", "pnl", "return_pct"}, ...]` |
| `summary` | JSONB | `{return_pct, max_drawdown_pct, sharpe, trades, win_rate}` |
| `dataset_meta` | JSONB | `{symbol, timeframe, date_min, date_max}` |
| `params` | JSONB | Strategy parameters used |
| `status` | TEXT | Only show charts for `completed` runs |

**Drawdown calculation**: Computed client-side from equity curve:
```javascript
peak = Math.max(...equity_so_far)
drawdown_pct = (peak - equity) / peak
```

---

## API Design

### GET /api/backtests/runs/{run_id}/chart-data

Returns chart-ready data for a single backtest run. **Single endpoint powers the whole page.**

**Query params**:
- `page` (int, default 1): Trades page number
- `page_size` (int, default 50): Trades per page (25/50/100)

**Response** (200):
```json
{
  "run_id": "uuid",
  "status": "completed",
  "dataset_meta": {
    "symbol": "BTC-USDT",
    "timeframe": "1h",
    "date_min": "2024-01-01T00:00:00Z",
    "date_max": "2024-06-01T00:00:00Z",
    "row_count": 3624
  },
  "params": {"lookback": 20, "threshold": 0.02},
  "summary": {
    "return_pct": 45.2,
    "max_drawdown_pct": 12.3,
    "sharpe": 1.8,
    "trades": 142,
    "win_rate": 58.5,
    "profit_factor": 1.65,
    "avg_trade_pct": 0.32
  },
  "equity": [
    {"t": "2024-01-01T00:00:00Z", "equity": 10000.0},
    {"t": "2024-01-01T01:00:00Z", "equity": 10050.0}
  ],
  "equity_source": "jsonb",
  "trades_page": [
    {
      "t_entry": "2024-01-02T14:00:00Z",
      "t_exit": "2024-01-02T18:00:00Z",
      "side": "long",
      "size": 0.1,
      "entry_price": 42000.0,
      "exit_price": 42500.0,
      "pnl": 125.50,
      "return_pct": 1.25
    }
  ],
  "trades_pagination": {
    "page": 1,
    "page_size": 50,
    "total": 142
  },
  "exports": {
    "trades_csv": "/api/backtests/runs/{run_id}/export/trades.csv",
    "json_snapshot": "/api/backtests/runs/{run_id}/export/snapshot.json"
  },
  "notes": []
}
```

**Field: `equity_source`** (future-proofing):
- `"jsonb"` - loaded from `backtest_runs.equity_curve` JSONB
- `"csv"` - loaded from artifact CSV file (future: best tune run)
- `"missing"` - no equity data available

**Field: `notes`** (edge case messages):
- `["Equity curve not stored for tune variants"]`
- `["Trades not stored for this run"]`

**Error responses**:
- 404: Run not found
- 403: Run belongs to different workspace

### GET /api/backtests/runs/{run_id}/export/trades.csv

CSV export of all trades (not paginated).

```csv
entry_time,exit_time,side,size,entry_price,exit_price,pnl,return_pct
2024-01-02T14:00:00Z,2024-01-02T18:00:00Z,long,0.1,42000.0,42500.0,125.50,1.25
```

Returns 404 if trades not stored.

### GET /api/backtests/runs/{run_id}/export/snapshot.json

Full JSON snapshot (summary + equity + all trades). Same shape as chart-data but with all trades, no pagination.

### GET /admin/backtests/runs/{run_id}

HTML page that renders the backtest visualization. Uses the JSON endpoint above.

**Query params**:
- None for MVP (future: `theme=dark`, `export=png`)

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Backtest Run: {run_id}                          [CSV] [JSON]│
├─────────────────────────────────────────────────────────────┤
│  BTC-USDT · 1h · 2024-01-01 → 2024-06-01 · 3624 bars       │
│  Params: lookback=20, threshold=0.02                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                                                      │   │
│  │              EQUITY CURVE (primary y-axis)          │   │
│  │              DRAWDOWN % (secondary y-axis, inverted)│   │
│  │                                                      │   │
│  │   [Plotly chart with hover, zoom, pan]              │   │
│  │                                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Return      │  Max DD      │  Sharpe      │  Win Rate     │
│  +45.2%      │  -12.3%      │  1.80        │  58.5%        │
├──────────────┴──────────────┴──────────────┴───────────────┤
│  Trades  │  Profit Factor  │  Avg Trade                    │
│  142     │  1.65           │  +0.32%                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRADE LIST                                    [Filter: All]│
│  ┌─────────┬─────────┬──────┬─────────┬──────────────────┐ │
│  │ Entry   │ Exit    │ Side │ PnL     │ Return           │ │
│  ├─────────┼─────────┼──────┼─────────┼──────────────────┤ │
│  │ Jan 2   │ Jan 2   │ LONG │ +$125.50│ +1.25%           │ │
│  │ Jan 3   │ Jan 4   │ SHORT│ -$42.00 │ -0.42%           │ │
│  │ ...     │ ...     │ ...  │ ...     │ ...              │ │
│  └─────────┴─────────┴──────┴─────────┴──────────────────┘ │
│                                              Page 1 of 15   │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Choice

### Chart Library: Plotly.js (CDN)

**Rationale**:
- PRD explicitly mentions "Plotly or similar"
- No build step required (CDN include)
- Interactive out of the box (hover, zoom, pan, export)
- Dual y-axis support for equity + drawdown
- Financial charting capability for future candlestick phase

**CDN**: `https://cdn.plot.ly/plotly-2.27.0.min.js` (~3.5MB, cached)

**Alternative considered**: Chart.js (smaller) - rejected because Plotly has better financial chart support for Phase +1.

### Page Structure

Single self-contained HTML template using Jinja2:

```
app/templates/admin/backtest_run_detail.html
```

Pattern matches existing admin pages (`/admin/ingest`, `/admin/coverage/cockpit`).

---

## Implementation Plan

### Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `app/routers/backtests/chart.py` | Create | JSON endpoint + HTML page |
| `app/templates/admin/backtest_run_detail.html` | Create | Visualization page |
| `app/routers/backtests/__init__.py` | Modify | Include chart router |

### Endpoint Registration

```python
# app/routers/backtests/chart.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/backtests", tags=["backtests"])

@router.get("/runs/{run_id}/chart-data")
async def get_chart_data(run_id: UUID, ...) -> BacktestChartData:
    """JSON data for chart rendering."""

# Admin HTML page (requires admin token)
admin_router = APIRouter(prefix="/admin/backtests", tags=["admin"])

@admin_router.get("/runs/{run_id}", response_class=HTMLResponse)
async def backtest_run_detail_page(request: Request, run_id: UUID, ...):
    """HTML visualization page."""
```

### Chart Configuration

```javascript
// Equity + Drawdown dual-axis chart
const equityTrace = {
  x: data.equity_curve.map(p => p.t),
  y: data.equity_curve.map(p => p.equity),
  name: 'Equity',
  type: 'scatter',
  mode: 'lines',
  line: { color: '#2ecc71', width: 2 },
  yaxis: 'y'
};

const drawdownTrace = {
  x: timestamps,
  y: drawdownPct,  // computed client-side
  name: 'Drawdown',
  type: 'scatter',
  mode: 'lines',
  fill: 'tozeroy',
  line: { color: '#e74c3c', width: 1 },
  yaxis: 'y2'
};

const layout = {
  title: `${data.dataset_meta.symbol} Backtest`,
  xaxis: { title: 'Time', type: 'date' },
  yaxis: { title: 'Equity ($)', side: 'left' },
  yaxis2: {
    title: 'Drawdown (%)',
    side: 'right',
    overlaying: 'y',
    autorange: 'reversed',  // 0% at top, -X% at bottom
    tickformat: '.1%'
  },
  hovermode: 'x unified',
  legend: { orientation: 'h', y: -0.15 }
};

Plotly.newPlot('chart', [equityTrace, drawdownTrace], layout, {responsive: true});
```

---

## Pydantic Models

```python
# app/routers/backtests/chart.py

class EquityPoint(BaseModel):
    t: datetime
    equity: float

class TradeRecord(BaseModel):
    t_entry: datetime
    t_exit: datetime
    side: str
    size: float | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    pnl: float
    return_pct: float

class BacktestSummary(BaseModel):
    return_pct: float | None = None
    max_drawdown_pct: float | None = None
    sharpe: float | None = None
    trades: int | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_trade_pct: float | None = None

class DatasetMeta(BaseModel):
    symbol: str | None = None
    timeframe: str | None = None
    date_min: datetime | None = None
    date_max: datetime | None = None
    row_count: int | None = None

class TradesPagination(BaseModel):
    page: int
    page_size: int
    total: int

class ExportLinks(BaseModel):
    trades_csv: str | None = None
    json_snapshot: str

class BacktestChartData(BaseModel):
    run_id: UUID
    status: str
    dataset_meta: DatasetMeta | None = None
    params: dict
    summary: BacktestSummary
    equity: list[EquityPoint]
    equity_source: Literal["jsonb", "csv", "missing"]
    trades_page: list[TradeRecord]
    trades_pagination: TradesPagination
    exports: ExportLinks
    notes: list[str] = []
```

---

## Edge Cases

### 1. No equity_curve / empty array
- Set `equity_source = "missing"`
- Add note: `"Equity curve not available for this run"`
- Still return summary + trades (chart section shows message)

### 2. Huge equity series (>10k points)
- MVP: Log warning, return full series (Plotly handles reasonably up to ~50k)
- Future: Server-side downsampling with `downsample=true` query param

### 3. Tune variants (equity_curve intentionally empty)
- Detect via `run_kind = 'tune_variant'`
- Add note: `"Equity curve not stored for tune variants"`
- If `best_run_id` available on parent tune, include in response for "Open best run" link

### 4. Trades not stored
- Set `trades_page = []`, `trades_pagination.total = 0`
- Add note: `"Trades not stored for this run"`
- Disable CSV export link (`exports.trades_csv = null`)

### 5. Timestamp normalization
- All timestamps returned with explicit `Z` suffix (UTC)
- Backend normalizes before response: `dt.isoformat() + "Z"` if naive
- Prevents Plotly x-axis timezone shifts

---

## Test Plan

### Unit Tests

```python
# tests/unit/routers/test_backtest_chart.py

def test_chart_data_returns_equity_and_summary():
    """Returns equity + summary for a completed run with curve."""

def test_chart_data_empty_equity_for_variant():
    """Returns empty equity and note for tune variant run."""

def test_chart_data_pagination_works():
    """Pagination returns correct slice and total."""

def test_chart_data_404_on_missing_run():
    """Return 404 for non-existent run_id."""

def test_chart_data_403_on_wrong_workspace():
    """Return 403 if run belongs to different workspace."""

def test_trades_csv_export_content_type():
    """CSV export returns text/csv with correct headers."""

def test_json_snapshot_includes_all_trades():
    """JSON snapshot includes all trades, not paginated."""
```

### Manual Verification (UI Smoke)

1. Open a standalone backtest: charts render + table loads
2. Open a tune variant: summary renders + "no curve" message
3. Export buttons download successfully
4. Pagination controls work (Prev/Next, page size change)

---

## Security

- `/api/backtests/runs/{run_id}/chart-data`: Workspace-scoped (requires valid workspace context)
- `/admin/backtests/runs/{run_id}`: Requires `X-Admin-Token` header (existing pattern)
- No new auth mechanisms needed

---

## Future Extensions (Phase +1)

After MVP, add:

1. **Candlestick chart**: Requires OHLCV data join (not in backtest_runs)
2. **Entry/exit markers**: Plot trade points on candlestick
3. **Benchmark line**: Add buy-and-hold equity trace
4. **Tune comparison**: Side-by-side chart for multiple runs
5. **WFO fold timeline**: Horizontal bar chart of fold periods

---

## Acceptance Criteria

- [ ] `/api/backtests/runs/{run_id}/chart-data` returns valid JSON with equity, summary, trades_page
- [ ] `/api/backtests/runs/{run_id}/export/trades.csv` returns CSV with all trades
- [ ] `/api/backtests/runs/{run_id}/export/snapshot.json` returns full snapshot
- [ ] `/admin/backtests/runs/{run_id}` renders interactive equity+drawdown chart
- [ ] Summary metrics display correctly (missing keys show "—")
- [ ] Trade table shows paginated trades with Prev/Next controls
- [ ] Export buttons download successfully
- [ ] Tune variants show "no curve" message + summary still works
- [ ] Empty equity curve shows appropriate message
- [ ] Chart is responsive (works on different screen sizes)
- [ ] All timestamps have explicit UTC suffix (no timezone shifts)
