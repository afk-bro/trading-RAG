# Admin UI Improvements - Design Document

**Date:** 2026-01-28
**Status:** Approved
**Approach:** Incremental enhancement (Jinja + progressive JS), no SPA rewrite

## Overview

The current admin UI is functional but lacks visual feedback for understanding backtest quality at a glance. Tables alone don't scale - users need to click into every run to understand if it's actually good.

### Primary Pain Points

1. **"Is this run actually good?"** - Needs equity curve + drawdown visualization
2. **"Does it only work in one regime?"** - Needs regime timeline overlay
3. **"Context loss"** - Compare selections reset on navigation, filters lost
4. **Visual inconsistency** - Inline styles duplicated across templates

### Guiding Principles

- **No SPA rewrite** - Too much working surface area to justify
- **Progressive enhancement** - JS failures degrade gracefully
- **Incremental delivery** - Each PR is independently shippable
- **Performance first** - Sparklines must not block page load

---

## PR 1: Sparklines in Tables

### Goal

Add 80×24px sparklines showing equity curve in runs/tunes list tables. Users can scan 30 runs in 5 seconds instead of 30 clicks.

### Technical Approach

**New Endpoint:**
```
GET /admin/api/backtests/runs/{run_id}/sparkline
Response: { "y": [float...] }  // 80-120 points max
```

**Downsampling Rule:**
- If N ≤ 120: return all points
- Else: evenly spaced indices (upgrade to LTTB if spikes matter later)

**Client Rendering:**
- uPlot (60KB) for fast, minimal sparklines
- In-memory cache (Map) to avoid refetch on scroll/filter
- Progressive loading: table renders immediately, sparklines fill in async

### Files to Create/Modify

| File | Action |
|------|--------|
| `app/admin/routers/chart_api.py` | New router for sparkline endpoint |
| `app/admin/templates/layout.html` | Add uPlot script include |
| `app/static/admin/js/sparklines.js` | New: fetch, render, cache logic |
| `app/admin/templates/tunes_list.html` | Add sparkline column |
| `app/admin/templates/leaderboard.html` | Add sparkline column |
| `tests/unit/admin/test_chart_api.py` | Endpoint tests |

### Acceptance Criteria

- [ ] Table loads without blocking; sparklines render within ~1s
- [ ] Each sparkline uses ≤120 points
- [ ] Endpoint handles missing equity data gracefully (empty sparkline)
- [ ] Works with pagination + filters (cache invalidation)
- [ ] No visual regression if JS disabled (column shows "-")
- [ ] uPlot bundle included only on pages that need it

---

## PR 2: Regime Timeline Overlay

### Goal

On backtest detail page, show colored regime bands under the equity curve. One glance answers "trend-only alpha?" or "chop kills it?"

### Technical Approach

**New Endpoint:**
```
GET /admin/api/backtests/runs/{run_id}/regimes
Response: [
  { "start_ts": "...", "end_ts": "...", "tag": "uptrend" },
  { "start_ts": "...", "end_ts": "...", "tag": "high_vol" }
]
```

**Rendering:**
- Reuse existing Plotly chart
- Add shapes layer for regime bands (colored rectangles)
- Thin band (20-30px) below main equity line

**Sidebar Enhancement:**
- "% time in each regime"
- "Sharpe per regime" breakdown (if data available)

### Files to Modify

| File | Action |
|------|--------|
| `app/admin/routers/chart_api.py` | Add regimes endpoint |
| `app/admin/templates/backtest_run_detail.html` | Regime overlay on chart |
| `app/admin/templates/tune_detail.html` | Same overlay |

### Acceptance Criteria

- [ ] Regime bands align with equity curve time axis
- [ ] Color coding matches existing regime badge colors
- [ ] Legend or tooltip explains regime colors
- [ ] Handles runs without regime data gracefully

---

## PR 3: Global Compare Tray

### Goal

Persist run/tune selections across navigation. Sticky bottom bar shows selected items with quick actions.

### Technical Approach

**State Management:**
```javascript
// localStorage persistence
localStorage['compare_selections'] = JSON.stringify({
  type: 'tunes',  // or 'runs'
  ids: ['uuid1', 'uuid2', ...]
})

// Custom events for cross-component sync
window.dispatchEvent(new CustomEvent('compare-changed', { detail: selections }))
```

**UI Component:**
- Sticky bar at bottom (position: fixed)
- Shows: "Selected: 3 tunes"
- Buttons: Compare | Clear | Export CSV
- Checkbox in each table row syncs with tray

### Files to Create/Modify

| File | Action |
|------|--------|
| `app/static/admin/js/compare-tray.js` | New: tray logic + localStorage |
| `app/admin/templates/layout.html` | Add tray HTML + include script |
| `app/admin/templates/tunes_list.html` | Wire up checkboxes |
| `app/admin/templates/leaderboard.html` | Wire up checkboxes |

### Acceptance Criteria

- [ ] Selections persist across page navigation
- [ ] Selections persist across browser refresh
- [ ] Tray updates live without reload
- [ ] Clear button works
- [ ] Compare button navigates to compare view with selected IDs
- [ ] Max selection limit (10?) with user feedback

---

## PR 4: Design System Extraction

### Goal

Extract shared CSS into reusable files. Reduce inline `<style>` duplication.

### Files to Create

| File | Contents |
|------|----------|
| `app/static/admin/css/tokens.css` | CSS variables: colors, spacing, typography |
| `app/static/admin/css/components.css` | Shared components: buttons, badges, tables, cards, forms |
| `app/admin/templates/layout.html` | Remove inline styles, link to CSS files |

### Migration Strategy

1. Extract tokens.css first (copy `:root` block)
2. Extract components.css (copy shared classes)
3. Update layout.html to link files
4. Remove duplicated inline styles from individual templates
5. Keep truly page-specific styles inline

### Acceptance Criteria

- [ ] No visual regression after extraction
- [ ] layout.html `<style>` block reduced by >80%
- [ ] Individual templates have minimal inline CSS
- [ ] Hot reload works for CSS files in dev

---

## PR 5: HTMX for List Filtering

### Goal

Filters and pagination update table body without full page reload.

### Scope (Surgical)

Apply HTMX only to:
- Tunes list
- Leaderboard
- Alerts list
- WFO list

### Technical Approach

```html
<!-- Filter form -->
<form hx-get="/admin/backtests/tunes"
      hx-target="#tunes-table-body"
      hx-swap="innerHTML"
      hx-trigger="change">
  ...filters...
</form>

<!-- Table body -->
<tbody id="tunes-table-body" hx-swap-oob="true">
  ...rows...
</tbody>
```

**Server Changes:**
- Detect `HX-Request` header
- Return partial HTML (table body only) for HTMX requests
- Return full page for regular requests

### Files to Modify

| File | Action |
|------|--------|
| `app/admin/templates/layout.html` | Add HTMX script |
| `app/admin/routers/*.py` | Add partial response logic |
| `app/admin/templates/*_list.html` | Add hx-* attributes |

### Acceptance Criteria

- [ ] Filter changes don't cause full page reload
- [ ] Pagination doesn't cause full page reload
- [ ] Browser back button still works (URL updates)
- [ ] Non-JS fallback still works (full page reload)
- [ ] Sparklines re-render after HTMX swap

---

## Implementation Order

```
PR 1 (Sparklines)     ████████░░░░  Foundation for viz
       ↓
PR 2 (Regime overlay) ████████████  Builds on chart infra

PR 3 (Compare tray)   ████████░░░░  Independent track

PR 4 (Design system)  ████░░░░░░░░  Do when touching templates anyway
       ↓
PR 5 (HTMX)           ████████████  Last, needs stable templates
```

**Estimated Effort:**
- PR 1: Medium (new endpoint + JS)
- PR 2: Small (extends PR 1 infra)
- PR 3: Medium (localStorage + UI)
- PR 4: Small (extraction, no new logic)
- PR 5: Medium (server changes + testing)

---

## Open Questions

1. **Sparkline data source**: Use `equity_curve` from backtest result, or compute from trades?
2. **uPlot vs alternatives**: Stick with uPlot, or use simpler canvas-based sparkline?
3. **Compare tray limit**: Max 10 selections? Show warning at limit?

---

## Non-Goals (Explicitly Out of Scope)

- Full SPA rewrite
- Mobile-first redesign (desktop is primary use case)
- Real-time WebSocket updates
- Complex saved view management (querystring is sufficient)
