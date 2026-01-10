# Trial Ingestion: Bridging Test Results to KB

**Date:** 2026-01-10
**Status:** Design Complete
**Author:** Claude + User collaboration

## Overview

This design bridges the gap between test runner results and the KB recommendation pipeline. It introduces a **promotion-based ingestion model** where:

- **Tune runs** are recommendation-grade by default (curated, controlled search spaces)
- **Test variants** require explicit promotion or auto-candidacy with quality gates

This prevents the KB from becoming a "dumping ground" of noisy experimental data while still enabling high-volume research workflows.

## Design Principles

1. **Merge via promotion** - Test variants are excluded by default, promoted explicitly or via quality gates
2. **Human curation signal matters** - Promoted trials get tie-break preference
3. **Circuit breakers prevent runaway candidacy** - Rate limits and volume caps
4. **Deterministic, explainable behavior** - All decisions have reasons, all rankings are reproducible

---

## 1. KB Status Model

### Status Enum

```python
class KBStatus(str, Enum):
    EXCLUDED = "excluded"    # Default for test variants - never ingest
    CANDIDATE = "candidate"  # Auto-eligible, pending batch ingestion
    PROMOTED = "promoted"    # Explicitly approved - highest priority
    REJECTED = "rejected"    # Explicitly blocked - never recommend
```

### Defaults by Source

| Table | Default Status | Rationale |
|-------|----------------|-----------|
| `backtest_tune_runs` | `promoted` | Curated by design |
| `backtest_runs` (test variants) | `excluded` | Noisy by default |

### Schema Changes

```sql
ALTER TABLE backtest_runs
    ADD COLUMN kb_status TEXT DEFAULT 'excluded',
    ADD COLUMN kb_status_changed_at TIMESTAMPTZ,
    ADD COLUMN kb_status_changed_by TEXT,
    ADD COLUMN kb_status_reason TEXT,
    ADD COLUMN kb_promoted_at TIMESTAMPTZ,
    ADD COLUMN kb_promoted_by TEXT,
    ADD COLUMN auto_candidate_gate TEXT,      -- Gate decision
    ADD COLUMN auto_candidate_breaker TEXT;   -- Breaker decision

ALTER TABLE backtest_tune_runs
    ADD COLUMN kb_status TEXT DEFAULT 'promoted',
    ADD COLUMN kb_status_changed_at TIMESTAMPTZ,
    ADD COLUMN kb_status_changed_by TEXT,
    ADD COLUMN kb_status_reason TEXT,
    ADD COLUMN kb_promoted_at TIMESTAMPTZ DEFAULT NOW(),
    ADD COLUMN kb_promoted_by TEXT;
```

---

## 2. Experiment Types

### Enum Definition

```sql
CREATE TYPE experiment_type AS ENUM ('tune', 'sweep', 'ablation', 'manual');
```

### Eligibility Rules

| Type | Auto-Candidate | Can Promote | Notes |
|------|----------------|-------------|-------|
| `tune` | N/A (default promoted) | Yes | Curated source |
| `sweep` | Yes (if gates pass) | Yes | Standard test variants |
| `ablation` | Yes (if gates pass) | Yes | Controlled experiments |
| `manual` | **No** | Yes (admin only) | Explicit promotion required |

Unknown experiment types are excluded with reason `unknown_experiment_type`.

---

## 3. Unified Ingestion View

```sql
CREATE VIEW kb_eligible_trials AS
  SELECT
    'tune_run' AS source_type,
    'tune' AS experiment_type,
    id AS source_id,
    tune_id AS group_id,
    workspace_id,
    strategy_name,
    params,
    'success' AS trial_status,
    regime_is,
    regime_oos,
    'regime_v1' AS regime_schema_version,
    sharpe_oos, return_frac_oos, max_dd_frac_oos, n_trades_oos,
    kb_status,
    kb_promoted_at,
    created_at
  FROM backtest_tune_runs
  WHERE kb_status IN ('candidate', 'promoted')
    AND status = 'completed'

  UNION ALL

  SELECT
    'test_variant' AS source_type,
    COALESCE(summary->>'experiment_type', 'sweep') AS experiment_type,
    id AS source_id,
    run_plan_id AS group_id,
    workspace_id,
    summary->>'strategy_name' AS strategy_name,
    summary->'params' AS params,
    status AS trial_status,
    regime_is,
    regime_oos,
    NULLIF(regime_schema_version, '') AS regime_schema_version,
    (summary->>'sharpe_oos')::float,
    (summary->>'return_frac_oos')::float,
    (summary->>'max_dd_frac_oos')::float,
    (summary->>'n_trades_oos')::int,
    kb_status,
    kb_promoted_at,
    created_at
  FROM backtest_runs
  WHERE run_kind = 'test_variant'
    AND kb_status IN ('candidate', 'promoted')
    AND status = 'success'
    -- Policy: candidates need regime, promoted can skip
    AND (
      kb_status = 'promoted'
      OR (kb_status = 'candidate' AND regime_oos IS NOT NULL)
    )
    -- Manual runs only if promoted
    AND (
      experiment_type != 'manual'
      OR kb_status = 'promoted'
    );
```

---

## 4. Candidacy Policy

### Gate Function

Location: `app/services/kb/candidacy.py`

```python
KNOWN_EXPERIMENT_TYPES = {"tune", "sweep", "ablation", "manual"}

@dataclass
class CandidacyDecision:
    eligible: bool
    reason: str

@dataclass
class CandidacyConfig:
    require_regime: bool = True
    min_trades: int = 5
    min_oos_bars: Optional[int] = None  # Future: OOS coverage sanity
    max_drawdown: float = 0.25
    max_overfit_gap: float = 0.30
    min_sharpe: float = 0.1

def is_candidate(
    metrics: VariantMetrics,
    regime_oos: Optional[RegimeSnapshot],
    experiment_type: str,
    config: CandidacyConfig,
) -> CandidacyDecision:
    """Pure function - no DB access, no side effects."""

    # Unknown types excluded
    if experiment_type not in KNOWN_EXPERIMENT_TYPES:
        return CandidacyDecision(False, "unknown_experiment_type")

    # Manual runs never auto-candidate
    if experiment_type == "manual":
        return CandidacyDecision(False, "manual_experiment_excluded")

    # Regime requirement (configurable)
    if config.require_regime and regime_oos is None:
        return CandidacyDecision(False, "missing_regime_oos")

    # Hard gates
    if metrics.n_trades_oos < config.min_trades:
        return CandidacyDecision(False, "insufficient_oos_trades")
    if metrics.max_dd_frac_oos > config.max_drawdown:
        return CandidacyDecision(False, "dd_too_high")
    if metrics.overfit_gap > config.max_overfit_gap:
        return CandidacyDecision(False, "overfit_too_high")
    if metrics.sharpe_oos < config.min_sharpe:
        return CandidacyDecision(False, "sharpe_too_low")

    return CandidacyDecision(True, "passed_all_gates")
```

### Circuit Breaker

Location: `app/services/kb/circuit_breaker.py`

```python
class CandidacyCircuitBreaker:
    MAX_CANDIDATE_RATE = 0.30      # candidates / successes
    RATE_WINDOW_SIZE = 50          # last N successful runs
    MAX_CANDIDATES_24H = 200       # rolling 24h cap
    COOLDOWN_HOURS = 6

    async def check(self, workspace_id: UUID) -> tuple[bool, Optional[str]]:
        # Check persisted breaker state
        state = await self._get_breaker_state(workspace_id)

        if state.kb_auto_candidacy_state == "disabled":
            return False, "disabled"

        if state.kb_auto_candidacy_state == "degraded":
            if datetime.now(UTC) < state.kb_auto_candidacy_disabled_until:
                return False, "cooldown"

        # Check rate over last N successful runs
        recent = await self._get_recent_successful_decisions(
            workspace_id, self.RATE_WINDOW_SIZE
        )
        if len(recent) >= self.RATE_WINDOW_SIZE:
            rate = sum(1 for d in recent if d == 'candidate') / len(recent)
            if rate > self.MAX_CANDIDATE_RATE:
                await self._trip_breaker(workspace_id, f"rate_spike:{rate:.2f}")
                return False, f"rate_spike:{rate:.2f}"

        # Check rolling 24h volume
        count_24h = await self._get_candidate_count_rolling_24h(workspace_id)
        if count_24h >= self.MAX_CANDIDATES_24H:
            await self._trip_breaker(workspace_id, f"daily_cap:{count_24h}")
            return False, f"daily_cap:{count_24h}"

        return True, None

    async def _trip_breaker(self, workspace_id: UUID, reason: str):
        await self._update_breaker_state(
            workspace_id,
            state="degraded",
            disabled_until=datetime.now(UTC) + timedelta(hours=self.COOLDOWN_HOURS),
            trip_reason=reason,
        )
```

### Breaker State Schema

```sql
ALTER TABLE workspaces
    ADD COLUMN kb_auto_candidacy_state TEXT DEFAULT 'enabled',
    ADD COLUMN kb_auto_candidacy_disabled_until TIMESTAMPTZ,
    ADD COLUMN kb_auto_candidacy_trip_reason TEXT,
    ADD COLUMN kb_auto_candidacy_tripped_at TIMESTAMPTZ;
```

### Integration in RunOrchestrator

```python
# After metrics + regime computed
decision = is_candidate(metrics, regime_oos, experiment_type, config)

if decision.eligible:
    allowed, trip_reason = await circuit_breaker.check(workspace_id)
    if allowed:
        kb_status = 'candidate'
        kb_gate = decision.reason
        kb_breaker = None
    else:
        kb_status = 'excluded'
        kb_gate = decision.reason
        kb_breaker = trip_reason
else:
    kb_status = 'excluded'
    kb_gate = decision.reason
    kb_breaker = None

run.kb_status = kb_status
run.auto_candidate_gate = kb_gate
run.auto_candidate_breaker = kb_breaker
```

---

## 5. State Machine Transitions

### Allowed Transitions

```
┌──────────┐     auto/admin      ┌───────────┐
│ excluded │ ──────────────────► │ candidate │
└──────────┘                     └───────────┘
     │                                 │
     │ admin                     admin │
     │                                 │
     ▼                                 ▼
┌──────────┐ ◄─────────────────► ┌──────────┐
│ promoted │      admin only     │ rejected │
└──────────┘                     └──────────┘
```

| From | To | Actor | Requires Reason |
|------|-----|-------|-----------------|
| `excluded` | `candidate` | auto, admin | No |
| `excluded` | `promoted` | admin | No |
| `candidate` | `promoted` | admin | No |
| `candidate` | `rejected` | admin | **Yes** |
| `promoted` | `rejected` | admin | **Yes** |
| `rejected` | `promoted` | admin | **Yes** (override) |

### Disallowed

- `rejected` → `candidate` (must go through `promoted`)
- Auto-anything to `promoted` or `rejected`

### Transition Validator

```python
class KBStatusTransition:
    ALLOWED = {
        ("excluded", "candidate"): {"auto", "admin"},
        ("excluded", "promoted"): {"admin"},
        ("candidate", "promoted"): {"admin"},
        ("candidate", "rejected"): {"admin"},
        ("promoted", "rejected"): {"admin"},
        ("rejected", "promoted"): {"admin"},
    }

    def validate(
        self,
        from_status: KBStatus,
        to_status: KBStatus,
        actor: Literal["auto", "admin"],
        reason: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        key = (from_status, to_status)

        if key not in self.ALLOWED:
            return False, f"transition_{from_status}_to_{to_status}_not_allowed"

        if actor not in self.ALLOWED[key]:
            return False, f"actor_{actor}_cannot_{from_status}_to_{to_status}"

        if to_status == "rejected" and not reason:
            return False, "rejection_requires_reason"
        if from_status == "rejected" and to_status == "promoted" and not reason:
            return False, "unrejection_requires_reason"

        return True, None
```

### Promotion Timestamp Rules

- Set `kb_promoted_at` only on transition **to** `promoted`
- `excluded → candidate`: do NOT set
- `rejected → promoted`: set fresh timestamp (new curation decision)

---

## 6. Audit Log

### History Table

```sql
CREATE TABLE kb_status_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    source_type TEXT NOT NULL,
    source_id UUID NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_id TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_kb_status_history_source
    ON kb_status_history(source_type, source_id);
CREATE INDEX idx_kb_status_history_workspace_created
    ON kb_status_history(workspace_id, created_at DESC);
```

### Transition Service

```python
class KBStatusService:
    async def transition(
        self,
        source_type: str,
        source_id: UUID,
        to_status: KBStatus,
        actor_type: Literal["auto", "admin"],
        actor_id: Optional[str] = None,
        reason: Optional[str] = None,
        trigger_ingest: bool = False,
    ) -> KBStatusResult:
        async with self.db.transaction():
            current = await self._get_current_status(source_type, source_id)

            valid, error = self.validator.validate(
                current.kb_status, to_status, actor_type, reason
            )
            if not valid:
                raise InvalidTransitionError(error)

            # Update row
            await self._update_status(
                source_type, source_id, to_status, actor_id, reason
            )

            # Set promoted_at on promotion
            if to_status == "promoted":
                await self._set_promoted_at(source_type, source_id, actor_id)

            # Append history
            await self._insert_history(
                workspace_id=current.workspace_id,
                source_type=source_type,
                source_id=source_id,
                from_status=current.kb_status,
                to_status=to_status,
                actor_type=actor_type,
                actor_id=actor_id,
                reason=reason,
            )

            # Archive on rejection
            if to_status == "rejected":
                await self.kb_index.archive_trial(
                    current.workspace_id, source_type, source_id,
                    reason="rejected", actor=actor_id
                )

            # Unarchive + optionally ingest on promotion from rejected
            if current.kb_status == "rejected" and to_status == "promoted":
                await self.kb_index.unarchive_trial(source_type, source_id)
                if trigger_ingest:
                    await self.ingest_single(source_type, source_id)

            return KBStatusResult(
                source_id=source_id,
                from_status=current.kb_status,
                to_status=to_status,
            )
```

---

## 7. Ingestion Idempotency

### kb_trial_index Table

```sql
CREATE TABLE kb_trial_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    source_type TEXT NOT NULL,
    source_id UUID NOT NULL,
    qdrant_point_id UUID NOT NULL,
    content_hash TEXT NOT NULL,
    content_hash_algo TEXT NOT NULL DEFAULT 'sha256_v1',
    regime_schema_version TEXT,
    embed_model TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    archived_at TIMESTAMPTZ,
    archived_reason TEXT,
    archived_by TEXT,

    UNIQUE (workspace_id, source_type, source_id)
);

CREATE INDEX idx_kb_trial_index_lookup
    ON kb_trial_index(workspace_id, source_type, source_id)
    WHERE archived_at IS NULL;
CREATE INDEX idx_kb_trial_index_qdrant
    ON kb_trial_index(qdrant_point_id);
```

### Deterministic Point ID

```python
KB_NAMESPACE = UUID("c8f4e2a1-5b3d-4c7e-9f1a-2d8b6e0c3a5f")

def compute_point_id(workspace_id: UUID, source_type: str, source_id: UUID) -> UUID:
    return uuid5(KB_NAMESPACE, f"{workspace_id}:{source_type}:{source_id}")
```

### Content Hash

```python
def compute_content_hash(trial: TrialDoc, collection_name: str) -> str:
    canonical = json.dumps({
        "embed_text": trial_to_text(trial),
        "collection": collection_name,
        "experiment_type": trial.experiment_type,
        "kb_status": trial.kb_status,
        "strategy_name": trial.strategy_name,
        "params": trial.params,
        "metrics": {
            "sharpe_oos": trial.sharpe_oos,
            "return_frac_oos": trial.return_frac_oos,
            "max_dd_frac_oos": trial.max_dd_frac_oos,
        },
        "regime_schema_version": trial.regime_schema_version,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
```

### Ingestion Logic

```python
async def ingest_trial(self, trial: TrialDoc) -> IngestResult:
    point_id = compute_point_id(trial.workspace_id, trial.source_type, trial.source_id)
    content_hash = compute_content_hash(trial, self.collection_name)

    existing = await self._get_index_entry(
        trial.workspace_id, trial.source_type, trial.source_id
    )

    if existing is None:
        await self._upsert_to_qdrant(point_id, trial)
        await self._insert_index(trial, point_id, content_hash)
        return IngestResult(action="inserted", point_id=point_id)

    if existing.archived_at is not None:
        await self._upsert_to_qdrant(point_id, trial)
        await self._unarchive_index(existing.id, content_hash)
        return IngestResult(action="unarchived", point_id=point_id)

    if existing.content_hash == content_hash:
        return IngestResult(action="skipped", point_id=point_id)

    await self._upsert_to_qdrant(point_id, trial)
    await self._update_index_hash(existing.id, content_hash)
    return IngestResult(action="updated", point_id=point_id)
```

---

## 8. Archive Policy

### Triggers

| Trigger | Archive? | Rationale |
|---------|----------|-----------|
| `→ rejected` | **Yes** | Explicit "never recommend" |
| `POST /admin/kb/trials/archive` | **Yes** | Manual removal |
| `candidate → excluded` | **No** | May be transient (breaker, thresholds) |

### Behavior

- **Delete from Qdrant immediately** (cost savings, index row preserves audit)
- Store `archived_reason` and `archived_by` for debugging
- Unarchive = clear `archived_at` + re-upsert on next ingest

### Retention

```sql
-- Optional: cleanup old archived entries after 90 days
DELETE FROM kb_trial_index
WHERE archived_at IS NOT NULL
  AND archived_at < NOW() - INTERVAL '90 days';
```

---

## 9. Tie-Break Rules

### Epsilon-Aware Comparator

When two trials have scores within ε = 0.02, apply tie-breaks in order:

| Priority | Rule | Rationale |
|----------|------|-----------|
| 1 | Primary score | Objective-based ranking |
| 2 | `promoted` > `candidate` | Human curation signal |
| 3 | Current schema > other > null | Prefer compatible |
| 4 | Higher `kb_promoted_at` | Recent curation |
| 5 | Newer `created_at` | Recency tiebreaker |

```python
from functools import cmp_to_key

EPSILON = 0.02
CURRENT_REGIME_SCHEMA = "regime_v1"

def compare_candidates(a: ScoredCandidate, b: ScoredCandidate) -> int:
    # Rule 1: Primary score
    if abs(a.score - b.score) > EPSILON:
        return -1 if a.score > b.score else 1

    # Within epsilon - apply tie-breaks

    # Rule 2: promoted > candidate
    if a.kb_status == "promoted" and b.kb_status != "promoted":
        return -1
    if b.kb_status == "promoted" and a.kb_status != "promoted":
        return 1

    # Rule 3: Schema preference
    a_rank = _schema_rank(a.regime_schema_version)
    b_rank = _schema_rank(b.regime_schema_version)
    if a_rank != b_rank:
        return -1 if a_rank < b_rank else 1

    # Rule 4: Recent promotion
    a_promoted = a.kb_promoted_at or datetime.min.replace(tzinfo=UTC)
    b_promoted = b.kb_promoted_at or datetime.min.replace(tzinfo=UTC)
    if a_promoted != b_promoted:
        return -1 if a_promoted > b_promoted else 1

    # Rule 5: Recency
    if a.created_at != b.created_at:
        return -1 if a.created_at > b.created_at else 1

    return 0

def _schema_rank(version: Optional[str]) -> int:
    if version == CURRENT_REGIME_SCHEMA:
        return 0
    if version is not None:
        return 1
    return 2

def rank_candidates(candidates: list[ScoredCandidate]) -> list[ScoredCandidate]:
    return sorted(candidates, key=cmp_to_key(compare_candidates))
```

---

## 10. Admin Endpoints

### Promotion Endpoints

```
POST /admin/kb/trials/promote
{
  "source_type": "test_variant",
  "source_ids": ["uuid1", "uuid2"],
  "trigger_ingest": true
}

POST /admin/kb/trials/reject
{
  "source_type": "test_variant",
  "source_ids": ["uuid1"],
  "reason": "Outlier - unrealistic fill assumptions"
}

POST /admin/kb/trials/mark-candidate
{
  "source_type": "test_variant",
  "source_ids": ["uuid1", "uuid2"]
}
```

**Response:**
```json
{
  "updated": 2,
  "skipped": 0,
  "ingested": 2,
  "results": [
    {"source_id": "uuid1", "group_id": "plan-123", "status": "promoted"},
    {"source_id": "uuid2", "group_id": "plan-123", "status": "promoted"}
  ],
  "errors": []
}
```

### Promotion Preview

```
GET /admin/kb/trials/promotion-preview
  ?source_type=test_variant
  &group_id=uuid
  &workspace_id=uuid
  &limit=50
  &offset=0
  &sort=sharpe_oos
  &include_ineligible=true
```

**Sort options:** `sharpe_oos`, `return_frac_oos`, `max_dd_frac_oos`, `created_at`

Secondary sort always: `created_at DESC`, `source_id` (deterministic)

**Response:**
```json
{
  "summary": {
    "would_promote": 12,
    "already_promoted": 3,
    "would_skip": 8,
    "missing_regime": 5
  },
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 23
  },
  "trials": [
    {
      "source_type": "test_variant",
      "source_id": "abc-123",
      "group_id": "run-plan-456",
      "kb_status": "candidate",
      "experiment_type": "sweep",
      "sharpe_oos": 0.85,
      "return_frac_oos": 0.12,
      "max_dd_frac_oos": 0.08,
      "n_trades_oos": 23,
      "passes_auto_gates": true,
      "can_promote": true,
      "is_eligible": true,
      "ineligibility_reasons": [],
      "has_regime_is": true,
      "has_regime_oos": true,
      "regime_schema_version": "regime_v1"
    }
  ]
}
```

**Eligibility fields:**
- `passes_auto_gates` - Pure candidacy gate check
- `can_promote` - Admin can promote (not rejected, has required fields)
- `is_eligible` - Would be included in bulk promote default selection

**Important:** Preview must use the same `is_candidate()` function as execution to avoid drift.

### Extended Ingestion Endpoint

```
POST /kb/trials/ingest
{
  "workspace_id": "...",
  "since": "2025-01-01T00:00:00Z",
  "sources": ["tune_runs", "test_variants"],
  "dry_run": false
}
```

**Response includes:**
```json
{
  "ingested_count": 57,
  "skipped_count": 12,
  "error_count": 0,
  "by_source": {
    "tune_runs": 45,
    "test_variants": 12
  }
}
```

---

## 11. Regime Data for Test Variants

### Compute at Execution Time

When a test variant completes successfully, the `RunOrchestrator` computes and stores:

- `regime_is` - Regime snapshot from IS segment
- `regime_oos` - Regime snapshot from OOS segment
- `regime_schema_version` - Current schema version

### Storage

First-class columns on `backtest_runs`:

```sql
ALTER TABLE backtest_runs
    ADD COLUMN regime_is JSONB,
    ADD COLUMN regime_oos JSONB,
    ADD COLUMN regime_schema_version TEXT;
```

### Policy

- Candidates must have `regime_oos` (filtered in view)
- Promoted can skip regime (human explicitly wanted it)
- Legacy runs without regime: exclude as candidates, allow promotion

---

## 12. TrialDoc Metadata

### Updated Schema

```python
class TrialDoc(BaseModel):
    # Identity
    source_type: Literal["tune_run", "test_variant"]
    source_id: UUID
    group_id: UUID  # tune_id or run_plan_id
    workspace_id: UUID

    # Experiment metadata
    experiment_type: Literal["tune", "sweep", "ablation", "manual"]
    strategy_name: str
    params: dict

    # KB status
    kb_status: KBStatus
    kb_promoted_at: Optional[datetime]

    # Regime
    regime_is: Optional[RegimeSnapshot]
    regime_oos: Optional[RegimeSnapshot]
    regime_schema_version: Optional[str]
    regime_missing: bool = False  # True if promoted without regime

    # Metrics
    sharpe_oos: Optional[float]
    return_frac_oos: Optional[float]
    max_dd_frac_oos: Optional[float]
    n_trades_oos: Optional[int]
    overfit_gap: Optional[float]

    # Provenance
    created_at: datetime
```

---

## Implementation Phases

### Phase 1: Schema + Status Model
- Add `kb_status` columns to both tables
- Add `kb_status_history` table
- Add `kb_trial_index` table
- Add workspace breaker state columns

### Phase 2: Candidacy Policy
- Implement `is_candidate()` gate function
- Implement `CandidacyCircuitBreaker`
- Integrate into `RunOrchestrator`

### Phase 3: State Machine + Audit
- Implement `KBStatusTransition` validator
- Implement `KBStatusService` with audit logging
- Add archive triggers

### Phase 4: Ingestion Pipeline
- Create `kb_eligible_trials` view
- Update ingestion endpoint to read from view
- Implement idempotency with `kb_trial_index`

### Phase 5: Admin UI
- Promotion preview endpoint
- Bulk promote/reject endpoints
- UI integration in run plan detail page

### Phase 6: Tie-Break + Retrieval
- Implement epsilon-aware comparator
- Add `kb_status` to TrialDoc metadata
- Wire into reranking pipeline

---

## Open Questions / Future Work

1. **Backfill job** - Sweep existing `backtest_runs` to apply candidacy gates retroactively
2. **Regime computation for legacy runs** - Background job to compute missing regime data
3. **Per-workspace candidacy config** - Override thresholds per workspace
4. **Quality drift monitoring** - Alert when median candidate quality drops
