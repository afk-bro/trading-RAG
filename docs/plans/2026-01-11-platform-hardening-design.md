# Phase 1 Platform Hardening Design

**Date:** 2026-01-11
**Status:** Approved
**Scope:** Run plan idempotency, event retention, Supabase auth wiring

---

## Overview

Three foundational improvements to harden the platform for production use:

1. **Run Plan Idempotency** - Prevent duplicate run plans from client retries
2. **Event Retention** - Tiered retention with daily rollups for historical analytics
3. **Auth Wiring** - Supabase Auth integration with workspace-level authorization

---

## 1. Run Plan Idempotency

### Problem

Client retries (timeouts, network blips) can create duplicate `run_plans`. Need safe retry semantics.

### Design

**Dual-key approach:**
- `idempotency_key` - Client-provided (e.g., `X-Idempotency-Key` header)
- `request_hash` - Server-computed from canonical request body

**Behavior:**
- If `idempotency_key` matches existing row → return that row (409 if status beyond `pending`)
- If `request_hash` matches but key differs → 409 Conflict (same request, different key)
- Otherwise → create new plan

### Schema

```sql
ALTER TABLE run_plans
    ADD COLUMN idempotency_key TEXT,
    ADD COLUMN request_hash TEXT;

-- Unique constraint on idempotency_key (when provided)
CREATE UNIQUE INDEX idx_run_plans_idempotency_key
    ON run_plans(idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Index for request_hash lookups
CREATE INDEX idx_run_plans_request_hash
    ON run_plans(request_hash)
    WHERE request_hash IS NOT NULL;
```

### Request Hash Canonicalization

```python
def compute_request_hash(request: RunPlanRequest) -> str:
    """Canonical hash of request for duplicate detection."""
    canonical = {
        "workspace_id": str(request.workspace_id),
        "strategy_entity_id": str(request.strategy_entity_id) if request.strategy_entity_id else None,
        "objective_name": request.objective_name,
        "plan": request.plan,  # Full plan dict
    }
    # Sort keys, dump with separators for determinism
    json_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:32]
```

### API Behavior

```python
async def create_run_plan(
    request: RunPlanRequest,
    idempotency_key: Optional[str] = Header(None, alias="X-Idempotency-Key"),
) -> RunPlanResponse:
    request_hash = compute_request_hash(request)

    # Check for existing by idempotency_key
    if idempotency_key:
        existing = await repo.get_by_idempotency_key(idempotency_key)
        if existing:
            if existing.status != "pending":
                raise HTTPException(409, f"Plan {existing.id} already {existing.status}")
            return RunPlanResponse(id=existing.id, status="existing")

    # Check for existing by request_hash
    existing_by_hash = await repo.get_by_request_hash(request_hash)
    if existing_by_hash:
        raise HTTPException(409, f"Duplicate request (plan {existing_by_hash.id})")

    # Create new plan
    plan_id = await repo.create_run_plan(
        ...,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
    )
    return RunPlanResponse(id=plan_id, status="created")
```

### Race Safety

Use INSERT with ON CONFLICT for atomic upsert:

```sql
INSERT INTO run_plans (
    workspace_id, strategy_entity_id, objective_name,
    n_variants, plan, status, idempotency_key, request_hash
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL
DO UPDATE SET id = run_plans.id  -- no-op update
RETURNING id, (xmax = 0) AS inserted;
```

---

## 2. Event Retention

### Problem

`trade_events` journal grows unbounded. Need tiered retention that preserves observability while managing storage.

### Design

**Tiered retention by severity:**
- INFO/DEBUG events: 30 days
- WARN/ERROR events: 90 days
- Pinned events (`pinned=true`): Never deleted

**Daily rollups** for historical analytics (event counts by type/strategy/day).

### Schema Changes

```sql
-- Add severity and pinned columns to trade_events
ALTER TABLE trade_events
    ADD COLUMN severity TEXT NOT NULL DEFAULT 'info'
        CHECK (severity IN ('debug', 'info', 'warn', 'error')),
    ADD COLUMN pinned BOOLEAN NOT NULL DEFAULT FALSE;

-- Index for retention queries
CREATE INDEX idx_trade_events_retention
    ON trade_events(created_at, severity, pinned);

-- Daily rollup table
CREATE TABLE trade_event_rollups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id),
    strategy_entity_id UUID REFERENCES kb_entities(id),
    event_type TEXT NOT NULL,
    rollup_date DATE NOT NULL,
    event_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    sample_correlation_ids TEXT[],  -- Up to 5 sample IDs for drilldown
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_rollup_key UNIQUE (workspace_id, strategy_entity_id, event_type, rollup_date)
);

CREATE INDEX idx_rollups_workspace_date
    ON trade_event_rollups(workspace_id, rollup_date DESC);
```

### Rollup Job

```python
async def run_daily_rollup(target_date: date) -> int:
    """
    Aggregate events from target_date into rollups.
    Idempotent via UNIQUE constraint + ON CONFLICT.
    """
    query = """
        INSERT INTO trade_event_rollups (
            workspace_id, strategy_entity_id, event_type, rollup_date,
            event_count, error_count, sample_correlation_ids
        )
        SELECT
            workspace_id,
            strategy_entity_id,
            event_type,
            $1::date as rollup_date,
            COUNT(*) as event_count,
            COUNT(*) FILTER (WHERE severity = 'error') as error_count,
            (ARRAY_AGG(DISTINCT correlation_id) FILTER (WHERE correlation_id IS NOT NULL))[1:5]
        FROM trade_events
        WHERE created_at >= $1::date
          AND created_at < ($1::date + INTERVAL '1 day')
        GROUP BY workspace_id, strategy_entity_id, event_type
        ON CONFLICT (workspace_id, strategy_entity_id, event_type, rollup_date)
        DO UPDATE SET
            event_count = EXCLUDED.event_count,
            error_count = EXCLUDED.error_count,
            sample_correlation_ids = EXCLUDED.sample_correlation_ids;
    """
    async with pool.acquire() as conn:
        result = await conn.execute(query, target_date)
    return int(result.split()[-1])  # Returns row count
```

### Retention Job

```python
async def run_retention_cleanup() -> dict:
    """
    Delete expired events based on severity tier.
    Returns counts of deleted events per tier.
    """
    now = datetime.utcnow()

    # Delete INFO/DEBUG older than 30 days (not pinned)
    info_cutoff = now - timedelta(days=30)
    info_deleted = await conn.execute("""
        DELETE FROM trade_events
        WHERE created_at < $1
          AND severity IN ('debug', 'info')
          AND pinned = FALSE
    """, info_cutoff)

    # Delete WARN/ERROR older than 90 days (not pinned)
    error_cutoff = now - timedelta(days=90)
    error_deleted = await conn.execute("""
        DELETE FROM trade_events
        WHERE created_at < $1
          AND severity IN ('warn', 'error')
          AND pinned = FALSE
    """, error_cutoff)

    return {
        "info_debug_deleted": int(info_deleted.split()[-1]),
        "warn_error_deleted": int(error_deleted.split()[-1]),
    }
```

### Job Scheduling

**Option A (preferred):** pg_cron if available on Supabase
```sql
-- Daily at 2 AM UTC
SELECT cron.schedule('rollup-events', '0 2 * * *', $$
    SELECT run_daily_rollup(CURRENT_DATE - INTERVAL '1 day');
$$);

SELECT cron.schedule('cleanup-events', '0 3 * * *', $$
    SELECT run_retention_cleanup();
$$);
```

**Option B:** External cron via admin endpoint
```
POST /admin/jobs/rollup-events  (protected by X-Admin-Token)
POST /admin/jobs/cleanup-events
```

---

## 3. Supabase Auth Wiring

### Problem

Current auth is admin-token only. Need user identity for workspace-scoped authorization.

### Design

**Authentication:** `Authorization: Bearer <JWT>` validated via Supabase Auth API
**Admin bypass:** `X-Admin-Token` header for service-to-service calls
**Authorization:** Workspace membership checked via `workspace_members` table

### Schema

```sql
CREATE TABLE workspace_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,  -- References auth.users in Supabase
    role TEXT NOT NULL DEFAULT 'member'
        CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_workspace_member UNIQUE (workspace_id, user_id)
);

CREATE INDEX idx_workspace_members_user ON workspace_members(user_id);
CREATE INDEX idx_workspace_members_workspace ON workspace_members(workspace_id);
```

### Request Context

```python
@dataclass
class RequestContext:
    """Auth context resolved from request."""
    user_id: Optional[UUID] = None
    workspace_id: Optional[UUID] = None
    role: Optional[str] = None
    is_admin: bool = False

ROLE_RANK = {"viewer": 1, "member": 2, "admin": 3, "owner": 4}
```

### Dependencies

```python
# app/deps/security.py

async def get_current_user(
    authorization: str = Header(None),
    x_admin_token: str = Header(None, alias="X-Admin-Token"),
) -> RequestContext:
    """
    Resolve user identity from JWT or admin token.
    Does NOT resolve workspace - that's separate.
    """
    # (1) Admin token bypass
    if x_admin_token:
        if verify_admin_token(x_admin_token):
            return RequestContext(is_admin=True)
        raise HTTPException(401, "Invalid admin token")

    # (2) Require Authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing authorization header")

    token = authorization.split(" ", 1)[1]

    # (3) Validate via Supabase Auth API
    try:
        user = await supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(401, "Invalid or expired token")
        return RequestContext(user_id=UUID(user.user.id))
    except Exception as e:
        logger.warning("Auth validation failed", error=str(e))
        raise HTTPException(401, "Authentication failed")


async def require_workspace_access(
    ctx: RequestContext = Depends(get_current_user),
    workspace_id: UUID = Query(...),
    min_role: str = "viewer",
    pool = Depends(get_pool),
) -> RequestContext:
    """
    Verify user has access to workspace with minimum role.
    Admin bypass still requires explicit workspace_id.
    """
    # Admin bypass - still needs workspace_id for scoping
    if ctx.is_admin:
        return RequestContext(
            is_admin=True,
            workspace_id=workspace_id,
        )

    # Look up membership
    query = """
        SELECT role FROM workspace_members
        WHERE workspace_id = $1 AND user_id = $2
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, workspace_id, ctx.user_id)

    if not row:
        raise HTTPException(403, "Not a member of this workspace")

    role = row["role"]
    if ROLE_RANK.get(role, 0) < ROLE_RANK.get(min_role, 0):
        raise HTTPException(403, f"Requires {min_role} role, you have {role}")

    return RequestContext(
        user_id=ctx.user_id,
        workspace_id=workspace_id,
        role=role,
    )
```

### Endpoint Integration

```python
# Example: workspace-scoped endpoint with auth

@router.get("/kb/trials/recommend")
async def recommend_trials(
    ctx: RequestContext = Depends(require_workspace_access),
    # ctx.workspace_id is guaranteed to be set
    # ctx.user_id is set (unless admin)
    # ctx.role is set (unless admin)
):
    # Use ctx.workspace_id for all queries
    ...
```

### Migration Path

1. Add `workspace_members` table
2. Seed initial memberships (workspace owners)
3. Update endpoints incrementally:
   - Start with new endpoints using `require_workspace_access`
   - Existing endpoints continue with admin-token until migrated
4. Eventually require auth on all workspace-scoped endpoints

---

## Implementation Order

1. **Run Plan Idempotency** (lowest risk, immediate value)
   - Migration: Add columns to `run_plans`
   - Repository: Update `create_run_plan`
   - Router: Add idempotency handling

2. **Event Retention** (medium risk, storage management)
   - Migration: Add columns to `trade_events`, create `trade_event_rollups`
   - Service: Rollup and cleanup jobs
   - Admin endpoint or pg_cron scheduling

3. **Auth Wiring** (highest risk, incremental rollout)
   - Migration: Create `workspace_members`
   - Dependencies: `get_current_user`, `require_workspace_access`
   - Seed: Initial workspace memberships
   - Migrate endpoints incrementally

---

## Testing Strategy

### Idempotency
- Unit: Hash canonicalization determinism
- Integration: Duplicate key returns existing, duplicate hash returns 409
- Contract: Header parsing, response codes

### Retention
- Unit: Rollup aggregation logic
- Integration: Cleanup respects severity tiers and pinned flag
- Golden: Rollup counts match event counts

### Auth
- Unit: Token parsing, role ranking
- Integration: Membership checks, 403 on missing/insufficient role
- Contract: Header handling, error responses
