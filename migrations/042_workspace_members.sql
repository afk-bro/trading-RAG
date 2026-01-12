-- migrations/042_workspace_members.sql
-- Workspace membership for user authorization

CREATE TABLE IF NOT EXISTS workspace_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT workspace_members_role_check
        CHECK (role IN ('owner', 'admin', 'member', 'viewer')),
    CONSTRAINT uq_workspace_member
        UNIQUE (workspace_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_workspace_members_user
    ON workspace_members(user_id);

CREATE INDEX IF NOT EXISTS idx_workspace_members_workspace
    ON workspace_members(workspace_id);

-- Seed: Make existing workspace owners members
INSERT INTO workspace_members (workspace_id, user_id, role)
SELECT id, owner_id, 'owner'
FROM workspaces
WHERE owner_id IS NOT NULL
ON CONFLICT DO NOTHING;
