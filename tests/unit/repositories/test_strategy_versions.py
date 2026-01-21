"""Tests for strategy versions repository."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock
from uuid import uuid4

import pytest

from app.repositories.strategy_versions import (
    StrategyVersion,
    VersionTransition,
    StrategyVersionsRepository,
    compute_config_hash,
    VALID_TRANSITIONS,
)


def create_mock_transaction():
    """Create a mock that works as an async context manager for transactions."""
    mock = MagicMock()
    mock.__aenter__ = AsyncMock(return_value=None)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


class TestComputeConfigHash:
    """Tests for config hash computation."""

    def test_compute_config_hash_deterministic(self):
        """Same config produces same hash."""
        config = {"param1": 10, "param2": "value"}
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_compute_config_hash_key_order_independent(self):
        """Key order doesn't affect hash (sorted)."""
        config1 = {"b": 2, "a": 1}
        config2 = {"a": 1, "b": 2}
        assert compute_config_hash(config1) == compute_config_hash(config2)

    def test_compute_config_hash_different_configs(self):
        """Different configs produce different hashes."""
        config1 = {"param": 10}
        config2 = {"param": 20}
        assert compute_config_hash(config1) != compute_config_hash(config2)

    def test_compute_config_hash_nested(self):
        """Nested objects are properly hashed."""
        config = {"outer": {"inner": [1, 2, 3]}}
        h = compute_config_hash(config)
        assert len(h) == 64


class TestStrategyVersion:
    """Tests for StrategyVersion dataclass."""

    def test_from_row_full(self):
        """Test creating from full database row."""
        row = {
            "id": uuid4(),
            "strategy_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "version_number": 1,
            "version_tag": "v1.0-beta",
            "config_snapshot": {"param": 10},
            "config_hash": "a" * 64,
            "state": "draft",
            "regime_awareness": {"enabled": True},
            "created_at": datetime.now(timezone.utc),
            "created_by": "admin:test",
            "activated_at": None,
            "paused_at": None,
            "retired_at": None,
            "kb_strategy_spec_id": uuid4(),
        }

        version = StrategyVersion.from_row(row)

        assert version.id == row["id"]
        assert version.strategy_id == row["strategy_id"]
        assert version.version_number == 1
        assert version.version_tag == "v1.0-beta"
        assert version.config_snapshot == {"param": 10}
        assert version.state == "draft"
        assert version.regime_awareness == {"enabled": True}
        assert version.created_by == "admin:test"

    def test_from_row_json_strings(self):
        """Test parsing JSON string fields."""
        row = {
            "id": uuid4(),
            "strategy_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "version_number": 2,
            "version_tag": None,
            "config_snapshot": json.dumps({"nested": {"value": 5}}),
            "config_hash": "b" * 64 + "  ",  # Trailing whitespace
            "state": "active",
            "regime_awareness": json.dumps({"modes": ["bull", "bear"]}),
            "created_at": datetime.now(timezone.utc),
            "created_by": None,
            "activated_at": datetime.now(timezone.utc),
            "paused_at": None,
            "retired_at": None,
            "kb_strategy_spec_id": None,
        }

        version = StrategyVersion.from_row(row)

        assert version.config_snapshot == {"nested": {"value": 5}}
        assert version.config_hash == "b" * 64  # Trimmed
        assert version.regime_awareness == {"modes": ["bull", "bear"]}


class TestVersionTransition:
    """Tests for VersionTransition dataclass."""

    def test_from_row(self):
        """Test creating transition from row."""
        row = {
            "id": uuid4(),
            "version_id": uuid4(),
            "from_state": "draft",
            "to_state": "active",
            "triggered_by": "admin:deploy",
            "triggered_at": datetime.now(timezone.utc),
            "reason": "Initial activation",
        }

        transition = VersionTransition.from_row(row)

        assert transition.from_state == "draft"
        assert transition.to_state == "active"
        assert transition.triggered_by == "admin:deploy"
        assert transition.reason == "Initial activation"

    def test_from_row_initial_transition(self):
        """Test transition with NULL from_state (initial creation)."""
        row = {
            "id": uuid4(),
            "version_id": uuid4(),
            "from_state": None,
            "to_state": "draft",
            "triggered_by": "system",
            "triggered_at": datetime.now(timezone.utc),
            "reason": "Version created",
        }

        transition = VersionTransition.from_row(row)

        assert transition.from_state is None
        assert transition.to_state == "draft"


class TestValidTransitions:
    """Tests for state transition rules."""

    def test_draft_transitions(self):
        """Draft can go to active or retired."""
        assert VALID_TRANSITIONS["draft"] == {"active", "retired"}

    def test_active_transitions(self):
        """Active can go to paused or retired."""
        assert VALID_TRANSITIONS["active"] == {"paused", "retired"}

    def test_paused_transitions(self):
        """Paused can go to active or retired."""
        assert VALID_TRANSITIONS["paused"] == {"active", "retired"}

    def test_retired_is_terminal(self):
        """Retired is terminal state."""
        assert VALID_TRANSITIONS["retired"] == set()


class TestStrategyVersionsRepository:
    """Tests for StrategyVersionsRepository."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock database pool."""
        return MagicMock()

    def test_repository_creation(self, mock_pool):
        """Test creating repository instance."""
        repo = StrategyVersionsRepository(mock_pool)
        assert repo.pool == mock_pool

    @pytest.mark.asyncio
    async def test_create_version(self, mock_pool):
        """Test creating a new version."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        strategy_id = uuid4()
        entity_id = uuid4()
        version_id = uuid4()

        # Mock parent strategy query
        mock_conn.fetchrow.side_effect = [
            {"strategy_entity_id": entity_id},  # Parent lookup
            {  # Version insert result
                "id": version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": entity_id,
                "version_number": 1,
                "version_tag": "v1.0",
                "config_snapshot": {"param": 10},
                "config_hash": "a" * 64,
                "state": "draft",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin:test",
                "activated_at": None,
                "paused_at": None,
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
        ]
        mock_conn.fetchval.return_value = 0  # No existing versions
        # transaction() is sync in asyncpg, returns context manager
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.create_version(
            strategy_id=strategy_id,
            config_snapshot={"param": 10},
            created_by="admin:test",
            version_tag="v1.0",
        )

        assert version.id == version_id
        assert version.version_number == 1
        assert version.state == "draft"
        assert version.config_snapshot == {"param": 10}

    @pytest.mark.asyncio
    async def test_create_version_strategy_not_found(self, mock_pool):
        """Test creating version for nonexistent strategy."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None  # Strategy not found
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        repo = StrategyVersionsRepository(mock_pool)

        with pytest.raises(ValueError, match="not found"):
            await repo.create_version(
                strategy_id=uuid4(),
                config_snapshot={"param": 10},
            )

    @pytest.mark.asyncio
    async def test_create_version_no_entity_id(self, mock_pool):
        """Test creating version when strategy has no entity_id mapping."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = {"strategy_entity_id": None}
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        repo = StrategyVersionsRepository(mock_pool)

        with pytest.raises(ValueError, match="no strategy_entity_id mapping"):
            await repo.create_version(
                strategy_id=uuid4(),
                config_snapshot={"param": 10},
            )

    @pytest.mark.asyncio
    async def test_get_version(self, mock_pool):
        """Test getting version by ID."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        version_id = uuid4()
        mock_conn.fetchrow.return_value = {
            "id": version_id,
            "strategy_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "version_number": 1,
            "version_tag": None,
            "config_snapshot": {"param": 5},
            "config_hash": "c" * 64,
            "state": "active",
            "regime_awareness": {},
            "created_at": datetime.now(timezone.utc),
            "created_by": "system",
            "activated_at": datetime.now(timezone.utc),
            "paused_at": None,
            "retired_at": None,
            "kb_strategy_spec_id": None,
        }

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.get_version(version_id)

        assert version is not None
        assert version.id == version_id
        assert version.state == "active"

    @pytest.mark.asyncio
    async def test_get_version_not_found(self, mock_pool):
        """Test getting nonexistent version."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.get_version(uuid4())

        assert version is None

    @pytest.mark.asyncio
    async def test_list_versions(self, mock_pool):
        """Test listing versions for a strategy."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        strategy_id = uuid4()
        mock_conn.fetchval.return_value = 2
        mock_conn.fetch.return_value = [
            {
                "id": uuid4(),
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 2,
                "version_tag": "v2.0",
                "config_snapshot": {"param": 20},
                "config_hash": "d" * 64,
                "state": "active",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": datetime.now(timezone.utc),
                "paused_at": None,
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
            {
                "id": uuid4(),
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 1,
                "version_tag": "v1.0",
                "config_snapshot": {"param": 10},
                "config_hash": "e" * 64,
                "state": "paused",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": None,
                "paused_at": datetime.now(timezone.utc),
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
        ]

        repo = StrategyVersionsRepository(mock_pool)
        versions, total = await repo.list_versions(strategy_id)

        assert total == 2
        assert len(versions) == 2
        assert versions[0].version_number == 2
        assert versions[1].version_number == 1

    @pytest.mark.asyncio
    async def test_activate_from_draft(self, mock_pool):
        """Test activating a draft version."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        version_id = uuid4()
        strategy_id = uuid4()

        # Version to activate
        mock_conn.fetchrow.side_effect = [
            {  # Version lookup
                "id": version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 1,
                "version_tag": None,
                "config_snapshot": {"param": 10},
                "config_hash": "f" * 64,
                "state": "draft",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": None,
                "paused_at": None,
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
            None,  # No existing active version
            {  # Updated version
                "id": version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 1,
                "version_tag": None,
                "config_snapshot": {"param": 10},
                "config_hash": "f" * 64,
                "state": "active",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": datetime.now(timezone.utc),
                "paused_at": None,
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
        ]

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.activate(
            version_id=version_id,
            triggered_by="admin:deploy",
            reason="Initial deployment",
        )

        assert version.state == "active"
        assert version.activated_at is not None

    @pytest.mark.asyncio
    async def test_activate_pauses_old_active(self, mock_pool):
        """Test that activating a version pauses the old active."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        new_version_id = uuid4()
        old_version_id = uuid4()
        strategy_id = uuid4()

        mock_conn.fetchrow.side_effect = [
            {  # New version lookup
                "id": new_version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 2,
                "version_tag": None,
                "config_snapshot": {"param": 20},
                "config_hash": "g" * 64,
                "state": "draft",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": None,
                "paused_at": None,
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
            {  # Old active version
                "id": old_version_id,
                "version_number": 1,
            },
            {  # Updated new version
                "id": new_version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 2,
                "version_tag": None,
                "config_snapshot": {"param": 20},
                "config_hash": "g" * 64,
                "state": "active",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": datetime.now(timezone.utc),
                "paused_at": None,
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
        ]

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.activate(
            version_id=new_version_id,
            triggered_by="admin:upgrade",
        )

        assert version.state == "active"
        # Verify old version was paused (execute was called)
        assert mock_conn.execute.call_count >= 3  # pause old + update new + transition

    @pytest.mark.asyncio
    async def test_activate_invalid_state(self, mock_pool):
        """Test activating from invalid state (retired)."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        version_id = uuid4()
        mock_conn.fetchrow.return_value = {
            "id": version_id,
            "strategy_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "version_number": 1,
            "version_tag": None,
            "config_snapshot": {},
            "config_hash": "h" * 64,
            "state": "retired",  # Cannot activate from retired
            "regime_awareness": {},
            "created_at": datetime.now(timezone.utc),
            "created_by": "admin",
            "activated_at": None,
            "paused_at": None,
            "retired_at": datetime.now(timezone.utc),
            "kb_strategy_spec_id": None,
        }

        repo = StrategyVersionsRepository(mock_pool)

        with pytest.raises(ValueError, match="Cannot activate from state 'retired'"):
            await repo.activate(version_id=version_id, triggered_by="admin")

    @pytest.mark.asyncio
    async def test_pause(self, mock_pool):
        """Test pausing an active version."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        version_id = uuid4()
        strategy_id = uuid4()

        mock_conn.fetchrow.side_effect = [
            {  # Version lookup
                "id": version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 1,
                "version_tag": None,
                "config_snapshot": {},
                "config_hash": "i" * 64,
                "state": "active",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": datetime.now(timezone.utc),
                "paused_at": None,
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
            {  # Updated version
                "id": version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 1,
                "version_tag": None,
                "config_snapshot": {},
                "config_hash": "i" * 64,
                "state": "paused",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": datetime.now(timezone.utc),
                "paused_at": datetime.now(timezone.utc),
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
        ]

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.pause(
            version_id=version_id,
            triggered_by="admin:emergency",
            reason="Market volatility",
        )

        assert version.state == "paused"

    @pytest.mark.asyncio
    async def test_pause_invalid_state(self, mock_pool):
        """Test pausing from invalid state (draft)."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        mock_conn.fetchrow.return_value = {
            "id": uuid4(),
            "strategy_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "version_number": 1,
            "version_tag": None,
            "config_snapshot": {},
            "config_hash": "j" * 64,
            "state": "draft",  # Cannot pause draft
            "regime_awareness": {},
            "created_at": datetime.now(timezone.utc),
            "created_by": "admin",
            "activated_at": None,
            "paused_at": None,
            "retired_at": None,
            "kb_strategy_spec_id": None,
        }

        repo = StrategyVersionsRepository(mock_pool)

        with pytest.raises(ValueError, match="Cannot pause from state 'draft'"):
            await repo.pause(version_id=uuid4(), triggered_by="admin")

    @pytest.mark.asyncio
    async def test_retire(self, mock_pool):
        """Test retiring a version."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        version_id = uuid4()
        strategy_id = uuid4()

        mock_conn.fetchrow.side_effect = [
            {  # Version lookup
                "id": version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 1,
                "version_tag": None,
                "config_snapshot": {},
                "config_hash": "k" * 64,
                "state": "paused",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": None,
                "paused_at": datetime.now(timezone.utc),
                "retired_at": None,
                "kb_strategy_spec_id": None,
            },
            {  # Updated version
                "id": version_id,
                "strategy_id": strategy_id,
                "strategy_entity_id": uuid4(),
                "version_number": 1,
                "version_tag": None,
                "config_snapshot": {},
                "config_hash": "k" * 64,
                "state": "retired",
                "regime_awareness": {},
                "created_at": datetime.now(timezone.utc),
                "created_by": "admin",
                "activated_at": None,
                "paused_at": datetime.now(timezone.utc),
                "retired_at": datetime.now(timezone.utc),
                "kb_strategy_spec_id": None,
            },
        ]

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.retire(
            version_id=version_id,
            triggered_by="admin:cleanup",
            reason="Obsolete config",
        )

        assert version.state == "retired"

    @pytest.mark.asyncio
    async def test_retire_already_retired(self, mock_pool):
        """Test retiring an already retired version."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.transaction = MagicMock(return_value=create_mock_transaction())

        mock_conn.fetchrow.return_value = {
            "id": uuid4(),
            "strategy_id": uuid4(),
            "strategy_entity_id": uuid4(),
            "version_number": 1,
            "version_tag": None,
            "config_snapshot": {},
            "config_hash": "l" * 64,
            "state": "retired",
            "regime_awareness": {},
            "created_at": datetime.now(timezone.utc),
            "created_by": "admin",
            "activated_at": None,
            "paused_at": None,
            "retired_at": datetime.now(timezone.utc),
            "kb_strategy_spec_id": None,
        }

        repo = StrategyVersionsRepository(mock_pool)

        with pytest.raises(ValueError, match="Cannot retire from state 'retired'"):
            await repo.retire(version_id=uuid4(), triggered_by="admin")

    @pytest.mark.asyncio
    async def test_get_transitions(self, mock_pool):
        """Test getting state transition history."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        version_id = uuid4()
        mock_conn.fetch.return_value = [
            {
                "id": uuid4(),
                "version_id": version_id,
                "from_state": "draft",
                "to_state": "active",
                "triggered_by": "admin:deploy",
                "triggered_at": datetime.now(timezone.utc),
                "reason": "Initial activation",
            },
            {
                "id": uuid4(),
                "version_id": version_id,
                "from_state": None,
                "to_state": "draft",
                "triggered_by": "system",
                "triggered_at": datetime.now(timezone.utc),
                "reason": "Version created",
            },
        ]

        repo = StrategyVersionsRepository(mock_pool)
        transitions = await repo.get_transitions(version_id)

        assert len(transitions) == 2
        assert transitions[0].to_state == "active"
        assert transitions[1].from_state is None

    @pytest.mark.asyncio
    async def test_get_version_by_hash(self, mock_pool):
        """Test getting version by config hash."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        strategy_id = uuid4()
        config_hash = "m" * 64

        mock_conn.fetchrow.return_value = {
            "id": uuid4(),
            "strategy_id": strategy_id,
            "strategy_entity_id": uuid4(),
            "version_number": 1,
            "version_tag": None,
            "config_snapshot": {"param": 10},
            "config_hash": config_hash,
            "state": "draft",
            "regime_awareness": {},
            "created_at": datetime.now(timezone.utc),
            "created_by": "system",
            "activated_at": None,
            "paused_at": None,
            "retired_at": None,
            "kb_strategy_spec_id": None,
        }

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.get_version_by_hash(strategy_id, config_hash)

        assert version is not None
        assert version.config_hash == config_hash

    @pytest.mark.asyncio
    async def test_get_active_version(self, mock_pool):
        """Test getting the active version for a strategy."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        strategy_id = uuid4()
        mock_conn.fetchrow.return_value = {
            "id": uuid4(),
            "strategy_id": strategy_id,
            "strategy_entity_id": uuid4(),
            "version_number": 2,
            "version_tag": "production",
            "config_snapshot": {"param": 50},
            "config_hash": "n" * 64,
            "state": "active",
            "regime_awareness": {},
            "created_at": datetime.now(timezone.utc),
            "created_by": "admin",
            "activated_at": datetime.now(timezone.utc),
            "paused_at": None,
            "retired_at": None,
            "kb_strategy_spec_id": None,
        }

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.get_active_version(strategy_id)

        assert version is not None
        assert version.state == "active"
        assert version.version_tag == "production"

    @pytest.mark.asyncio
    async def test_get_active_version_none(self, mock_pool):
        """Test getting active version when none exists."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_conn.fetchrow.return_value = None

        repo = StrategyVersionsRepository(mock_pool)
        version = await repo.get_active_version(uuid4())

        assert version is None
