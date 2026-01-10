"""Unit tests for KB router endpoints."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.schemas import (
    KBEntityType,
    KBClaimType,
    KBClaimStatus,
    KBEntityItem,
    KBEntityListResponse,
    KBEntityDetailResponse,
    KBEntityStats,
    KBClaimItem,
    KBClaimListResponse,
    KBClaimDetailResponse,
    KBEvidenceItem,
)


class TestListEntities:
    """Tests for GET /kb/entities endpoint."""

    @pytest.fixture
    def mock_kb_repo(self):
        """Create mock KB repository."""
        return MagicMock()

    @pytest.fixture
    def sample_entities(self):
        """Sample entity data from DB."""
        return [
            {
                "id": uuid4(),
                "type": "strategy",
                "name": "Pairs Mean Reversion",
                "aliases": json.dumps(["pairs trading", "statistical arbitrage"]),
                "description": "A market-neutral trading strategy",
                "verified_claim_count": 12,
                "created_at": datetime.now(),
            },
            {
                "id": uuid4(),
                "type": "indicator",
                "name": "RSI",
                "aliases": json.dumps(["Relative Strength Index"]),
                "description": "Momentum oscillator",
                "verified_claim_count": 8,
                "created_at": datetime.now(),
            },
        ]

    def test_entity_item_from_db_row(self, sample_entities):
        """Test converting DB row to KBEntityItem."""
        row = sample_entities[0]
        item = KBEntityItem(
            id=row["id"],
            type=KBEntityType(row["type"]),
            name=row["name"],
            aliases=(
                json.loads(row["aliases"])
                if isinstance(row["aliases"], str)
                else row["aliases"]
            ),
            description=row["description"],
            verified_claim_count=row["verified_claim_count"],
            created_at=row["created_at"],
        )

        assert item.name == "Pairs Mean Reversion"
        assert item.type == KBEntityType.STRATEGY
        assert "pairs trading" in item.aliases
        assert item.verified_claim_count == 12

    def test_entity_list_response_structure(self, sample_entities):
        """Test KBEntityListResponse structure."""
        items = [
            KBEntityItem(
                id=e["id"],
                type=KBEntityType(e["type"]),
                name=e["name"],
                aliases=json.loads(e["aliases"]),
                description=e["description"],
                verified_claim_count=e["verified_claim_count"],
            )
            for e in sample_entities
        ]

        response = KBEntityListResponse(
            items=items,
            total=123,
            limit=50,
            offset=0,
        )

        assert len(response.items) == 2
        assert response.total == 123
        assert response.limit == 50
        assert response.offset == 0

    def test_entity_type_enum_values(self):
        """Test all entity type enum values."""
        assert KBEntityType.CONCEPT.value == "concept"
        assert KBEntityType.INDICATOR.value == "indicator"
        assert KBEntityType.STRATEGY.value == "strategy"
        assert KBEntityType.EQUATION.value == "equation"
        assert KBEntityType.PATTERN.value == "pattern"
        assert KBEntityType.ASSET.value == "asset"
        assert KBEntityType.OTHER.value == "other"


class TestGetEntity:
    """Tests for GET /kb/entities/{entity_id} endpoint."""

    def test_entity_detail_response_with_stats(self):
        """Test KBEntityDetailResponse with stats."""
        entity_id = uuid4()
        response = KBEntityDetailResponse(
            id=entity_id,
            type=KBEntityType.STRATEGY,
            name="Momentum",
            aliases=["trend following"],
            description="Buy high, sell higher",
            stats=KBEntityStats(
                verified_claims=15,
                weak_claims=3,
                total_claims=20,
                relations_count=5,
            ),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        assert response.stats.verified_claims == 15
        assert response.stats.weak_claims == 3
        assert response.stats.total_claims == 20
        assert response.stats.relations_count == 5

    def test_entity_stats_defaults(self):
        """Test KBEntityStats default values."""
        stats = KBEntityStats()

        assert stats.verified_claims == 0
        assert stats.weak_claims == 0
        assert stats.total_claims == 0
        assert stats.relations_count == 0


class TestListClaims:
    """Tests for GET /kb/claims endpoint."""

    @pytest.fixture
    def sample_claims(self):
        """Sample claim data from DB."""
        return [
            {
                "id": uuid4(),
                "claim_type": "definition",
                "text": "RSI measures the speed and magnitude of recent price changes",
                "status": "verified",
                "confidence": 0.92,
                "entity_id": uuid4(),
                "entity_name": "RSI",
                "entity_type": "indicator",
                "created_at": datetime.now(),
            },
            {
                "id": uuid4(),
                "claim_type": "rule",
                "text": "RSI above 70 indicates overbought conditions",
                "status": "verified",
                "confidence": 0.88,
                "entity_id": uuid4(),
                "entity_name": "RSI",
                "entity_type": "indicator",
                "created_at": datetime.now(),
            },
        ]

    def test_claim_item_from_db_row(self, sample_claims):
        """Test converting DB row to KBClaimItem."""
        row = sample_claims[0]
        item = KBClaimItem(
            id=row["id"],
            claim_type=KBClaimType(row["claim_type"]),
            text=row["text"],
            status=KBClaimStatus(row["status"]),
            confidence=row["confidence"],
            entity_id=row["entity_id"],
            entity_name=row["entity_name"],
            entity_type=KBEntityType(row["entity_type"]),
            created_at=row["created_at"],
        )

        assert item.claim_type == KBClaimType.DEFINITION
        assert item.status == KBClaimStatus.VERIFIED
        assert item.confidence == 0.92
        assert item.entity_name == "RSI"

    def test_claim_list_response_structure(self, sample_claims):
        """Test KBClaimListResponse structure."""
        items = [
            KBClaimItem(
                id=c["id"],
                claim_type=KBClaimType(c["claim_type"]),
                text=c["text"],
                status=KBClaimStatus(c["status"]),
                confidence=c["confidence"],
                entity_id=c["entity_id"],
                entity_name=c["entity_name"],
                entity_type=KBEntityType(c["entity_type"]),
            )
            for c in sample_claims
        ]

        response = KBClaimListResponse(
            items=items,
            total=50,
            limit=20,
            offset=10,
        )

        assert len(response.items) == 2
        assert response.total == 50
        assert response.offset == 10

    def test_claim_type_enum_values(self):
        """Test all claim type enum values."""
        assert KBClaimType.DEFINITION.value == "definition"
        assert KBClaimType.RULE.value == "rule"
        assert KBClaimType.ASSUMPTION.value == "assumption"
        assert KBClaimType.WARNING.value == "warning"
        assert KBClaimType.PARAMETER.value == "parameter"
        assert KBClaimType.EQUATION.value == "equation"
        assert KBClaimType.OBSERVATION.value == "observation"
        assert KBClaimType.RECOMMENDATION.value == "recommendation"

    def test_claim_status_enum_values(self):
        """Test all claim status enum values."""
        assert KBClaimStatus.PENDING.value == "pending"
        assert KBClaimStatus.VERIFIED.value == "verified"
        assert KBClaimStatus.WEAK.value == "weak"
        assert KBClaimStatus.REJECTED.value == "rejected"


class TestGetClaim:
    """Tests for GET /kb/claims/{claim_id} endpoint."""

    def test_claim_detail_response_with_evidence(self):
        """Test KBClaimDetailResponse with evidence."""
        claim_id = uuid4()
        evidence = [
            KBEvidenceItem(
                id=uuid4(),
                doc_id=uuid4(),
                chunk_id=uuid4(),
                quote="RSI measures the speed and magnitude...",
                relevance_score=0.95,
                doc_title="Technical Analysis Guide",
            ),
            KBEvidenceItem(
                id=uuid4(),
                doc_id=uuid4(),
                chunk_id=uuid4(),
                quote="The Relative Strength Index...",
                relevance_score=0.88,
                doc_title="Trading Indicators",
            ),
        ]

        response = KBClaimDetailResponse(
            id=claim_id,
            claim_type=KBClaimType.DEFINITION,
            text="RSI measures momentum",
            status=KBClaimStatus.VERIFIED,
            confidence=0.92,
            entity_id=uuid4(),
            entity_name="RSI",
            entity_type=KBEntityType.INDICATOR,
            evidence=evidence,
            extraction_model="claude-haiku-3-5",
            verification_model="claude-haiku-3-5",
            created_at=datetime.now(),
        )

        assert len(response.evidence) == 2
        assert response.evidence[0].relevance_score == 0.95
        assert response.extraction_model == "claude-haiku-3-5"

    def test_evidence_item_structure(self):
        """Test KBEvidenceItem structure."""
        evidence = KBEvidenceItem(
            id=uuid4(),
            doc_id=uuid4(),
            chunk_id=uuid4(),
            quote="Some quote text",
            relevance_score=0.9,
            doc_title="Document Title",
        )

        assert evidence.relevance_score == 0.9
        assert evidence.doc_title == "Document Title"


class TestParseAliases:
    """Tests for _parse_aliases helper function."""

    def test_parse_json_string(self):
        """Test parsing JSON string aliases."""
        from app.routers.kb import _parse_aliases

        result = _parse_aliases('["alias1", "alias2"]')
        assert result == ["alias1", "alias2"]

    def test_parse_list(self):
        """Test parsing list aliases (already parsed)."""
        from app.routers.kb import _parse_aliases

        result = _parse_aliases(["alias1", "alias2"])
        assert result == ["alias1", "alias2"]

    def test_parse_none(self):
        """Test parsing None aliases."""
        from app.routers.kb import _parse_aliases

        result = _parse_aliases(None)
        assert result == []

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON string."""
        from app.routers.kb import _parse_aliases

        result = _parse_aliases("not valid json")
        assert result == []

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        from app.routers.kb import _parse_aliases

        result = _parse_aliases("")
        assert result == []


class TestKBRepositoryListEntities:
    """Tests for KB repository list_entities method."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        conn = AsyncMock()

        # Create async context manager mock
        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = conn
        async_cm.__aexit__.return_value = None

        pool = MagicMock()
        pool.acquire.return_value = async_cm

        return pool, conn

    @pytest.mark.asyncio
    async def test_list_entities_basic(self, mock_pool):
        """Test basic entity listing."""
        from app.repositories.kb import KnowledgeBaseRepository

        pool, conn = mock_pool
        workspace_id = uuid4()

        # Mock return values
        conn.fetchval.return_value = 10  # total count
        conn.fetch.return_value = [
            {
                "id": uuid4(),
                "type": "concept",
                "name": "Test Entity",
                "aliases": "[]",
                "description": None,
                "verified_claim_count": None,
                "created_at": datetime.now(),
            }
        ]

        repo = KnowledgeBaseRepository(pool)
        entities, total = await repo.list_entities(workspace_id=workspace_id)

        assert total == 10
        assert len(entities) == 1
        assert entities[0]["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_list_entities_with_search(self, mock_pool):
        """Test entity listing with search query."""
        from app.repositories.kb import KnowledgeBaseRepository

        pool, conn = mock_pool
        workspace_id = uuid4()

        conn.fetchval.return_value = 5
        conn.fetch.return_value = []

        repo = KnowledgeBaseRepository(pool)
        await repo.list_entities(workspace_id=workspace_id, q="RSI")

        # Verify search parameter was passed
        fetch_call = conn.fetch.call_args
        assert "%RSI%" in fetch_call[0]  # Check params contain search term


class TestKBRepositoryListClaims:
    """Tests for KB repository list_claims method."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        conn = AsyncMock()

        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = conn
        async_cm.__aexit__.return_value = None

        pool = MagicMock()
        pool.acquire.return_value = async_cm

        return pool, conn

    @pytest.mark.asyncio
    async def test_list_claims_default_verified(self, mock_pool):
        """Test that claims default to verified status filter."""
        from app.repositories.kb import KnowledgeBaseRepository

        pool, conn = mock_pool
        workspace_id = uuid4()

        conn.fetchval.return_value = 20
        conn.fetch.return_value = []

        repo = KnowledgeBaseRepository(pool)
        await repo.list_claims(workspace_id=workspace_id)

        # Verify status filter was applied
        fetch_call = conn.fetch.call_args
        assert "verified" in fetch_call[0]  # Check params contain verified status


class TestKBRepositorySearchClaimsForAnswer:
    """Tests for KB repository search_claims_for_answer method."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        conn = AsyncMock()

        async_cm = AsyncMock()
        async_cm.__aenter__.return_value = conn
        async_cm.__aexit__.return_value = None

        pool = MagicMock()
        pool.acquire.return_value = async_cm

        return pool, conn

    @pytest.mark.asyncio
    async def test_search_splits_query_words(self, mock_pool):
        """Test that search splits query into words."""
        from app.repositories.kb import KnowledgeBaseRepository

        pool, conn = mock_pool
        workspace_id = uuid4()

        conn.fetch.return_value = []

        repo = KnowledgeBaseRepository(pool)
        await repo.search_claims_for_answer(
            workspace_id=workspace_id,
            query_text="what is RSI indicator",
        )

        # Verify word-based search was used
        fetch_call = conn.fetch.call_args
        params = fetch_call[0]
        assert any("%what%" in str(p) for p in params) or any(
            "%rsi%" in str(p) for p in params
        )

    @pytest.mark.asyncio
    async def test_search_limits_words(self, mock_pool):
        """Test that search limits to first 5 words."""
        from app.repositories.kb import KnowledgeBaseRepository

        pool, conn = mock_pool
        workspace_id = uuid4()

        conn.fetch.return_value = []

        repo = KnowledgeBaseRepository(pool)
        await repo.search_claims_for_answer(
            workspace_id=workspace_id,
            query_text="one two three four five six seven eight",  # 8 words
        )

        # Should only use first 5 words
        fetch_call = conn.fetch.call_args
        params = fetch_call[0]
        # Count ILIKE patterns (one per word, max 5)
        like_params = [p for p in params if isinstance(p, str) and "%" in p]
        assert len(like_params) <= 5
