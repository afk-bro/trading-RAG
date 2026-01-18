"""
Unit tests for Pine Script ingestion service.

Tests content formatting and ingest logic without database/vector dependencies.
"""

import pytest

from app.services.pine.ingest import (
    extract_symbols_from_script,
    format_script_content,
    PineIngestResult,
    ScriptIngestResult,
)
from app.services.pine.models import (
    InputType,
    LintSummary,
    PineImport,
    PineInput,
    PineScriptEntry,
    PineVersion,
    ScriptType,
)


class TestFormatScriptContent:
    """Tests for format_script_content."""

    def test_format_basic_indicator(self):
        """Formats a basic indicator script."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test Indicator",
        )

        content = format_script_content(
            entry, source_content=None, include_source=False
        )

        assert "# Test Indicator" in content
        assert "**Type**: indicator" in content
        assert "**Pine Version**: 5" in content
        assert "**File**: test.pine" in content

    def test_format_with_inputs(self):
        """Includes input parameters in formatted content."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
            inputs=[
                PineInput(
                    name="Length",
                    type=InputType.INT,
                    default=14,
                    line=3,
                ),
                PineInput(
                    name="Source",
                    type=InputType.SOURCE,
                    default_expr="close",
                    tooltip="Price source to use",
                    line=4,
                ),
            ],
        )

        content = format_script_content(entry, include_source=False)

        assert "## Inputs" in content
        assert "**Length** (int)" in content
        assert "(default: 14)" in content
        assert "**Source** (source)" in content
        assert "(default: close)" in content
        assert "Price source to use" in content

    def test_format_with_imports(self):
        """Includes library imports in formatted content."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
            imports=[
                PineImport(path="TradingView/ta", alias="ta"),
                PineImport(path="MyLib/utils", alias="utils"),
            ],
        )

        content = format_script_content(entry, include_source=False)

        assert "## Imports" in content
        assert "TradingView/ta" in content
        assert "as ta" in content
        assert "MyLib/utils" in content

    def test_format_with_features(self):
        """Includes detected features in formatted content."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.STRATEGY,
            title="Test Strategy",
            features={
                "uses_request_security": True,
                "uses_arrays": True,
                "uses_alert": False,
                "is_library": False,
            },
        )

        content = format_script_content(entry, include_source=False)

        assert "## Features" in content
        assert "Uses arrays" in content
        assert "Uses request security" in content
        assert "Uses alert" not in content  # False features excluded

    def test_format_with_lint_summary(self):
        """Includes lint summary when present."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
            lint=LintSummary(error_count=2, warning_count=1, info_count=0),
        )

        content = format_script_content(entry, include_source=False)

        assert "## Lint Summary" in content
        assert "Errors: 2" in content
        assert "Warnings: 1" in content
        assert "Info:" not in content  # 0 count excluded

    def test_format_with_source_code(self):
        """Includes source code when provided."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
        )

        source = "//@version=5\nindicator('Test')\nplot(close)"
        content = format_script_content(
            entry, source_content=source, include_source=True
        )

        assert "## Source Code" in content
        assert "```pine" in content
        assert "//@version=5" in content
        assert "plot(close)" in content
        assert "```" in content

    def test_format_truncates_long_source(self):
        """Truncates source code exceeding max lines."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title="Test",
        )

        # Create source with 150 lines
        source = "\n".join([f"line_{i}" for i in range(150)])
        content = format_script_content(
            entry, source_content=source, include_source=True, max_source_lines=50
        )

        assert "line_0" in content
        assert "line_49" in content
        assert "line_50" not in content
        assert "(100 more lines)" in content

    def test_format_uses_rel_path_as_title_fallback(self):
        """Uses rel_path as title when title is None."""
        entry = PineScriptEntry(
            rel_path="scripts/my_indicator.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
            title=None,
        )

        content = format_script_content(entry, include_source=False)

        assert "# scripts/my_indicator.pine" in content


class TestExtractSymbolsFromScript:
    """Tests for extract_symbols_from_script."""

    def test_extract_ticker_strings(self):
        """Extracts ticker symbols from quoted strings."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
//@version=5
indicator("Test")
spy = request.security("SPY", "D", close)
aapl = request.security("AAPL", "D", close)
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "SPY" in symbols
        assert "AAPL" in symbols

    def test_extract_index_symbols(self):
        """Extracts common index symbols."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
//@version=5
indicator("Market Dashboard")
// Compare SPX, QQQ, and VIX
plot(close)
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "SPX" in symbols
        assert "QQQ" in symbols
        assert "VIX" in symbols

    def test_extract_crypto_pairs(self):
        """Extracts cryptocurrency pairs."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
btc = request.security("BTCUSD", "D", close)
eth = request.security("ETHUSDT", "D", close)
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "BTCUSD" in symbols
        assert "ETHUSDT" in symbols

    def test_filters_common_non_tickers(self):
        """Filters out common non-ticker strings."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        content = """
if condition == "TRUE"
    label.set_text(lbl, "NA")
"""

        symbols = extract_symbols_from_script(entry, content)

        assert "TRUE" not in symbols
        assert "NA" not in symbols

    def test_limits_symbol_count(self):
        """Limits extracted symbols to 10."""
        entry = PineScriptEntry(
            rel_path="test.pine",
            sha256="abc123",
            pine_version=PineVersion.V5,
            script_type=ScriptType.INDICATOR,
        )

        # Create content with many tickers
        tickers = " ".join([f'"TICK{i}"' for i in range(20)])
        content = f"//@version=5\n{tickers}"

        symbols = extract_symbols_from_script(entry, content)

        assert len(symbols) <= 10


class TestPineIngestResultProperties:
    """Tests for PineIngestResult properties."""

    def test_empty_result(self):
        """Empty result has default values."""
        result = PineIngestResult()

        assert result.scripts_processed == 0
        assert result.scripts_indexed == 0
        assert result.scripts_skipped == 0
        assert result.scripts_failed == 0
        assert result.total_chunks == 0
        assert result.success is True  # No failures = success

    def test_success_false_when_failures(self):
        """success is False when there are failures."""
        result = PineIngestResult(
            scripts_processed=3,
            scripts_indexed=2,
            scripts_failed=1,
        )

        assert result.success is False

    def test_success_true_when_no_failures(self):
        """success is True when no failures."""
        result = PineIngestResult(
            scripts_processed=3,
            scripts_indexed=2,
            scripts_skipped=1,
            scripts_failed=0,
        )

        assert result.success is True


class TestScriptIngestResult:
    """Tests for ScriptIngestResult."""

    def test_default_values(self):
        """Default values are set correctly."""
        result = ScriptIngestResult(rel_path="test.pine", success=True)

        assert result.rel_path == "test.pine"
        assert result.success is True
        assert result.doc_id is None
        assert result.chunks_created == 0
        assert result.status == "pending"
        assert result.error is None

    def test_failed_result(self):
        """Failed result captures error."""
        result = ScriptIngestResult(
            rel_path="bad.pine",
            success=False,
            status="failed",
            error="Parse error",
        )

        assert result.success is False
        assert result.status == "failed"
        assert result.error == "Parse error"


class TestIngestPipelineVectorCleanup:
    """
    Tests for ingest_pipeline vector cleanup on script edit.

    Verifies that when a script is re-ingested with update_existing=True,
    old vectors are deleted before new ones are created.
    """

    @pytest.mark.asyncio
    async def test_edit_script_deletes_old_vectors(self):
        """
        When a script is edited and re-ingested, old vectors are cleaned up.

        This is the critical test for preventing stale vector accumulation.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from uuid import uuid4

        from app.schemas import SourceType

        # Setup: IDs for tracking
        workspace_id = uuid4()
        old_doc_id = uuid4()
        new_doc_id = uuid4()
        old_chunk_ids = [str(uuid4()), str(uuid4())]  # 2 chunks from first ingest
        new_chunk_ids = [
            str(uuid4()),
            str(uuid4()),
            str(uuid4()),
        ]  # 3 chunks after edit

        # Mock repositories
        mock_doc_repo = MagicMock()
        mock_chunk_repo = MagicMock()
        mock_chunk_vector_repo = MagicMock()
        mock_vector_repo = MagicMock()

        # First call: doc exists (simulating re-ingest)
        mock_doc_repo.get_by_canonical_url = AsyncMock(
            return_value={
                "id": old_doc_id,
                "version": 1,
                "canonical_url": "pine://local/test.pine",
            }
        )
        mock_doc_repo.get_chunk_ids = AsyncMock(return_value=old_chunk_ids)
        mock_doc_repo.supersede_and_create = AsyncMock(return_value=(new_doc_id, 2))
        mock_doc_repo.get_by_content_hash = AsyncMock(return_value=None)
        mock_doc_repo.update_last_indexed = AsyncMock(return_value=None)

        mock_chunk_repo.create_batch = AsyncMock(return_value=new_chunk_ids)
        mock_chunk_vector_repo.create_batch = AsyncMock(return_value=None)

        mock_vector_repo.upsert_batch = AsyncMock(return_value=None)
        mock_vector_repo.delete_batch = AsyncMock(return_value=None)

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed_batch = AsyncMock(
            return_value=[[0.1] * 768, [0.2] * 768, [0.3] * 768]  # 3 embeddings
        )

        # Mock extractor
        mock_extractor = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.symbols = []
        mock_metadata.entities = []
        mock_metadata.topics = []
        mock_metadata.quality_score = 0.8
        mock_metadata.speaker = None
        mock_extractor.extract = MagicMock(return_value=mock_metadata)

        # Mock health validation result
        from app.services.ingest.health import HealthResult, HealthStatus

        mock_health_result = HealthResult(
            source_id=new_doc_id,
            workspace_id=workspace_id,
            status=HealthStatus.OK,
            checks=[],
        )

        # Patch the repository modules (imports happen inside ingest_pipeline)
        with patch(
            "app.repositories.documents.DocumentRepository", return_value=mock_doc_repo
        ), patch(
            "app.repositories.chunks.ChunkRepository", return_value=mock_chunk_repo
        ), patch(
            "app.repositories.vectors.ChunkVectorRepository",
            return_value=mock_chunk_vector_repo,
        ), patch(
            "app.repositories.vectors.VectorRepository", return_value=mock_vector_repo
        ), patch(
            "app.routers.ingest.get_embedder", return_value=mock_embedder
        ), patch(
            "app.routers.ingest.get_extractor", return_value=mock_extractor
        ), patch(
            "app.services.ingest.validate_source_health",
            new=AsyncMock(return_value=mock_health_result),
        ), patch(
            "app.routers.ingest._db_pool", MagicMock()
        ), patch(
            "app.routers.ingest._qdrant_client", MagicMock()
        ):
            from app.routers.ingest import ingest_pipeline

            # Call ingest with update_existing=True (simulating script edit)
            response = await ingest_pipeline(
                workspace_id=workspace_id,
                content="// Edited script content\nplot(close)",
                source_type=SourceType.PINE_SCRIPT,
                source_url=None,
                canonical_url="pine://local/test.pine",
                idempotency_key=None,
                content_hash="newhash123",
                title="Test Script",
                update_existing=True,
            )

        # Assertions: verify the cleanup contract

        # 1. Old chunk IDs were retrieved
        mock_doc_repo.get_chunk_ids.assert_called_once_with(old_doc_id)

        # 2. Old document was superseded
        mock_doc_repo.supersede_and_create.assert_called_once()

        # 3. New chunks were created
        mock_chunk_repo.create_batch.assert_called_once()

        # 4. CRITICAL: Old vectors were deleted
        mock_vector_repo.delete_batch.assert_called_once_with(old_chunk_ids)

        # 5. New vectors were created
        mock_vector_repo.upsert_batch.assert_called_once()

        # 6. Response reflects the update
        assert response.doc_id == new_doc_id
        assert response.superseded_doc_id == old_doc_id
        assert response.version == 2
        assert response.chunks_created == 3

    @pytest.mark.asyncio
    async def test_first_ingest_no_delete_called(self):
        """
        First ingest (no existing doc) should not call delete_batch.
        """
        from unittest.mock import AsyncMock, MagicMock, patch
        from uuid import uuid4

        from app.schemas import SourceType

        workspace_id = uuid4()
        new_doc_id = uuid4()
        chunk_ids = [str(uuid4())]

        mock_doc_repo = MagicMock()
        mock_chunk_repo = MagicMock()
        mock_chunk_vector_repo = MagicMock()
        mock_vector_repo = MagicMock()

        # No existing doc
        mock_doc_repo.get_by_canonical_url = AsyncMock(return_value=None)
        mock_doc_repo.get_by_content_hash = AsyncMock(return_value=None)
        mock_doc_repo.create = AsyncMock(return_value=new_doc_id)
        mock_doc_repo.update_last_indexed = AsyncMock(return_value=None)

        mock_chunk_repo.create_batch = AsyncMock(return_value=chunk_ids)
        mock_chunk_vector_repo.create_batch = AsyncMock(return_value=None)
        mock_vector_repo.upsert_batch = AsyncMock(return_value=None)
        mock_vector_repo.delete_batch = AsyncMock(return_value=None)

        mock_embedder = MagicMock()
        mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 768])

        mock_extractor = MagicMock()
        mock_metadata = MagicMock()
        mock_metadata.symbols = []
        mock_metadata.entities = []
        mock_metadata.topics = []
        mock_metadata.quality_score = 0.8
        mock_metadata.speaker = None
        mock_extractor.extract = MagicMock(return_value=mock_metadata)

        # Mock health validation result
        from app.services.ingest.health import HealthResult, HealthStatus

        mock_health_result = HealthResult(
            source_id=new_doc_id,
            workspace_id=workspace_id,
            status=HealthStatus.OK,
            checks=[],
        )

        with patch(
            "app.repositories.documents.DocumentRepository", return_value=mock_doc_repo
        ), patch(
            "app.repositories.chunks.ChunkRepository", return_value=mock_chunk_repo
        ), patch(
            "app.repositories.vectors.ChunkVectorRepository",
            return_value=mock_chunk_vector_repo,
        ), patch(
            "app.repositories.vectors.VectorRepository", return_value=mock_vector_repo
        ), patch(
            "app.routers.ingest.get_embedder", return_value=mock_embedder
        ), patch(
            "app.routers.ingest.get_extractor", return_value=mock_extractor
        ), patch(
            "app.services.ingest.validate_source_health",
            new=AsyncMock(return_value=mock_health_result),
        ), patch(
            "app.routers.ingest._db_pool", MagicMock()
        ), patch(
            "app.routers.ingest._qdrant_client", MagicMock()
        ):
            from app.routers.ingest import ingest_pipeline

            await ingest_pipeline(
                workspace_id=workspace_id,
                content="// New script",
                source_type=SourceType.PINE_SCRIPT,
                source_url=None,
                canonical_url="pine://local/new.pine",
                idempotency_key=None,
                content_hash="hash123",
                title="New Script",
                update_existing=False,
            )

        # delete_batch should NOT be called for first ingest
        mock_vector_repo.delete_batch.assert_not_called()
