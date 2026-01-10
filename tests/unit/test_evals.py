"""Unit tests for eval repository helpers."""

from app.repositories.evals import (
    compute_question_hash,
    compute_config_fingerprint,
)


class TestQuestionHash:
    """Tests for question hashing."""

    def test_hash_is_deterministic(self):
        """Same question produces same hash."""
        q = "What causes inflation?"
        assert compute_question_hash(q) == compute_question_hash(q)

    def test_hash_is_normalized(self):
        """Whitespace and case are normalized."""
        q1 = "What causes inflation?"
        q2 = "  what causes inflation?  "
        q3 = "WHAT CAUSES INFLATION?"

        assert compute_question_hash(q1) == compute_question_hash(q2)
        assert compute_question_hash(q1) == compute_question_hash(q3)

    def test_different_questions_different_hashes(self):
        """Different questions produce different hashes."""
        q1 = "What causes inflation?"
        q2 = "What causes deflation?"

        assert compute_question_hash(q1) != compute_question_hash(q2)

    def test_hash_length(self):
        """Hash is truncated to 32 chars."""
        h = compute_question_hash("Any question")
        assert len(h) == 32


class TestConfigFingerprint:
    """Tests for config fingerprinting."""

    def test_fingerprint_is_deterministic(self):
        """Same config produces same fingerprint."""
        fp1 = compute_config_fingerprint(50, 10, "cross_encoder", "bge-v2", True)
        fp2 = compute_config_fingerprint(50, 10, "cross_encoder", "bge-v2", True)
        assert fp1 == fp2

    def test_different_configs_different_fingerprints(self):
        """Different configs produce different fingerprints."""
        fp1 = compute_config_fingerprint(50, 10, "cross_encoder", "bge-v2", True)
        fp2 = compute_config_fingerprint(30, 10, "cross_encoder", "bge-v2", True)
        fp3 = compute_config_fingerprint(50, 5, "cross_encoder", "bge-v2", True)
        fp4 = compute_config_fingerprint(50, 10, "llm", "gpt-4", True)
        fp5 = compute_config_fingerprint(50, 10, "cross_encoder", "bge-v2", False)

        fingerprints = {fp1, fp2, fp3, fp4, fp5}
        assert len(fingerprints) == 5  # All unique

    def test_fingerprint_length(self):
        """Fingerprint is truncated to 16 chars."""
        fp = compute_config_fingerprint(50, 10, "cross_encoder", "bge-v2", True)
        assert len(fp) == 16

    def test_none_values_handled(self):
        """None values don't break fingerprinting."""
        fp = compute_config_fingerprint(50, 10, None, None, True)
        assert len(fp) == 16
