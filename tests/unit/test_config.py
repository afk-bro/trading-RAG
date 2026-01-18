"""Unit tests for app.config module."""

import os
from unittest.mock import patch


def test_data_config_defaults():
    """Test data layer config defaults."""
    # Clear cache before import to ensure fresh settings
    from app.config import get_settings

    get_settings.cache_clear()

    # Must set required env vars for Settings to instantiate
    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        },
    ):
        get_settings.cache_clear()
        settings = get_settings()

        # These should have defaults
        assert hasattr(settings, "data_dir")
        assert hasattr(settings, "ccxt_rate_limit_ms")
        assert hasattr(settings, "core_timeframes")


def test_ccxt_rate_limit_default():
    """Test CCXT rate limit default value."""
    from app.config import get_settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        },
    ):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.ccxt_rate_limit_ms == 100


def test_core_timeframes_default():
    """Test core timeframes default value."""
    from app.config import get_settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        },
    ):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.core_timeframes == ["1m", "5m", "15m", "1h", "1d"]


def test_job_poll_interval_default():
    """Test job poll interval default value."""
    from app.config import get_settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        },
    ):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.job_poll_interval_s == 1.0


def test_job_stale_timeout_default():
    """Test job stale timeout default value."""
    from app.config import get_settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        },
    ):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.job_stale_timeout_minutes == 30


def test_artifacts_dir_default():
    """Test artifacts directory default value."""
    from app.config import get_settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        },
    ):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.artifacts_dir == "/data/artifacts"


def test_artifacts_retention_days_default():
    """Test artifacts retention days default value."""
    from app.config import get_settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
        },
    ):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.artifacts_retention_days == 90


def test_ccxt_rate_limit_validation():
    """Test CCXT rate limit field validation bounds."""
    from app.config import get_settings
    from pydantic import ValidationError

    # Test with too low value
    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
            "CCXT_RATE_LIMIT_MS": "5",  # Below minimum of 10
        },
    ):
        get_settings.cache_clear()
        try:
            get_settings()
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            assert "ccxt_rate_limit_ms" in str(e).lower()


def test_job_poll_interval_validation():
    """Test job poll interval field validation bounds."""
    from app.config import get_settings
    from pydantic import ValidationError

    # Test with too high value
    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
            "JOB_POLL_INTERVAL_S": "15.0",  # Above maximum of 10
        },
    ):
        get_settings.cache_clear()
        try:
            get_settings()
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            assert "job_poll_interval_s" in str(e).lower()


def test_artifacts_retention_days_validation():
    """Test artifacts retention days field validation bounds."""
    from app.config import get_settings
    from pydantic import ValidationError

    # Test with too high value
    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_ROLE_KEY": "test-key",
            "ARTIFACTS_RETENTION_DAYS": "500",  # Above maximum of 365
        },
    ):
        get_settings.cache_clear()
        try:
            get_settings()
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            assert "artifacts_retention_days" in str(e).lower()
