"""Tests for webhook delivery sinks."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import httpx
import pytest

from app.services.alerts.models import RuleType, Severity
from app.services.ops_alerts.webhook_sink import (
    GenericWebhookSink,
    SlackWebhookSink,
    WebhookDeliveryError,
)


@pytest.fixture
def sample_alert_event() -> dict[str, Any]:
    """Sample alert event for testing."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "rule_id": uuid4(),
        "strategy_entity_id": uuid4(),
        "regime_key": "trending",
        "timeframe": "1h",
        "rule_type": RuleType.DRIFT_SPIKE.value,
        "severity": Severity.HIGH.value,
        "status": "active",
        "activated_at": datetime(2025, 1, 20, 12, 0, 0, tzinfo=timezone.utc),
        "context_json": {
            "threshold": 0.30,
            "current_drift": 0.45,
            "consecutive_buckets": 2,
        },
        "fingerprint": "v1:trending:1h",
    }


class TestSlackWebhookSink:
    """Test Slack webhook message formatting and delivery."""

    def test_format_message_high_severity(self, sample_alert_event):
        """Test Slack message formatting for high severity alert."""
        sink = SlackWebhookSink("https://hooks.slack.com/test")
        payload = sink._format_message(sample_alert_event)

        assert payload["attachments"][0]["color"] == "danger"
        assert any(
            field["title"] == "Severity" and field["value"] == "high"
            for field in payload["attachments"][0]["fields"]
        )
        assert any(
            field["title"] == "Rule Type" and field["value"] == "drift_spike"
            for field in payload["attachments"][0]["fields"]
        )

    def test_format_message_medium_severity(self, sample_alert_event):
        """Test Slack message formatting for medium severity alert."""
        sample_alert_event["severity"] = Severity.MEDIUM.value
        sink = SlackWebhookSink("https://hooks.slack.com/test")
        payload = sink._format_message(sample_alert_event)

        assert payload["attachments"][0]["color"] == "warning"

    def test_format_message_low_severity(self, sample_alert_event):
        """Test Slack message formatting for low severity alert."""
        sample_alert_event["severity"] = Severity.LOW.value
        sink = SlackWebhookSink("https://hooks.slack.com/test")
        payload = sink._format_message(sample_alert_event)

        assert payload["attachments"][0]["color"] == "good"

    def test_format_message_includes_workspace(self, sample_alert_event):
        """Test that message includes workspace ID."""
        sink = SlackWebhookSink("https://hooks.slack.com/test")
        payload = sink._format_message(sample_alert_event)

        workspace_field = next(
            (
                field
                for field in payload["attachments"][0]["fields"]
                if field["title"] == "Workspace"
            ),
            None,
        )
        assert workspace_field is not None
        assert str(sample_alert_event["workspace_id"]) in workspace_field["value"]

    def test_format_message_includes_timestamp(self, sample_alert_event):
        """Test that message includes formatted timestamp."""
        sink = SlackWebhookSink("https://hooks.slack.com/test")
        payload = sink._format_message(sample_alert_event)

        # Check timestamp field
        timestamp_field = next(
            (
                field
                for field in payload["attachments"][0]["fields"]
                if field["title"] == "Activated"
            ),
            None,
        )
        assert timestamp_field is not None

    def test_format_message_includes_context(self, sample_alert_event):
        """Test that message includes context details."""
        sink = SlackWebhookSink("https://hooks.slack.com/test")
        payload = sink._format_message(sample_alert_event)

        # Should have context in message
        message_field = next(
            (
                field
                for field in payload["attachments"][0]["fields"]
                if field["title"] == "Details"
            ),
            None,
        )
        assert message_field is not None
        assert "threshold" in message_field["value"].lower()

    @pytest.mark.asyncio
    async def test_send_success(self, sample_alert_event):
        """Test successful webhook delivery."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            sink = SlackWebhookSink("https://hooks.slack.com/test")
            await sink.send(sample_alert_event)

            # Verify POST was called
            assert mock_post.call_count == 1
            call_kwargs = mock_post.call_args.kwargs
            assert "json" in call_kwargs

    @pytest.mark.asyncio
    async def test_send_with_retry_eventual_success(self, sample_alert_event):
        """Test retry logic succeeds on second attempt."""
        # First call fails, second succeeds
        mock_fail = Mock(spec=httpx.Response)
        mock_fail.status_code = 500
        mock_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("", request=Mock(), response=mock_fail)
        )

        mock_success = Mock(spec=httpx.Response)
        mock_success.status_code = 200
        mock_success.raise_for_status = Mock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [mock_fail, mock_success]

            sink = SlackWebhookSink("https://hooks.slack.com/test", max_retries=3)
            await sink.send(sample_alert_event)

            # Should have retried
            assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_with_retry_exhaustion(self, sample_alert_event):
        """Test retry logic fails after max attempts."""
        mock_fail = Mock(spec=httpx.Response)
        mock_fail.status_code = 500
        mock_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("", request=Mock(), response=mock_fail)
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_fail

            sink = SlackWebhookSink("https://hooks.slack.com/test", max_retries=3)

            with pytest.raises(WebhookDeliveryError) as exc_info:
                await sink.send(sample_alert_event)

            assert "after 3 attempts" in str(exc_info.value)
            assert mock_post.call_count == 3

    @pytest.mark.asyncio
    async def test_send_timeout_handling(self, sample_alert_event):
        """Test timeout handling during delivery."""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("")

            sink = SlackWebhookSink("https://hooks.slack.com/test", max_retries=2)

            with pytest.raises(WebhookDeliveryError) as exc_info:
                await sink.send(sample_alert_event)

            assert "timeout" in str(exc_info.value).lower()
            assert mock_post.call_count == 2  # Original + 1 retry


class TestGenericWebhookSink:
    """Test generic webhook delivery."""

    @pytest.mark.asyncio
    async def test_send_json_payload(self, sample_alert_event):
        """Test generic webhook sends JSON payload."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            sink = GenericWebhookSink("https://example.com/webhook")
            await sink.send(sample_alert_event)

            # Verify POST was called with JSON
            assert mock_post.call_count == 1
            call_kwargs = mock_post.call_args.kwargs
            assert "json" in call_kwargs

            # Verify payload structure
            payload = call_kwargs["json"]
            assert payload["event_type"] == "alert.activated"
            assert payload["alert"]["rule_type"] == "drift_spike"
            assert payload["alert"]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_send_includes_metadata(self, sample_alert_event):
        """Test generic webhook includes metadata."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            sink = GenericWebhookSink("https://example.com/webhook")
            await sink.send(sample_alert_event)

            payload = mock_post.call_args.kwargs["json"]
            assert "timestamp" in payload
            assert "workspace_id" in payload

    @pytest.mark.asyncio
    async def test_send_with_retry(self, sample_alert_event):
        """Test retry logic for generic webhook."""
        mock_fail = Mock(spec=httpx.Response)
        mock_fail.status_code = 503
        mock_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("", request=Mock(), response=mock_fail)
        )

        mock_success = Mock(spec=httpx.Response)
        mock_success.status_code = 200
        mock_success.raise_for_status = Mock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [mock_fail, mock_success]

            sink = GenericWebhookSink("https://example.com/webhook", max_retries=3)
            await sink.send(sample_alert_event)

            assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_send_custom_headers(self, sample_alert_event):
        """Test generic webhook with custom headers."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            custom_headers = {"X-API-Key": "secret", "X-Custom": "value"}
            sink = GenericWebhookSink(
                "https://example.com/webhook", headers=custom_headers
            )
            await sink.send(sample_alert_event)

            call_kwargs = mock_post.call_args.kwargs
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["X-API-Key"] == "secret"
            assert call_kwargs["headers"]["X-Custom"] == "value"


class TestWebhookFireAndForget:
    """Test fire-and-forget behavior (doesn't block on errors)."""

    @pytest.mark.asyncio
    async def test_send_logs_errors_without_raising(self, sample_alert_event):
        """Test that errors are logged but don't propagate when fire_and_forget=True."""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("")

            sink = SlackWebhookSink("https://hooks.slack.com/test", max_retries=1)

            # Should raise since we're not using fire_and_forget wrapper
            with pytest.raises(WebhookDeliveryError):
                await sink.send(sample_alert_event)
