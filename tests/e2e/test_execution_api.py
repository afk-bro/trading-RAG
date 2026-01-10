"""
E2E tests for the Execution API endpoints (/execute/*).

Tests paper trading state, positions, reconciliation, and intent execution.

Run with:
    ADMIN_TOKEN=e2e-test-token pytest tests/e2e/test_execution_api.py --base-url http://localhost:8000
"""

import uuid
import pytest
from playwright.sync_api import APIRequestContext


pytestmark = pytest.mark.e2e


class TestPaperStateEndpoint:
    """E2E tests for GET /execute/paper/state/{workspace_id}."""

    def test_state_endpoint_returns_200(self, api_request: APIRequestContext):
        """State endpoint returns 200 for any valid UUID."""
        workspace_id = str(uuid.uuid4())
        response = api_request.get(f"/execute/paper/state/{workspace_id}")

        assert response.status == 200
        data = response.json()
        assert "cash" in data
        assert "starting_equity" in data
        assert "positions" in data

    def test_state_returns_default_equity(self, api_request: APIRequestContext):
        """Fresh state has default starting equity."""
        workspace_id = str(uuid.uuid4())
        response = api_request.get(f"/execute/paper/state/{workspace_id}")

        assert response.status == 200
        data = response.json()
        assert data["starting_equity"] == 10000.0
        assert data["cash"] == 10000.0
        assert data["realized_pnl"] == 0.0

    def test_state_requires_auth(self, api_request_no_auth: APIRequestContext):
        """State endpoint requires admin token."""
        workspace_id = str(uuid.uuid4())
        response = api_request_no_auth.get(f"/execute/paper/state/{workspace_id}")

        assert response.status in (401, 403)

    def test_state_validates_uuid_format(self, api_request: APIRequestContext):
        """State endpoint validates UUID format."""
        response = api_request.get("/execute/paper/state/not-a-uuid")

        assert response.status == 422


class TestPaperPositionsEndpoint:
    """E2E tests for GET /execute/paper/positions/{workspace_id}."""

    def test_positions_returns_200(self, api_request: APIRequestContext):
        """Positions endpoint returns 200."""
        workspace_id = str(uuid.uuid4())
        response = api_request.get(f"/execute/paper/positions/{workspace_id}")

        assert response.status == 200
        data = response.json()
        assert isinstance(data, list)

    def test_positions_empty_for_new_workspace(self, api_request: APIRequestContext):
        """Fresh workspace has no positions."""
        workspace_id = str(uuid.uuid4())
        response = api_request.get(f"/execute/paper/positions/{workspace_id}")

        assert response.status == 200
        data = response.json()
        assert len(data) == 0

    def test_positions_requires_auth(self, api_request_no_auth: APIRequestContext):
        """Positions endpoint requires admin token."""
        workspace_id = str(uuid.uuid4())
        response = api_request_no_auth.get(f"/execute/paper/positions/{workspace_id}")

        assert response.status in (401, 403)


class TestPaperReconcileEndpoint:
    """E2E tests for POST /execute/paper/reconcile/{workspace_id}."""

    def test_reconcile_returns_200(self, api_request: APIRequestContext):
        """Reconcile endpoint returns 200."""
        workspace_id = str(uuid.uuid4())
        response = api_request.post(f"/execute/paper/reconcile/{workspace_id}")

        assert response.status == 200
        data = response.json()
        assert data["success"] is True
        assert "events_replayed" in data
        assert "positions_rebuilt" in data

    def test_reconcile_empty_journal(self, api_request: APIRequestContext):
        """Reconcile on fresh workspace returns zero counts."""
        workspace_id = str(uuid.uuid4())
        response = api_request.post(f"/execute/paper/reconcile/{workspace_id}")

        assert response.status == 200
        data = response.json()
        assert data["events_replayed"] == 0
        assert data["positions_rebuilt"] == 0
        assert data["errors"] == []

    def test_reconcile_requires_auth(self, api_request_no_auth: APIRequestContext):
        """Reconcile endpoint requires admin token."""
        workspace_id = str(uuid.uuid4())
        response = api_request_no_auth.post(f"/execute/paper/reconcile/{workspace_id}")

        assert response.status in (401, 403)


class TestPaperResetEndpoint:
    """E2E tests for POST /execute/paper/reset/{workspace_id}."""

    def test_reset_returns_200_in_dev(self, api_request: APIRequestContext):
        """Reset endpoint returns 200 in development mode."""
        workspace_id = str(uuid.uuid4())
        response = api_request.post(f"/execute/paper/reset/{workspace_id}")

        # May return 200 (dev) or 403 (production)
        assert response.status in (200, 403)

        if response.status == 200:
            data = response.json()
            assert data["status"] == "reset"
            assert data["workspace_id"] == workspace_id

    def test_reset_requires_auth(self, api_request_no_auth: APIRequestContext):
        """Reset endpoint requires admin token."""
        workspace_id = str(uuid.uuid4())
        response = api_request_no_auth.post(f"/execute/paper/reset/{workspace_id}")

        assert response.status in (401, 403)


class TestExecuteIntentsEndpoint:
    """E2E tests for POST /execute/intents."""

    @pytest.fixture
    def valid_intent_payload(self):
        """Create a valid execution request payload."""
        return {
            "intent": {
                "id": str(uuid.uuid4()),
                "workspace_id": str(uuid.uuid4()),
                "correlation_id": f"e2e-test-{uuid.uuid4().hex[:8]}",
                "action": "open_long",
                "strategy_entity_id": str(uuid.uuid4()),
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "quantity": 0.1,
                "reason": "E2E test intent",
            },
            "fill_price": 50000.0,
            "mode": "paper",
        }

    def test_execute_requires_auth(
        self, api_request_no_auth: APIRequestContext, valid_intent_payload
    ):
        """Execute endpoint requires admin token."""
        response = api_request_no_auth.post(
            "/execute/intents",
            data=valid_intent_payload,
        )

        assert response.status in (401, 403)

    def test_execute_validates_mode(
        self, api_request: APIRequestContext, valid_intent_payload
    ):
        """Execute rejects live mode in PR1."""
        valid_intent_payload["mode"] = "live"

        response = api_request.post(
            "/execute/intents",
            data=valid_intent_payload,
        )

        assert response.status == 400
        data = response.json()
        assert data["detail"]["error_code"] == "UNSUPPORTED_MODE"

    def test_execute_validates_action(
        self, api_request: APIRequestContext, valid_intent_payload
    ):
        """Execute rejects unsupported actions."""
        valid_intent_payload["intent"]["action"] = "open_short"

        response = api_request.post(
            "/execute/intents",
            data=valid_intent_payload,
        )

        assert response.status == 400
        data = response.json()
        assert data["detail"]["error_code"] == "UNSUPPORTED_ACTION"

    def test_execute_validates_fill_price(
        self, api_request: APIRequestContext, valid_intent_payload
    ):
        """Execute rejects zero/negative fill price."""
        valid_intent_payload["fill_price"] = 0

        response = api_request.post(
            "/execute/intents",
            data=valid_intent_payload,
        )

        # Schema validation (Pydantic gt=0) returns 422, business logic returns 400
        assert response.status in (400, 422)

    def test_execute_successful_open_long(
        self, api_request: APIRequestContext, valid_intent_payload
    ):
        """Execute successfully opens a long position."""
        response = api_request.post(
            "/execute/intents",
            data=valid_intent_payload,
        )

        # May succeed (200) or fail policy check (200 with success=false)
        assert response.status == 200
        data = response.json()

        if data["success"]:
            assert data["order_id"] is not None
            assert data["fill_price"] == 50000.0
            assert data["position_action"] == "opened"
        else:
            # Policy rejected - that's ok for E2E
            assert data["error_code"] == "POLICY_REJECTED"

    def test_execute_idempotent(
        self, api_request: APIRequestContext, valid_intent_payload
    ):
        """Duplicate intent execution returns 409."""
        # First execution
        response1 = api_request.post(
            "/execute/intents",
            data=valid_intent_payload,
        )

        # Skip if first execution failed (policy rejection)
        if response1.status != 200 or not response1.json().get("success"):
            pytest.skip("First execution failed, cannot test idempotency")

        # Second execution with same intent_id
        response2 = api_request.post(
            "/execute/intents",
            data=valid_intent_payload,
        )

        assert response2.status == 409
        data = response2.json()
        assert data["detail"]["error_code"] == "ALREADY_EXECUTED"


class TestExecutionWorkflow:
    """E2E tests for complete execution workflows."""

    def test_open_and_close_position(self, api_request: APIRequestContext):
        """Test opening and closing a position."""
        workspace_id = str(uuid.uuid4())
        strategy_id = str(uuid.uuid4())
        correlation_id = f"e2e-workflow-{uuid.uuid4().hex[:8]}"

        # Open position
        open_payload = {
            "intent": {
                "id": str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "correlation_id": correlation_id,
                "action": "open_long",
                "strategy_entity_id": strategy_id,
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "quantity": 1.0,
                "reason": "E2E workflow test - open",
            },
            "fill_price": 3000.0,
            "mode": "paper",
        }

        open_response = api_request.post("/execute/intents", data=open_payload)

        # Skip if policy rejected
        if open_response.status != 200:
            pytest.skip("Open position request failed")
        open_data = open_response.json()
        if not open_data.get("success"):
            pytest.skip("Open position rejected by policy")

        assert open_data["position_action"] == "opened"

        # Check position exists
        positions_response = api_request.get(f"/execute/paper/positions/{workspace_id}")
        assert positions_response.status == 200
        positions = positions_response.json()
        assert len(positions) == 1
        assert positions[0]["symbol"] == "ETH/USDT"
        assert positions[0]["quantity"] == 1.0

        # Close position
        close_payload = {
            "intent": {
                "id": str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "correlation_id": correlation_id,
                "action": "close_long",
                "strategy_entity_id": strategy_id,
                "symbol": "ETH/USDT",
                "timeframe": "1h",
                "quantity": 1.0,  # Must match position qty
                "reason": "E2E workflow test - close",
            },
            "fill_price": 3100.0,  # Profit
            "mode": "paper",
        }

        close_response = api_request.post("/execute/intents", data=close_payload)

        if close_response.status != 200:
            pytest.skip("Close position request failed")
        close_data = close_response.json()
        if not close_data.get("success"):
            pytest.skip("Close position rejected by policy")

        assert close_data["position_action"] == "closed"

        # Verify position closed
        final_positions = api_request.get(f"/execute/paper/positions/{workspace_id}")
        assert final_positions.status == 200
        assert len(final_positions.json()) == 0

        # Verify P&L in state
        state_response = api_request.get(f"/execute/paper/state/{workspace_id}")
        assert state_response.status == 200
        state = state_response.json()
        assert state["realized_pnl"] == 100.0  # (3100 - 3000) * 1.0

    def test_reconcile_after_trades(self, api_request: APIRequestContext):
        """Test that reconciliation rebuilds state correctly."""
        workspace_id = str(uuid.uuid4())

        # Execute a trade
        open_payload = {
            "intent": {
                "id": str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "correlation_id": f"e2e-reconcile-{uuid.uuid4().hex[:8]}",
                "action": "open_long",
                "strategy_entity_id": str(uuid.uuid4()),
                "symbol": "SOL/USDT",
                "timeframe": "1h",
                "quantity": 10.0,
                "reason": "E2E reconcile test",
            },
            "fill_price": 100.0,
            "mode": "paper",
        }

        response = api_request.post("/execute/intents", data=open_payload)

        if response.status != 200 or not response.json().get("success"):
            pytest.skip("Trade execution failed")

        # Get state before reconcile
        state_before = api_request.get(f"/execute/paper/state/{workspace_id}").json()

        # Reconcile
        reconcile_response = api_request.post(
            f"/execute/paper/reconcile/{workspace_id}"
        )
        assert reconcile_response.status == 200
        reconcile_data = reconcile_response.json()
        assert reconcile_data["success"] is True
        assert reconcile_data["events_replayed"] >= 1

        # Get state after reconcile
        state_after = api_request.get(f"/execute/paper/state/{workspace_id}").json()

        # State should be consistent
        assert state_after["cash"] == state_before["cash"]
        assert len(state_after["positions"]) == len(state_before["positions"])
