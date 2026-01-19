"""Tests for admin data endpoints."""


class TestCoreSymbolsEndpoints:
    def test_list_core_symbols_schema(self):
        """Test that list endpoint returns expected schema."""
        from app.admin.data import CoreSymbolResponse

        response = CoreSymbolResponse(
            exchange_id="kucoin",
            canonical_symbol="BTC-USDT",
            raw_symbol="BTC-USDT",
            timeframes=["1h", "1d"],
            is_enabled=True,
        )
        assert response.exchange_id == "kucoin"

    def test_add_core_symbol_request(self):
        """Test add symbol request model."""
        from app.admin.data import AddCoreSymbolsRequest

        req = AddCoreSymbolsRequest(
            exchange_id="kucoin",
            symbols=["BTC-USDT", "ETH-USDT"],
        )
        assert len(req.symbols) == 2
