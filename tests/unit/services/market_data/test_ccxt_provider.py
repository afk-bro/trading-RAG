"""Tests for CCXT market data provider."""

from app.services.market_data.ccxt_provider import CcxtMarketDataProvider


class TestCcxtProvider:
    """Unit tests for CcxtMarketDataProvider."""

    def test_exchange_id(self):
        """Test exchange_id property returns the configured exchange."""
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.exchange_id == "kucoin"

    def test_normalize_symbol(self):
        """Test converting exchange symbol to canonical format."""
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.normalize_symbol("BTC/USDT") == "BTC-USDT"
        assert provider.normalize_symbol("ETH/USDT") == "ETH-USDT"

    def test_exchange_symbol(self):
        """Test converting canonical symbol to exchange format."""
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.exchange_symbol("BTC-USDT") == "BTC/USDT"
        assert provider.exchange_symbol("ETH-USDT") == "ETH/USDT"

    def test_canonical_timeframe(self):
        """Test timeframe normalization to canonical format."""
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.canonical_timeframe("1m") == "1m"
        assert provider.canonical_timeframe("1h") == "1h"

    def test_ccxt_timeframe(self):
        """Test converting canonical timeframe to CCXT format."""
        provider = CcxtMarketDataProvider("kucoin")
        assert provider.ccxt_timeframe("1m") == "1m"
        assert provider.ccxt_timeframe("1h") == "1h"
        assert provider.ccxt_timeframe("1d") == "1d"

    def test_exchange_id_different_exchange(self):
        """Test exchange_id with different exchanges."""
        provider = CcxtMarketDataProvider("binance")
        assert provider.exchange_id == "binance"

    def test_normalize_symbol_edge_cases(self):
        """Test symbol normalization with various formats."""
        provider = CcxtMarketDataProvider("kucoin")
        # Single slash
        assert provider.normalize_symbol("SOL/USDT") == "SOL-USDT"
        # Already canonical (should still work, just no-op on slash)
        assert provider.normalize_symbol("BTC-USDT") == "BTC-USDT"

    def test_exchange_symbol_edge_cases(self):
        """Test exchange symbol conversion with various formats."""
        provider = CcxtMarketDataProvider("kucoin")
        # Single dash
        assert provider.exchange_symbol("SOL-USDT") == "SOL/USDT"
        # Already exchange format (should still work)
        assert provider.exchange_symbol("BTC/USDT") == "BTC/USDT"

    def test_canonical_timeframe_with_aliases(self):
        """Test timeframe normalization handles aliases."""
        provider = CcxtMarketDataProvider("kucoin")
        # These should normalize via base.normalize_timeframe
        assert provider.canonical_timeframe("1min") == "1m"
        assert provider.canonical_timeframe("1hour") == "1h"
        assert provider.canonical_timeframe("1day") == "1d"

    def test_custom_rate_limit(self):
        """Test provider accepts custom rate limit."""
        provider = CcxtMarketDataProvider("kucoin", rate_limit_ms=200)
        # Access via property which checks the override
        assert provider._rate_limit_ms == 200
        # Also verify the override is stored
        assert provider._rate_limit_ms_override == 200
