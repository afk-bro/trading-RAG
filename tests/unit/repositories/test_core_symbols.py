"""Tests for core symbols repository."""
from unittest.mock import MagicMock

from app.repositories.core_symbols import CoreSymbolsRepository, CoreSymbol


class TestCoreSymbol:
    def test_core_symbol_creation(self):
        cs = CoreSymbol(
            exchange_id="kucoin",
            canonical_symbol="BTC-USDT",
            raw_symbol="BTC-USDT",
        )
        assert cs.is_enabled is True
        assert cs.timeframes == ["1m", "5m", "15m", "1h", "1d"]


class TestCoreSymbolsRepository:
    def test_repository_creation(self):
        mock_pool = MagicMock()
        repo = CoreSymbolsRepository(mock_pool)
        assert repo._pool == mock_pool
