"""Unit tests for app.utils.instruments."""

from app.utils.instruments import get_point_value, POINT_VALUE_NQ, POINT_VALUE_ES


class TestGetPointValue:
    """Tests for get_point_value()."""

    def test_mnq(self):
        assert get_point_value("MNQ") == POINT_VALUE_NQ / 10  # $2

    def test_nq(self):
        assert get_point_value("NQ") == POINT_VALUE_NQ  # $20

    def test_mes(self):
        assert get_point_value("MES") == POINT_VALUE_ES / 10  # $5

    def test_es(self):
        assert get_point_value("ES") == POINT_VALUE_ES  # $50

    def test_unknown_defaults_to_nq(self):
        assert get_point_value("AAPL") == POINT_VALUE_NQ

    def test_case_insensitive(self):
        assert get_point_value("mnq") == get_point_value("MNQ")
        assert get_point_value("Nq") == get_point_value("NQ")
        assert get_point_value("mes") == get_point_value("MES")
        assert get_point_value("es") == get_point_value("ES")

    def test_symbol_with_suffix(self):
        """Symbols like MNQ1!, NQ2412 should still match."""
        assert get_point_value("MNQ1!") == POINT_VALUE_NQ / 10
        assert get_point_value("NQ2412") == POINT_VALUE_NQ
        assert get_point_value("MES1!") == POINT_VALUE_ES / 10
        assert get_point_value("ESH25") == POINT_VALUE_ES

    def test_mnq_before_nq(self):
        """MNQ must match before NQ (substring ordering)."""
        # MNQ contains NQ, so MNQ check must come first
        assert get_point_value("MNQ") == 2.0
        assert get_point_value("NQ") == 20.0

    def test_mes_before_es(self):
        """MES must match before ES (substring ordering)."""
        assert get_point_value("MES") == 5.0
        assert get_point_value("ES") == 50.0
