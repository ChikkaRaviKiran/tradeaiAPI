import unittest
from unittest.mock import patch

from app.core.instruments import NIFTY, SENSEX
from app.engine.atl_straddle_scanner import ATLStraddleScanner


class StrikeRoundingTests(unittest.TestCase):
    def setUp(self):
        self.scanner = ATLStraddleScanner.__new__(ATLStraddleScanner)
        self.scanner._settings = {"strike_mode": "ATM"}

    def test_nifty_configured_rounding_changes_the_atm_strike(self):
        spot = 25548.35
        with patch.object(self.scanner, "_record_diag") as diag:
            for configured, expected in ((50, 25550), (100, 25500)):
                with self.subTest(configured=configured):
                    self.scanner._settings["strike_interval"] = configured
                    interval = self.scanner._effective_strike_interval(NIFTY)
                    self.assertEqual(interval, configured)
                    self.assertEqual(self.scanner._anchor_strike(spot, interval), expected)
            diag.assert_not_called()

    def test_incompatible_rounding_falls_back_to_the_exchange_grid(self):
        for instrument, configured, expected in (
            (NIFTY, 75, 50),
            (SENSEX, 50, 100),
            (NIFTY, 0, 50),
        ):
            with self.subTest(symbol=instrument.symbol, configured=configured):
                self.scanner._settings["strike_interval"] = configured
                with patch.object(self.scanner, "_record_diag") as diag:
                    self.assertEqual(self.scanner._effective_strike_interval(instrument), expected)
                    diag.assert_called_once_with(
                        "strike_interval_override",
                        f"Using {expected} (instrument) instead of configured {configured} for {instrument.symbol}",
                    )


if __name__ == "__main__":
    unittest.main()
