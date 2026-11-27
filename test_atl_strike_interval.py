import unittest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from app.core.instruments import NIFTY, SENSEX
from app.engine.atl_straddle_scanner import ATLLeg, ATLStraddleScanner, IST


class ATLStrikeIntervalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = IST.localize(datetime(2026, 11, 27, 11, 5))
        clock = patch("app.engine.atl_straddle_scanner.datetime", wraps=datetime)
        self.clock = clock.start()
        self.clock.now.return_value = self.now
        self.addCleanup(clock.stop)
        for method in ("_load_state_from_disk", "_save_state_to_disk"):
            persistence = patch.object(ATLStraddleScanner, method)
            persistence.start()
            self.addCleanup(persistence.stop)

    def scanner(self, interval, instrument=NIFTY, **overrides):
        config = {
            "enabled": True,
            "index": instrument.symbol,
            "strike_interval": interval,
            "strategy_type": "ATM_STRADDLE",
            "strike_mode": "ATM",
            "entry_time": "11:05",
            "exit_time": "15:25",
            "offset_points": 0,
            "adjustment_points": 100,
            "hedge_mode": "none",
            **overrides,
        }
        alerts = MagicMock()
        alerts.telegram.send = AsyncMock()
        scanner = ATLStraddleScanner(
            client=MagicMock(), alert_manager=alerts,
            settings_loader=lambda: dict(config),
        )
        scanner._fetch_option_quote = AsyncMock(return_value={"ltp": 2.0})

        async def build_leg(instrument, strike, option_type, fallback_quote=None):
            return ATLLeg(option_type=option_type, strike=strike, premium=2.0)

        scanner._build_leg = AsyncMock(side_effect=build_leg)
        scanner._execute_entry_legs = AsyncMock(return_value=True)
        scanner._place_leg_order = AsyncMock(return_value=True)
        scanner._switch_shorts = AsyncMock(return_value=True)
        return scanner

    async def test_entry_respects_exchange_compatible_strategy_intervals(self):
        cases = (
            (NIFTY, 100, 24145, 24100),
            (NIFTY, 200, 24145, 24200),
            (NIFTY, 50, 24145, 24150),
            (SENSEX, 200, 76145, 76200),
            (SENSEX, 100, 76145, 76100),
            (SENSEX, 50, 76145, 76100),
            (NIFTY, 75, 24145, 24150),
        )
        for instrument, interval, spot, expected in cases:
            with self.subTest(index=instrument.symbol, interval=interval):
                scanner = self.scanner(interval, instrument)
                await scanner.run_cycle(pd.DataFrame({"close": [spot]}), instrument, 0)
                scanner._execute_entry_legs.assert_awaited_once()
                _, ce_leg, pe_leg = scanner._execute_entry_legs.call_args.args
                self.assertEqual((ce_leg.strike, pe_leg.strike), (expected, expected))

    async def test_premium_hedges_use_the_same_100_point_grid_as_entry(self):
        scanner = self.scanner(100, hedge_mode="premium", hedge_enabled=True)
        await scanner.run_cycle(pd.DataFrame({"close": [24145]}), NIFTY, 0)
        scanner._place_leg_order.assert_has_awaits([
            unittest.mock.call(NIFTY, scanner._state.hedge_ce, "BUY", 1, "hedge_entry"),
            unittest.mock.call(NIFTY, scanner._state.hedge_pe, "BUY", 1, "hedge_entry"),
        ], any_order=True)
        self.assertEqual(scanner._state.straddle_strike, 24100)
        self.assertEqual(scanner._state.hedge_ce.strike, 24200)
        self.assertEqual(scanner._state.hedge_pe.strike, 24000)

    async def test_reform_uses_configured_interval_without_changing_static_legs(self):
        for static in (False, True):
            with self.subTest(static_legs=static):
                scanner = self.scanner(100, static_legs=static)
                await scanner.run_cycle(pd.DataFrame({"close": [24145]}), NIFTY, 0)
                await scanner.run_cycle(pd.DataFrame({"close": [24245]}), NIFTY, 1)
                if static:
                    scanner._switch_shorts.assert_not_awaited()
                    self.assertEqual(scanner._state.straddle_strike, 24100)
                else:
                    scanner._switch_shorts.assert_awaited_once()
                    _, ce_leg, pe_leg = scanner._switch_shorts.call_args.args
                    self.assertEqual((ce_leg.strike, pe_leg.strike), (24200, 24200))


if __name__ == "__main__":
    unittest.main()
