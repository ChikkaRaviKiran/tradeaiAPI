import unittest

from sqlalchemy.pool import NullPool

from app.backtest.strategy_evaluator import StrategyEvaluator


class EvaluatorDatabaseTests(unittest.TestCase):
    def test_connections_are_not_reused_across_event_loops(self):
        evaluator = StrategyEvaluator()
        self.assertIsInstance(evaluator._engine.sync_engine.pool, NullPool)


if __name__ == "__main__":
    unittest.main()
