import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.pool import NullPool

from app.db import models
from app.engine.atl_settings import _load_from_db_sync, _mirror_to_db_sync
from app.execution.broker_factory import _fetch_account_row_sync


class ShortLivedDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.row = SimpleNamespace(
            id=5, name="test", broker="dhan", client_id="id", api_key=None,
            api_secret=None, password=None, mpin=None, totp_secret=None,
            access_token=None, refresh_token=None, proxy_url=None, updated_at=None,
        )
        self.result = MagicMock()
        self.result.scalar_one_or_none.return_value = self.row
        self.result.scalars.return_value.first.return_value = None
        self.session = MagicMock()
        self.session.execute = AsyncMock(return_value=self.result)
        self.session.__aenter__ = AsyncMock(return_value=self.session)
        self.session.__aexit__ = AsyncMock(return_value=False)
        self.transaction = MagicMock()
        self.transaction.__aenter__ = AsyncMock()
        self.transaction.__aexit__ = AsyncMock(return_value=False)
        self.session.begin.return_value = self.transaction

    def test_short_lived_factory_has_no_connection_pool(self):
        self.assertIsInstance(models._short_lived_engine.sync_engine.pool, NullPool)

    def test_sync_and_running_loop_callers_use_isolated_sessions(self):
        with patch.object(models, "ShortLivedAsyncSessionLocal", return_value=self.session) as factory, \
             patch.object(models, "AsyncSessionLocal", side_effect=AssertionError("shared pool used")):
            for _ in range(2):
                self.assertEqual(_fetch_account_row_sync(5)["id"], 5)
                self.assertIsNone(_load_from_db_sync())
                _mirror_to_db_sync({})

            async def in_running_loop():
                self.assertEqual(_fetch_account_row_sync(5)["id"], 5)
                self.assertIsNone(_load_from_db_sync())
                _mirror_to_db_sync({})

            asyncio.run(in_running_loop())
            self.assertEqual(factory.call_count, 9)

    def test_mirror_runtime_error_does_not_retry(self):
        self.session.execute.side_effect = RuntimeError("database failure")
        with patch.object(models, "ShortLivedAsyncSessionLocal", return_value=self.session):
            async def in_running_loop():
                with self.assertRaisesRegex(RuntimeError, "database failure"):
                    _mirror_to_db_sync({})

            asyncio.run(in_running_loop())
            self.assertEqual(self.session.execute.await_count, 1)


if __name__ == "__main__":
    unittest.main()
