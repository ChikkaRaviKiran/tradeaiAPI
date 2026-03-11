"""Application entry point — starts FastAPI server and trading orchestrator."""

import asyncio
import logging
import logging.handlers
import os
import sys
import threading

import uvicorn

from app.api.routes import app, get_state
from app.core.config import settings
from app.engine.orchestrator import Orchestrator

# ── Logging setup: both console AND file ──────────────────────────────────
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "tradeai.log")

log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=log_format,
    datefmt=log_datefmt,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
    ],
)

logger = logging.getLogger(__name__)


async def run_orchestrator() -> None:
    """Run the trading orchestrator in the background."""
    orchestrator = Orchestrator()
    state = get_state()
    state["orchestrator"] = orchestrator
    await orchestrator.start()


def start_orchestrator_thread() -> None:
    """Run orchestrator in a separate thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_orchestrator())


def main() -> None:
    logger.info("Starting TradeAI System v1.0")

    # Start orchestrator in background thread
    orch_thread = threading.Thread(target=start_orchestrator_thread, daemon=True)
    orch_thread.start()

    # Start FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
