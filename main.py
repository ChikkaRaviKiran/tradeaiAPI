"""Application entry point — starts FastAPI server and trading orchestrator.

The orchestrator runs as an asyncio background task inside uvicorn's event
loop (via the FastAPI lifespan handler in routes.py).  This keeps everything
in a single event loop, avoiding asyncpg "Future attached to a different
loop" errors that occur when using a separate thread.
"""

import logging
import logging.handlers
import os
import sys

import uvicorn

from app.api.routes import app
from app.core.config import settings

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


def main() -> None:
    logger.info("Starting TradeAI System v1.0")

    # Start FastAPI server — orchestrator runs as a background task
    # inside the SAME event loop (via lifespan in routes.py)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
