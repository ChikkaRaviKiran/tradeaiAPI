"""Tiny .env loader so short-lived credentials live in one gitignored file.

No dependency on python-dotenv. A variable exported in the SHELL always wins, so
you can override the file for a single command without editing it.

A value this module previously loaded FROM the file does not win, and the
difference matters. `load_dotenv` used to skip any key already in `os.environ`,
which is correct for shell exports and wrong for its own earlier work: the
server loads .env at startup, so the token lands in `os.environ`, and every
later reload then refuses to replace it. The credential panel went on
validating a token that had been overwritten on disk hours before and reported
it as invalid - a stale read presented as a verdict.

So keys are tracked by origin. Shell-exported keys are left alone; keys this
module loaded from the file are refreshed from the file.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

ENV_PATH = BASE_DIR / ".env"

# Keys this module put into os.environ. Anything already in os.environ that is
# NOT in here came from the shell and is never overwritten.
_LOADED_FROM_FILE: set[str] = set()


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into the environment.

    Shell exports are preserved. Values this module loaded earlier are
    refreshed, so editing .env takes effect without restarting the process.
    """
    env_file = path or ENV_PATH
    if not env_file.exists():
        return

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key in os.environ and key not in _LOADED_FROM_FILE:
            continue          # exported in the shell - that wins
        os.environ[key] = value
        _LOADED_FROM_FILE.add(key)


def write_env_value(key: str, value: str, path: Path | None = None) -> Path:
    """Upsert a single KEY in .env, preserving other lines."""
    env_file = path or ENV_PATH
    lines: list[str] = []
    replaced = False

    if env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            if raw.strip().startswith(f"{key}=") or raw.strip().startswith(f"{key} ="):
                lines.append(f"{key}={value}")
                replaced = True
            else:
                lines.append(raw)

    if not replaced:
        lines.append(f"{key}={value}")

    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The file is now the newest source for this key, so a later load_dotenv
    # must be allowed to pick it up. Without this a hand-edited .env would be
    # shadowed by the process's own stale value - the bug above.
    _LOADED_FROM_FILE.add(key)
    return env_file
