"""REST endpoints for the Market Story (option positioning) page.

Read-only over stored chain snapshots. The timeline is recomputed on every
request rather than stored, so a detector corrected tomorrow also corrects every
session already recorded. The only write here is a manual poll, which exists so
the page can be used before a collector loop is running.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from app.positioning import agent as positioning_agent
from app.positioning import option_chain, storage
from app.positioning import view as positioning_view
from app.positioning import writer as positioning_writer
from app.positioning.dhan import DhanError

router = APIRouter(prefix="/api", tags=["positioning"])

# The write endpoints are not loopback-only any more: the dashboard is a
# separate origin served by nginx, so the client address is the proxy's rather
# than the reader's. The custom header survives as the guard that matters -
# a cross-origin form or image cannot set one, so it still blocks the drive-by
# request this was protecting against.
UI_HEADER = "x-requested-with"
UI_HEADER_VALUE = "agentic-trading-ui"


def _guard(request: Request) -> None:
    if request.headers.get(UI_HEADER) != UI_HEADER_VALUE:
        raise HTTPException(status_code=403, detail="Missing UI request header.")


def _today_ist() -> str:
    return datetime.now(option_chain.IST).date().isoformat()


def _push(alerts: list[dict], level: str, title: str, message: str,
          *, source: str, at: str = "") -> None:
    alerts.append({
        "level": level,
        "title": title,
        "message": message,
        "source": source,
        "at": at,
    })


def _positioning_alerts(view: dict, out: dict, *, session_date: str) -> list[dict]:
    """Reader-facing alerts distilled from the positioning payload.

    These are observational prompts, not orders. They exist so a reader can see
    important state changes in one place without replacing the full positioning
    page.
    """
    alerts: list[dict] = []
    direction = (view.get("direction") or {})
    stability = (view.get("stability") or {})
    story = (view.get("story") or {})
    coverage = (view.get("coverage") or {})

    label = str(direction.get("label") or "")
    at = str(direction.get("at") or view.get("reading_at") or "")
    if label and label != "Neutral":
        _push(alerts, "info", "Direction Shift", f"Direction is {label}.",
              source="positioning", at=at)

    if stability.get("unstable"):
        _push(alerts, "warn", "Direction Unstable",
              "Direction has changed often today; treat the current bias as fragile.",
              source="positioning", at=at)

    changed = list(story.get("changed") or [])
    for line in changed[:4]:
        _push(alerts, "info", "New Behaviour", str(line),
              source="positioning", at=str(story.get("at") or at))

    in_session = int(coverage.get("in_session") or 0)
    if in_session < 12:
        _push(alerts, "warn", "Low Coverage",
              f"Only {in_session} in-session snapshots are available for {session_date};"
              " early reads are less reliable.",
              source="positioning", at=str(view.get("reading_at") or ""))

    # Preserve deterministic order and deduplicate exact repeats.
    dedup: set[tuple[str, str, str]] = set()
    out_rows: list[dict] = []
    for a in alerts:
        key = (a["level"], a["title"], a["message"])
        if key in dedup:
            continue
        dedup.add(key)
        out_rows.append(a)
    return out_rows


@router.get("/positioning/sessions")
def positioning_sessions(limit: int = 30) -> dict:
    days = storage.chain_session_dates("NIFTY", max(1, min(limit, 200)))
    return {"sessions": days, "latest_bucket": storage.latest_chain_bucket("NIFTY")}


@router.get("/positioning/collector")
def positioning_collector() -> dict:
    """Whether the server-side collector is running, and what it last stored.

    The series is only trustworthy if something was awake to record it, and the
    page cannot tell a quiet market from a dead collector by looking at the
    chart. This is how you check without opening the database.
    """
    try:
        from app.positioning.scheduler import get_scheduler_status
        return get_scheduler_status()
    except Exception as exc:  # pragma: no cover
        return {"enabled": False, "last_error": str(exc)}


@router.get("/positioning")
def positioning(session_date: str | None = None) -> dict:
    day = session_date or _today_ist()
    rows = storage.load_chain_day(day, "NIFTY")
    out = positioning_agent.interpret(rows)
    # How the session was obtained travels with it. A rebuilt session reads a
    # bar close where a poll reads last traded price and covers a narrower
    # ladder, so a page that showed both the same way would invite exactly the
    # comparison that is not valid.
    #
    # `view` is the reader's payload and everything beside it is the detail
    # underneath. Both are served from ONE interpretation rather than two
    # requests, so the story and the developer panel explaining it can never be
    # a bucket apart.
    view = positioning_view.build(rows, out)
    return {"session_date": day, "symbol": "NIFTY",
            "sources": storage.chain_day_sources(day, "NIFTY"),
            "alerts": _positioning_alerts(view, out, session_date=day),
            "view": view, **out}


@router.post("/positioning/story")
def positioning_story(request: Request, session_date: str | None = None) -> dict:
    """Rewrite the market story card through the local model, then check it.

    Separate from the GET because a cold Ollama load runs into minutes, and a
    page whose first card blocks on it is a page that is broken whenever Ollama
    is not running.
    """
    _guard(request)
    day = session_date or _today_ist()
    rows = storage.load_chain_day(day, "NIFTY")
    view = positioning_view.build(rows, positioning_agent.interpret(rows))
    if not view.get("available"):
        raise HTTPException(status_code=404, detail="no snapshots for that session")
    return {"session_date": day, **positioning_writer.write_narrative(view["story"])}


@router.post("/positioning/narrate")
def positioning_narrate(request: Request, session_date: str | None = None,
                        at: str | None = None) -> dict:
    """Ask the local model to rewrite one fifteen-minute story, then check it."""
    _guard(request)
    day = session_date or _today_ist()
    out = positioning_agent.interpret(storage.load_chain_day(day, "NIFTY"))
    told = out.get("stories") or []
    if not told:
        raise HTTPException(status_code=404, detail="no story for that session yet")
    story = next((s for s in told if s["at"] == at), told[-1])
    return {"session_date": day, **positioning_writer.write(story)}


@router.post("/positioning/poll")
def positioning_poll(request: Request) -> dict:
    """Fetch and store one chain snapshot now.

    Guarded like the other writes. Dhan allows one chain request every three
    seconds and this endpoint does not queue, so a page that polls it in a tight
    loop will simply be throttled by the fetcher rather than by the exchange.
    """
    _guard(request)
    try:
        snapshot = option_chain.fetch_chain("NIFTY")
    except DhanError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    written = storage.save_chain_snapshot(snapshot)
    return {"captured_at": snapshot["captured_at"], "spot": snapshot["spot"],
            "expiry": snapshot["expiry"], "strikes": written}
