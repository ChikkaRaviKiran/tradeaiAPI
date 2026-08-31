"""Market Story - the presentation layer over `positioning_agent`.

WHAT THIS MODULE IS ALLOWED TO DO
---------------------------------
Rephrase, group, count and lay out. Nothing here detects a behaviour, stores a
row, or moves a state machine. `positioning_agent` decides WHAT happened and is
imported read-only; this module decides only HOW it is said. That boundary is
the reason the two files exist separately: a wording change must never be able
to alter a detection, and a detection fixed tomorrow must reach the page without
anyone editing prose.

Every helper it needs from the agent is imported rather than reimplemented. A
second copy of `_dominant` that drifted by one line would give the story a
different support strike from the scoreboard beside it, and both would look
right.

THE ONE DERIVED CONCLUSION
--------------------------
`direction()` is the only thing on the page that takes a side, and it is
arithmetic - a fixed table of weights summed over the behaviours currently
running. It is deliberately NOT available to the language model. A model that
can be asked "what does this mean" will eventually answer, and the answer will
read exactly as confidently when it is wrong.

Three properties make it conservative on purpose:

  1. A behaviour counts ONCE no matter how many strikes carry it. Four strikes
     of call writing is one observation about call writers, not four; without
     this a single theme spread across a ladder reaches "Strongly" alone.
  2. `MIN_CONTRIBUTORS` independent behaviours are required before the label may
     leave Neutral, so one transient OI print cannot flip the page.
  3. The bands are wide and start at 2, so the common case - a quiet chain with
     one thing going on - reads Neutral, which is the honest answer.

WHAT IT IS NOT
--------------
Not a forecast. The weights were chosen, not measured, exactly like the
thresholds in the agent, and this repository has already measured that price
behaviour at call-OI resistance carried no information at all. The label
describes the balance of option positioning right now. `counterpoints` is
mandatory rather than decorative for that reason: a conclusion printed without
its opposing evidence is an opinion with a number stapled to it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any

from app.positioning.agent import (
    BUCKET_MINUTES,
    QUIET_UNTIL,
    _atm_iv,
    _dominant,
    _oi_total,
    _pct,
    buckets as _buckets,
)
from app.positioning.max_pain import max_pain as _max_pain

# --- window and freshness -------------------------------------------------
# The reading is taken from the last bucket INSIDE the session. Polls land
# outside it in practice - a chain fetched at 21:00 returns the frozen close
# and stores happily - and a "current positioning" card driven by a five-hour
# old frozen snapshot is not stale-looking, it is wrong-looking, because every
# change reads as exactly zero.

SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"
FULL_SESSION_BUCKETS = 75
STALE_AFTER_MIN = 20

# --- reporting cadence ------------------------------------------------------
# Reports land on the wall clock: 09:30, 09:45, 10:00. `positioning_agent`
# reports every fifteen minutes too, but it anchors the grid to the LAST bucket
# and counts backwards, so a session whose final poll arrived at 21:05 reported
# at 21:05, 20:50, ... 15:20 - the right spacing on the wrong offset, and no row
# at any time a reader would think to ask about. Spacing and phase are not the
# same property, and only the spacing was ever specified.
CHECKPOINT_MINUTES = 15

# --- direction --------------------------------------------------------------
DIRECTION_LOOKBACK = 6      # buckets a support/resistance move keeps its vote
MIN_CONTRIBUTORS = 2        # independent behaviours needed to leave Neutral
NEUTRAL_BAND = 1            # |net| at or below this is Neutral regardless
BAND_SLIGHT = 2
BAND_MODERATE = 4
BAND_STRONG = 7
UNSTABLE_FLIPS = 4          # label changes in a session before it is called unstable

# --- status calling ---------------------------------------------------------
FLOW_LOOKBACK = 3           # buckets used to call writers adding or reducing
FLOW_MIN_PCT = 1.0          # total OI change under this is "Steady"
IV_STABLE_PCT = 3.0         # ATM IV change under this is "Stable"
HOLD_BUCKETS = 4            # a level unchanged this long is "Holding"
TRADING_DAYS = 252          # expected move is quoted for one trading day

# A behaviour that stops for one bucket and resumes is one phase, not two. The
# first pass without bridging turned a single morning of put writing into four
# entries fifteen minutes apart, which is the five-minute detector leaking
# straight back through the panel built to hide it.
MIN_PHASE_BUCKETS = 2       # a phase spans at least two buckets, so ten minutes
PHASE_GAP_BUCKETS = 2       # bucket(s) of silence bridged inside one phase

# Weights are per BEHAVIOUR, not per strike. Sign is the conventional reading of
# who is under pressure, and the magnitudes are the ones in the specification:
# a level relocating is worth more than flow at a level, and a level relocating
# AGAINST the side that owns it is worth most of all.
#
# The four the specification did not name are carried at lower weight rather
# than dropped. Half the detector's output being invisible to the only derived
# number on the page would be a silent hole, and a hole is worse than a small
# vote because nobody can see it.
WEIGHTS: dict[tuple[str, str], int] = {
    ("pe", "writing"): +3,          # put writers accepting the obligation below
    ("ce", "short_covering"): +3,   # call writers buying back
    ("ce", "long_buildup"): +2,     # call buyers adding
    ("pe", "unwinding"): +1,        # put buyers closing - hedges lifted
    ("ce", "writing"): -3,          # fresh call writing
    ("pe", "short_covering"): -3,   # put writers buying back, often in a hurry
    ("pe", "long_buildup"): -2,     # put buyers adding
    ("ce", "unwinding"): -1,        # call buyers closing
}

# A level moving. Read as the side that owns it conceding or advancing.
SHIFT_WEIGHTS: dict[tuple[str, str], int] = {
    ("support", "higher"): +4,
    ("support", "lower"): -2,
    ("resistance", "higher"): +2,
    ("resistance", "lower"): -4,
}

# Implied volatility and range building carry NO weight and never will. Neither
# says anything about direction, and IV in particular is the reading most often
# smuggled into one.
NON_DIRECTIONAL = ("iv_expansion", "iv_crush", "range_building")

# --- market language --------------------------------------------------------
# The left column is what the reader is told is happening; the agent's own
# labels ("CE short covering", state "growing") stay in developer details.
PARTICIPANT: dict[tuple[str, str], tuple[str, str]] = {
    ("ce", "writing"): ("Call Writers", "Adding Exposure"),
    ("ce", "short_covering"): ("Call Writers", "Reducing Exposure"),
    ("ce", "long_buildup"): ("Call Buyers", "Adding Positions"),
    ("ce", "unwinding"): ("Call Buyers", "Closing Positions"),
    ("pe", "writing"): ("Put Writers", "Adding Exposure"),
    ("pe", "short_covering"): ("Put Writers", "Reducing Exposure"),
    ("pe", "long_buildup"): ("Put Buyers", "Adding Positions"),
    ("pe", "unwinding"): ("Put Buyers", "Closing Positions"),
}

PHASE_NAME: dict[tuple[str, str], str] = {
    ("ce", "writing"): "Call Writers Building",
    ("ce", "short_covering"): "Call Writers Exiting",
    ("ce", "long_buildup"): "Call Buying",
    ("ce", "unwinding"): "Call Buyers Leaving",
    ("pe", "writing"): "Put Writers Building",
    ("pe", "short_covering"): "Put Writers Exiting",
    ("pe", "long_buildup"): "Put Buying",
    ("pe", "unwinding"): "Put Buyers Leaving",
}

# Chain-wide moments, which have no side and so are not in PARTICIPANT.
MOMENT_PHRASE: dict[str, str] = {
    "support_shift": "Support relocated",
    "resistance_shift": "Resistance relocated",
    "iv_expansion": "Implied volatility expanded",
    "iv_crush": "Implied volatility collapsed",
    "range_building": "A range began forming",
}

# Clause completing "Whether ...". Phrased so neither answer is the wanted one.
WATCH_VERB: dict[tuple[str, str], str] = {
    ("ce", "writing"): "call writers keep adding exposure",
    ("ce", "short_covering"): "call writers keep reducing exposure",
    ("ce", "long_buildup"): "call buyers keep adding positions",
    ("ce", "unwinding"): "call buyers keep closing positions",
    ("pe", "writing"): "put writers keep adding exposure",
    ("pe", "short_covering"): "put writers keep reducing exposure",
    ("pe", "long_buildup"): "put buyers keep adding positions",
    ("pe", "unwinding"): "put buyers keep closing positions",
}

DISCLAIMER = ("This direction view is derived only from option positioning. "
              "It is an observation of participant positioning, not a prediction.")


# --- small helpers ----------------------------------------------------------

def _hhmm(ts: str) -> str:
    return ts[11:16]


def _minutes(a: str, b: str) -> int:
    fmt = "%Y-%m-%d %H:%M:%S"
    return int((datetime.strptime(b, fmt) - datetime.strptime(a, fmt)).total_seconds() // 60)


def _human(mins: int | None) -> str:
    if mins is None:
        return ""
    if mins <= 0:
        return "just now"
    if mins < 60:
        return f"{mins} minutes"
    h, m = divmod(mins, 60)
    return f"{h}h {m:02d}m"


def _theme(key: str) -> tuple[str, str]:
    """`ce:writing:24500` -> `("ce", "writing")`, the behaviour without the strike.

    Chain-wide keys (`support_shift`, `iv_expansion`) have no colons and are
    never grouped by theme, so they fall out here as a pair that matches nothing
    in `WEIGHTS` rather than raising on the unpack.
    """
    parts = key.split(":")
    return (parts[0], parts[1]) if len(parts) >= 2 else (key, "")


def in_session(seq: list[dict]) -> list[dict]:
    """Buckets inside market hours, in order."""
    return [b for b in seq if SESSION_OPEN <= _hhmm(b["at"]) <= SESSION_CLOSE]


def reading_index(seq: list[dict]) -> int:
    """Index of the bucket the page reads as "now".

    The last in-session bucket when there is one, otherwise the last bucket at
    all - a session that only ever got out-of-hours polls should still render
    something rather than an empty page that looks like a bug.
    """
    for i in range(len(seq) - 1, -1, -1):
        if SESSION_OPEN <= _hhmm(seq[i]["at"]) <= SESSION_CLOSE:
            return i
    return len(seq) - 1


def _mod(hhmm: str) -> int:
    """Minutes since midnight, for arithmetic on the clock grid."""
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def next_mark(hhmm: str) -> str | None:
    """The clock time of the reading after `hhmm`, or None once the bell has gone.

    Published so a card can say when it will next change. Without it a reading
    that is correct and simply has nothing new to say is indistinguishable from
    a page that has stopped updating, and the reader has no way to tell which
    they are looking at except by waiting to find out.
    """
    nxt = _mod(hhmm) + CHECKPOINT_MINUTES
    nxt -= nxt % CHECKPOINT_MINUTES
    if nxt > _mod(SESSION_CLOSE):
        return None
    return f"{nxt // 60:02d}:{nxt % 60:02d}"


def checkpoint_marks(rows: list[dict], i: int) -> list[int]:
    """Bucket indices for each wall-clock quarter hour up to the reading.

    Anchored to the clock rather than to the data, so the reports are always at
    09:30, 09:45, 10:00 and a reader asking "what did it say at ten" has a row
    to look at. Anchoring to the last bucket instead makes the whole grid slide
    whenever the final poll lands somewhere unusual, which is exactly when the
    session is most worth reading back.

    Marks before `QUIET_UNTIL` are skipped because the agent deliberately
    detects nothing in the opening auction, so those rows could only ever say
    Neutral and would read as a flat start that the market did not have.

    A mark with no bucket exactly on it takes the last bucket before it instead
    of being dropped: a missed poll should move a reading by minutes, not delete
    it. Two marks resolving to the same bucket yield one row, since repeating a
    bucket would invent a reading that was never separately observed.
    """
    if not rows or i < 0:
        return []

    close = _mod(SESSION_CLOSE)
    limit = min(_mod(_hhmm(rows[i]["at"])), close)
    step = CHECKPOINT_MINUTES
    first = max(_mod(SESSION_OPEN), _mod(QUIET_UNTIL))
    mark = -(-first // step) * step        # round up onto the quarter-hour grid

    marks: list[int] = []
    seen: set[int] = set()
    while mark <= limit:
        for n in range(i, -1, -1):
            at = _hhmm(rows[n]["at"])
            if SESSION_OPEN <= at <= SESSION_CLOSE and _mod(at) <= mark:
                if n not in seen:
                    seen.add(n)
                    marks.append(n)
                break
        mark += step
    return marks


# --- per-bucket derived series ---------------------------------------------

def derive(seq: list[dict]) -> list[dict]:
    """Flatten each bucket to the handful of numbers the page speaks about.

    Computed once and scanned backwards for durations. Recomputing `_max_pain`
    inside every run-length loop was the obvious version and was quadratic on a
    full session.
    """
    out: list[dict] = []
    for b in seq:
        ladder = {k: {"ce": float(v["ce_oi"] or 0), "pe": float(v["pe_oi"] or 0)}
                  for k, v in b["strikes"].items()}
        out.append({
            "at": b["at"],
            "spot": b["spot"],
            "support": _dominant(b, "pe"),
            "resistance": _dominant(b, "ce"),
            "ce_oi": _oi_total(b, "ce"),
            "pe_oi": _oi_total(b, "pe"),
            "iv": _atm_iv(b),
            "max_pain": _max_pain(ladder, b["spot"], half_width=10) if ladder else None,
        })
    return out


def _run_start(rows: list[dict], i: int, value_of) -> int:
    """How far back the current value of `value_of` has held, as an index."""
    want = value_of(rows[i])
    j = i
    while j > 0 and value_of(rows[j - 1]) == want:
        j -= 1
    return j


def _flow_status(rows: list[dict], i: int, col: str) -> str:
    j = max(0, i - FLOW_LOOKBACK)
    pct = _pct(rows[i][col], rows[j][col])
    if pct is None or abs(pct) < FLOW_MIN_PCT:
        return "Steady"
    return "Adding Exposure" if pct > 0 else "Reducing Exposure"


def _iv_status(rows: list[dict], i: int) -> str:
    j = max(0, i - FLOW_LOOKBACK)
    pct = _pct(rows[i]["iv"], rows[j]["iv"])
    if pct is None or abs(pct) < IV_STABLE_PCT:
        return "Stable"
    return "Rising" if pct > 0 else "Falling"


def _level_status(rows: list[dict], i: int, col: str) -> tuple[str, int]:
    """Status and the index the current level was first seen at."""
    start = _run_start(rows, i, lambda r: r[col])
    if rows[i][col] is None:
        return "Not available", i
    if start == 0 or (i - start) >= HOLD_BUCKETS:
        return "Holding", start
    prev = rows[start - 1][col]
    if prev is None:
        return "Holding", start
    return ("Shifting Higher" if rows[i][col] > prev else "Shifting Lower"), start


# --- current positioning ----------------------------------------------------

def positioning(rows: list[dict], i: int) -> list[dict]:
    """The seven fields, each with a status, how long it has held, and a time.

    Duration is the point of this card. "Support 24300" is a number that could
    have arrived one bucket ago; "Support 24300, holding 45 minutes" is the
    thing a reader actually wanted to know and cannot get from the number.
    """
    cur = rows[i]
    at = cur["at"]
    out: list[dict] = []

    def add(name: str, value: str, status: str, since_i: int | None,
            measure: str) -> None:
        # `since_i` of None means the quantity has no run length. Expected move
        # is recomputed from scratch every bucket, so "holding for 40 minutes"
        # would be a duration invented for layout symmetry.
        mins = None if since_i is None else _minutes(rows[since_i]["at"], at)
        out.append({
            "name": name, "value": value, "status": status,
            "duration_min": mins, "duration": _human(mins),
            "since": None if since_i is None else _hhmm(rows[since_i]["at"]),
            "updated": _hhmm(at), "measure": measure,
        })

    status, since = _level_status(rows, i, "support")
    add("Support", f"{cur['support']:.0f}" if cur["support"] is not None else "-",
        status, since, "heaviest put open interest near the money")

    status, since = _level_status(rows, i, "resistance")
    add("Resistance", f"{cur['resistance']:.0f}" if cur["resistance"] is not None else "-",
        status, since, "heaviest call open interest near the money")

    for col, name in (("ce_oi", "Call Writers"), ("pe_oi", "Put Writers")):
        status = _flow_status(rows, i, col)
        since = i
        while since > 0 and _flow_status(rows, since - 1, col) == status:
            since -= 1
        add(name, f"{cur[col] / 1e5:.1f}L" if cur[col] else "-", status, since,
            f"total {'call' if col == 'ce_oi' else 'put'} open interest near the money")

    status = _iv_status(rows, i)
    since = i
    while since > 0 and _iv_status(rows, since - 1) == status:
        since -= 1
    add("Volatility", f"{cur['iv']:.1f}" if cur["iv"] else "-", status, since,
        "at-the-money implied volatility")

    status, since = _level_status(rows, i, "max_pain")
    status = {"Holding": "Unchanged", "Shifting Higher": "Moved Higher",
              "Shifting Lower": "Moved Lower"}.get(status, status)
    add("Max Pain", f"{cur['max_pain']:.0f}" if cur["max_pain"] is not None else "-",
        status, since, "strike where the most open interest expires worthless")

    if cur["iv"] and cur["spot"]:
        move = cur["spot"] * (cur["iv"] / 100.0) / math.sqrt(TRADING_DAYS)
        add("Expected Move", f"+/-{move:.0f}", "One trading day", None,
            "implied by at-the-money volatility, not a forecast of direction")
    else:
        add("Expected Move", "-", "Not available", None,
            "needs at-the-money implied volatility")
    return out


# --- direction --------------------------------------------------------------

def active_at(events: list[dict], at: str) -> dict[str, dict]:
    """Episodes running as of `at`, by the same rule `interpret` uses.

    Re-derived rather than taken from `interpret`, because that one is fixed at
    the end of the session and the stability series needs the answer at every
    fifteen-minute mark.
    """
    live: dict[str, dict] = {}
    for e in sorted(events, key=lambda e: e["timestamp"]):
        if e["moment"] or e["timestamp"] > at:
            continue
        if e["state"] in ("started", "growing", "strong"):
            live[e["key"]] = e
        else:
            live.pop(e["key"], None)
    return live


def _level_votes(rows: list[dict], i: int) -> list[dict]:
    """Support and resistance relocation, read straight off the level series.

    Taken from the numbers rather than from shift EVENTS on purpose. An event
    fires once, at the bucket the move is confirmed; the bias should keep the
    vote for as long as the move is recent, and reading the series gives that
    without teaching the state machine a second job.
    """
    j = max(0, i - DIRECTION_LOOKBACK)
    votes: list[dict] = []
    for col, name in (("support", "Support"), ("resistance", "Resistance")):
        now, was = rows[i][col], rows[j][col]
        if now is None or was is None or now == was:
            continue
        way = "higher" if now > was else "lower"
        votes.append({
            "kind": f"{col}_shift",
            "label": f"{name} shifted {way}",
            "weight": SHIFT_WEIGHTS[(col, way)],
            "detail": f"heaviest {'put' if col == 'support' else 'call'} open "
                      f"interest {was:.0f} to {now:.0f}",
        })
    return votes


def _contributors(events: list[dict], rows: list[dict], i: int) -> list[dict]:
    """One entry per independent behaviour, weighted once."""
    live = active_at(events, rows[i]["at"])
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for key, e in live.items():
        side, kind = _theme(key)
        if (side, kind) in WEIGHTS:
            grouped[(side, kind)].append(e["strike"])

    out: list[dict] = []
    for (side, kind), strikes in sorted(grouped.items()):
        who, what = PARTICIPANT[(side, kind)]
        at_strikes = ", ".join(f"{s:.0f}" for s in sorted(x for x in strikes if x))
        out.append({
            "kind": f"{side}:{kind}",
            "label": f"{who} {what.lower()}",
            "weight": WEIGHTS[(side, kind)],
            "detail": f"at {at_strikes}" if at_strikes else "near the money",
        })
    return out + _level_votes(rows, i)


def _band(net: int, n_contributors: int) -> str:
    """Weights to words. Wide bands, and Neutral is the default answer."""
    if n_contributors < MIN_CONTRIBUTORS or abs(net) <= NEUTRAL_BAND:
        return "Neutral"
    size = abs(net)
    if size >= BAND_STRONG:
        word = "Strongly"
    elif size >= BAND_MODERATE:
        word = "Moderately"
    else:
        word = "Slightly"
    return f"{word} {'Bullish' if net > 0 else 'Bearish'}"


def _counterpoints(rows: list[dict], i: int, against: list[dict],
                   net: int, n: int) -> list[str]:
    """Opposing evidence. Never allowed to come back empty.

    A one-sided conclusion is the failure mode this whole page is built to
    avoid, so the structural checks below run even when the behaviours all
    agree - and when they all agree, that agreement is itself the thing worth
    warning about.
    """
    out = [f"{c['label'].capitalize()} ({c['detail']})." for c in against]
    cur = rows[i]

    if cur["resistance"] is not None and cur["spot"] is not None:
        if cur["spot"] < cur["resistance"]:
            out.append(f"Price has not traded above the heaviest call open "
                       f"interest at {cur['resistance']:.0f}.")
        else:
            out.append(f"Price is above the heaviest call open interest at "
                       f"{cur['resistance']:.0f}, which has not been tested from above.")

    if _iv_status(rows, i) == "Stable" and cur["iv"]:
        out.append(f"Implied volatility is flat at {cur['iv']:.1f}, so options "
                   f"are not being repriced for a wider outcome.")

    status, _ = _level_status(rows, i, "support")
    if status == "Holding" and cur["support"] is not None:
        out.append(f"Support has not moved from {cur['support']:.0f}.")

    if net and not against:
        out.append("No behaviour is pulling the other way, so this reading "
                   "rests on one side of the evidence only.")
    if n < MIN_CONTRIBUTORS:
        out.append(f"Only {n} independent behaviour(s) are running, below the "
                   f"{MIN_CONTRIBUTORS} required to leave Neutral.")
    return out


def direction(events: list[dict], rows: list[dict], i: int) -> dict[str, Any]:
    """The one derived conclusion on the page. Arithmetic, never a model."""
    contributors = _contributors(events, rows, i)
    net = sum(c["weight"] for c in contributors)
    label = _band(net, len(contributors))

    sign = (net > 0) - (net < 0)
    agree = [c for c in contributors if c["weight"] and (c["weight"] > 0) == (sign > 0)]
    against = [c for c in contributors if c["weight"] and (c["weight"] > 0) != (sign > 0)]
    if label == "Neutral":
        agree, against = [], contributors

    reasons = [f"{c['label'].capitalize()} ({c['detail']})." for c in agree]
    if not reasons:
        reasons = ["No behaviour is currently pushing the balance either way."]

    return {
        "label": label,
        "net": net,
        "bullish": sum(c["weight"] for c in contributors if c["weight"] > 0),
        "bearish": sum(c["weight"] for c in contributors if c["weight"] < 0),
        "contributors": contributors,
        "reasons": reasons,
        "counterpoints": _counterpoints(rows, i, against, net, len(contributors)),
        "at": _hhmm(rows[i]["at"]),
        "disclaimer": DISCLAIMER,
        "tested": False,
    }


def stability(events: list[dict], rows: list[dict], marks: list[int],
              i: int) -> dict[str, Any]:
    """How settled the label is, measured at the fifteen-minute marks.

    A bias that has flipped all day is not evidence, and the honest way to say
    so is to show the count rather than to hide the flips by smoothing them.
    """
    series = [{"at": _hhmm(rows[m]["at"]), "timestamp": rows[m]["at"],
               "label": direction(events, rows, m)["label"]}
              for m in marks if m <= i]
    if not series:
        return {"series": [], "changes": 0, "stable_since": None,
                "duration_min": 0, "duration": "0 minutes", "unstable": False}

    changes = sum(1 for a, b in zip(series, series[1:]) if a["label"] != b["label"])
    current = series[-1]["label"]
    start = series[-1]
    for point in reversed(series):
        if point["label"] != current:
            break
        start = point
    mins = _minutes(start["timestamp"], rows[i]["at"])
    return {
        "series": series,
        "changes": changes,
        "stable_since": start["at"],
        "duration_min": mins,
        "duration": _human(mins),
        "unstable": changes >= UNSTABLE_FLIPS,
    }


# --- fifteen-minute checkpoints ---------------------------------------------

def _window_changes(events: list[dict], since: str | None, until: str) -> list[str]:
    """Behaviours that began or ended inside `(since, until]`, in plain words.

    Only the two edges are reported. "Growing" and "strong" are the detector
    grading an episode it already told you about, and repeating a running
    behaviour at every checkpoint buries the one line where it actually started.
    """
    out: list[str] = []
    seen: set[str] = set()
    for e in sorted(events, key=lambda e: e["timestamp"]):
        if e["timestamp"] > until:
            break
        if since is not None and e["timestamp"] <= since:
            continue

        if e["moment"]:
            phrase = MOMENT_PHRASE.get(e["key"])
        else:
            theme = _theme(e["key"])
            if theme not in PARTICIPANT or e["state"] not in ("started", "finished"):
                continue
            who, what = PARTICIPANT[theme]
            verb = "began" if e["state"] == "started" else "stopped"
            phrase = f"{who} {verb} {what.lower()}"

        if phrase and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    return out


def checkpoints(events: list[dict], rows: list[dict], marks: list[int]) -> list[dict]:
    """One reading per quarter hour: where the market stood and what moved.

    Each row is the full direction calculation re-run at that bucket, not the
    final reading carried backwards, so scrolling the list shows the session
    being formed rather than today's answer painted over the morning.
    """
    out: list[dict] = []
    prev_at: str | None = None
    prev_label: str | None = None

    for n in marks:
        cur = rows[n]
        d = direction(events, rows, n)
        changed = _window_changes(events, prev_at, cur["at"])

        out.append({
            "at": _hhmm(cur["at"]),
            "timestamp": cur["at"],
            "spot": round(cur["spot"], 2) if cur["spot"] else None,
            "label": d["label"],
            "net": d["net"],
            "moved": prev_label is not None and d["label"] != prev_label,
            "support": None if cur["support"] is None else round(cur["support"]),
            "resistance": None if cur["resistance"] is None else round(cur["resistance"]),
            "iv": None if not cur["iv"] else round(cur["iv"], 1),
            "max_pain": None if cur["max_pain"] is None else round(cur["max_pain"]),
            "changed": changed,
            "summary": changed[0] + "." if changed else "No new behaviour.",
        })
        prev_at, prev_label = cur["at"], d["label"]
    return out


def _episodes(events: list[dict], since: str | None, until: str,
              state: str) -> list[str]:
    """Behaviours entering `state` inside `(since, until]`, grouped by theme.

    Grouped because the detector fires per strike: four strikes of fresh call
    writing is one thing that happened at four prices, and listing it four times
    makes a single behaviour look like a wave.
    """
    by: dict[tuple[str, str], set[float]] = defaultdict(set)
    moments: list[str] = []
    for e in sorted(events, key=lambda e: e["timestamp"]):
        if e["timestamp"] > until:
            break
        if since is not None and e["timestamp"] <= since:
            continue
        if e["moment"]:
            if state == "started" and e["key"] in MOMENT_PHRASE:
                phrase = MOMENT_PHRASE[e["key"]]
                if phrase not in moments:
                    moments.append(phrase)
            continue
        if e["state"] != state:
            continue
        theme = _theme(e["key"])
        if theme in PARTICIPANT:
            by[theme].add(e["strike"])

    verb = "began" if state == "started" else "stopped"
    out = []
    for theme, strikes in sorted(by.items()):
        who, what = PARTICIPANT[theme]
        at = ", ".join(f"{s:.0f}" for s in sorted(x for x in strikes if x))
        out.append(f"{who} {verb} {what.lower()}" + (f" at {at}" if at else ""))
    return out + moments


def window_report(events: list[dict], rows: list[dict], prev_i: int | None,
                  i: int) -> dict[str, Any]:
    """What happened in the reporting window ending at `i`.

    Built here rather than taken from `positioning_agent.stories` because that
    one anchors its grid to the last bucket and counts backwards, so its window
    boundaries do not line up with the marks this page displays. A card headed
    "last fifteen minutes" sitting beside a list of quarter-hour readings must
    mean the same fifteen minutes as the newest row in that list.
    """
    since = rows[prev_i]["at"] if prev_i is not None else None
    until = rows[i]["at"]

    started = _episodes(events, since, until, "started")
    stopped = _episodes(events, since, until, "finished")

    # Running now AND running at the window's start. Comparing the two live sets
    # is what separates "still going" from "started during this window", which
    # the newest event's own timestamp cannot answer once it has been re-graded.
    live_now = active_at(events, until)
    live_then = active_at(events, since) if since is not None else {}
    carried: dict[tuple[str, str], set[float]] = defaultdict(set)
    for key in set(live_now) & set(live_then):
        theme = _theme(key)
        if theme in PARTICIPANT:
            carried[theme].add(live_now[key]["strike"])

    continuing = []
    for theme, strikes in sorted(carried.items()):
        who, what = PARTICIPANT[theme]
        at = ", ".join(f"{s:.0f}" for s in sorted(x for x in strikes if x))
        continuing.append(f"{who} still {what.lower()}" + (f" at {at}" if at else ""))

    return {
        "changed": started,
        "stopped": stopped,
        "continuing": continuing,
        "from": _hhmm(since) if since else _hhmm(until),
        "at": _hhmm(until),
        "importance": "HIGH" if (started or stopped) else "LOW",
    }


# --- phases -----------------------------------------------------------------

def phases(events: list[dict], rows: list[dict]) -> list[dict]:
    """The timeline as spans of behaviour instead of a list of transitions.

    387 transitions is a log. Eleven phases is a session someone can read. The
    transitions are still there, one panel down, for anyone asking why.
    """
    ordered = sorted(events, key=lambda e: e["timestamp"])
    live: dict[str, str] = {}
    idx = 0
    per_bucket: list[set[tuple[str, str]]] = []
    for r in rows:
        while idx < len(ordered) and ordered[idx]["timestamp"] <= r["at"]:
            e = ordered[idx]
            idx += 1
            if e["moment"]:
                continue
            if e["state"] in ("started", "growing", "strong"):
                live[e["key"]] = e["state"]
            else:
                live.pop(e["key"], None)
        per_bucket.append({_theme(k) for k in live})

    spans: list[dict] = []
    open_at: dict[tuple[str, str], int] = {}
    last_seen: dict[tuple[str, str], int] = {}
    for n, themes in enumerate(per_bucket):
        for theme in themes:
            open_at.setdefault(theme, n)
            last_seen[theme] = n
        for theme in [t for t in open_at if t not in themes]:
            # Bridge a short silence rather than closing the phase. Only close
            # once the gap is longer than a behaviour is allowed to pause for.
            if n - last_seen[theme] > PHASE_GAP_BUCKETS:
                spans.append({"theme": theme, "start": open_at.pop(theme),
                              "end": last_seen[theme]})
    for theme, start in open_at.items():
        spans.append({"theme": theme, "start": start, "end": last_seen[theme]})

    out = []
    for s in spans:
        if s["end"] - s["start"] < MIN_PHASE_BUCKETS - 1:
            continue
        out.append({
            "from": _hhmm(rows[s["start"]]["at"]),
            "to": _hhmm(rows[s["end"]]["at"]),
            "name": PHASE_NAME[s["theme"]],
            "minutes": (s["end"] - s["start"]) * BUCKET_MINUTES,
        })
    return sorted(out, key=lambda p: (p["from"], p["name"]))


# --- watch next -------------------------------------------------------------

def watch_next(events: list[dict], rows: list[dict], i: int) -> list[str]:
    """Where to look, never what to do.

    Regenerated here rather than translated from the agent's version, because
    that one names behaviours the way the detector does ("call short covering
    at 24300") and this card is read by someone who should never need to learn
    that vocabulary. Every line is a question with no preferred answer - which
    is what separates "whether resistance holds" from "resistance should hold".
    """
    cur = rows[i]
    out: list[str] = []

    live = active_at(events, cur["at"])
    seen: set[tuple[str, str]] = set()
    for key in sorted(live):
        theme = _theme(key)
        if theme in WATCH_VERB and theme not in seen:
            seen.add(theme)
            out.append(f"Whether {WATCH_VERB[theme]}.")
        if len(out) >= 3:
            break

    status, _ = _level_status(rows, i, "support")
    if cur["support"] is not None:
        out.append(f"Whether support {'shifts' if status == 'Holding' else 'settles'}"
                   f" away from {cur['support']:.0f}.")

    if _iv_status(rows, i) == "Stable":
        out.append("Whether implied volatility begins expanding.")
    else:
        out.append("Whether implied volatility keeps moving or settles.")

    if cur["resistance"] is not None and cur["spot"] is not None:
        side = "above" if cur["spot"] < cur["resistance"] else "below"
        out.append(f"Whether price is accepted {side} {cur['resistance']:.0f}, "
                   f"where call open interest is heaviest.")
    return out[:5]


# --- the story --------------------------------------------------------------

def facts(fields: list[dict], story: dict | None) -> list[dict]:
    """The structured sheet the model is allowed to see. Nothing else.

    Statuses and durations, no raw ladder and no direction. The model cannot
    repeat a number it was not handed, and it is never handed the conclusion.
    """
    sheet = [{"name": f["name"], "status": f["status"], "value": f["value"],
              "duration": f["duration"]} for f in fields]
    if story:
        sheet.append({"name": "Recent changes", "status": "observed",
                      "value": "; ".join(story.get("changed") or ["nothing new"]),
                      "duration": f"{story.get('from')} to {story.get('at')}"})
    return sheet


def narrative(fields: list[dict], story: dict | None, phase_list: list[dict]) -> list[str]:
    """Five or six computed lines. The paragraph the page opens with.

    Assembled by concatenation, like everything else here, so it is correct
    before any model is involved. The rewrite endpoint improves the reading of
    these exact lines and can never add one.
    """
    by = {f["name"]: f for f in fields}
    lines: list[str] = []

    for name in ("Call Writers", "Put Writers"):
        f = by.get(name)
        if f and f["status"] != "Steady":
            lines.append(f"{name} have been {f['status'].lower()} for "
                         f"{f['duration']} ({f['value']} open interest).")
    if not lines:
        lines.append("Neither call nor put writers have moved enough to "
                     "register in the last fifteen minutes.")

    sup, res = by.get("Support"), by.get("Resistance")
    if sup and sup["value"] != "-":
        lines.append(f"Support at {sup['value']} is {sup['status'].lower()} "
                     f"({sup['duration']}).")
    if res and res["value"] != "-":
        lines.append(f"Resistance at {res['value']} is {res['status'].lower()} "
                     f"({res['duration']}).")

    vol = by.get("Volatility")
    if vol and vol["value"] != "-":
        lines.append(f"At-the-money implied volatility is {vol['status'].lower()} "
                     f"at {vol['value']}.")

    recent = (story or {}).get("changed") or []
    if len(recent) <= 1 and not phase_list:
        lines.append("No unusual positioning change has appeared so far.")
    elif story and story.get("importance") == "LOW":
        lines.append("No unusual positioning change has appeared in the last "
                     "fifteen minutes.")
    return lines[:6]


# --- assembly ---------------------------------------------------------------

def build(rows: list[dict[str, Any]], interpreted: dict[str, Any]) -> dict[str, Any]:
    """Everything the v2 page renders, from snapshots and one interpretation.

    `interpreted` is passed in rather than recomputed so the page and the
    developer panel underneath it can never disagree about what was detected.

    TWO CLOCKS, DELIBERATELY
    ------------------------
    Collection and detection run on the five-minute grid and lose nothing. The
    page reports on the fifteen-minute grid, from `report` rather than from the
    freshest bucket. A direction label that reconsiders itself every five
    minutes is not a faster read of the market, it is a flicker: the reader
    cannot tell a change that matters from one that will be gone by the next
    poll, and the only defence is to stop reading it.

    The gap between the two is published as `data_at` and `data_lag_min` rather
    than hidden. A page that says 09:30 while holding 09:40 data is fine; a page
    that says 09:30 and lets you assume it is live is not.
    """
    seq = _buckets(rows)
    if not seq:
        return {"available": False, "reason": "no snapshots recorded"}

    series = derive(seq)
    i = reading_index(seq)
    events = interpreted.get("timeline") or []

    marks = checkpoint_marks(series, i)
    # Before the first mark of the day there is no quarter-hour reading to show.
    # Falling back to the freshest bucket keeps the page from looking broken at
    # 09:20; `on_grid` says which of the two the reader is looking at.
    report = marks[-1] if marks else i
    prev_mark = marks[-2] if len(marks) >= 2 else None

    fields = positioning(series, report)
    inside = in_session(seq)
    window = window_report(events, series, prev_mark, report)
    phase_list = phases(events, series[:report + 1])

    return {
        "available": True,
        "reading_at": _hhmm(series[report]["at"]),
        "reading_timestamp": series[report]["at"],
        "data_at": _hhmm(series[i]["at"]),
        "data_lag_min": _minutes(series[report]["at"], series[i]["at"]),
        "next_reading_at": next_mark(_hhmm(series[report]["at"])),
        "on_grid": bool(marks),
        "report_minutes": CHECKPOINT_MINUTES,
        "spot": round(series[report]["spot"], 2),
        "clipped": i < len(seq) - 1,
        "coverage": {
            "in_session": len(inside),
            "expected": FULL_SESSION_BUCKETS,
            "total": len(seq),
            "first": _hhmm(seq[0]["at"]),
            "last": _hhmm(seq[-1]["at"]),
        },
        "story": {
            "lines": narrative(fields, window, phase_list),
            "facts": facts(fields, window),
            "from": window["from"],
            "at": window["at"],
            "importance": window["importance"],
        },
        "direction": direction(events, series, report),
        "stability": stability(events, series, marks, report),
        "positioning": fields,
        "changes": {
            "changed": window["changed"],
            "stopped": window["stopped"],
            "continuing": window["continuing"],
        },
        "watch_next": watch_next(events, series, report),
        "phases": phase_list,
        "checkpoints": checkpoints(events, series, marks),
        "checkpoint_minutes": CHECKPOINT_MINUTES,
    }
