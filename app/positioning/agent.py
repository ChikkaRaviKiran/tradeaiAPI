"""Market Positioning Intelligence - what option participants are doing.

This agent never says what to trade. It says what changed in the option chain
and what that change is conventionally taken to mean, and it keeps those two
things in separate fields so they can never be mistaken for each other.

WHY THE SEPARATION IS THE WHOLE DESIGN
--------------------------------------
"24500 CE open interest fell 18% in fifteen minutes" is an observation. It is
computed from stored numbers and cannot be wrong unless the arithmetic is wrong.

"Resistance is weakening" is a claim about the future wearing the clothes of an
observation. Nothing in this repository has ever measured it. Worse, the one
study that looked reported that price behaviour at call-OI resistance carried no
information at all - every cell across two reading times sat between z -1.12 and
z +1.00 (see /memories/repo/max-pain-findings.md).

So each event carries `what_changed`, generated from the data, and beside it
`commonly_read_as`, a fixed string marked `tested: false`. The second field is
folklore held for inspection, not a conclusion. When enough sessions have been
recorded, `spot` and `at` on every event are exactly what a study needs to ask
whether the folklore survives a control.

STATE, NOT ALERTS
-----------------
A behaviour is tracked as a streak of consecutive five-minute buckets and only
reported when it changes state - watching, developing, confirmed, weakening,
ended. One event per transition, so a call-writing episode that runs for ninety
minutes produces four lines rather than eighteen.

NOTHING IS PERSISTED BUT THE SNAPSHOTS
--------------------------------------
Every function here is pure. The timeline is recomputed from chain_snapshots on
each request, which means a bug fixed tomorrow repairs every day already
recorded, and a threshold changed tomorrow does not leave the database holding
two incompatible generations of the same label.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.positioning.max_pain import max_pain as _max_pain

# --- thresholds ----------------------------------------------------------
# Every number below was chosen, not measured. They are gathered here rather
# than scattered through the detectors so that the day someone measures them,
# there is one place to change and one place to look for what the old values
# were. Treat them as the arbitrary part of this agent.

BUCKET_MINUTES = 5
FOCUS_HALF_WIDTH = 5         # strikes either side of ATM that a reader can hold
SUMMARY_BUCKETS = 3          # a story covers 15 minutes, not 5
MIN_OI_CHANGE_PCT = 3.0      # below this, open interest is considered unchanged
MIN_PREMIUM_CHANGE_PCT = 2.0  # below this, premium is considered flat
MIN_OI_SHARE = 0.20          # ignore strikes under 20% of the busiest on that side
CONFIRM_BUCKETS = 3          # streak length before a behaviour counts as strong
SHIFT_HOLD_BUCKETS = 2       # a migrated support/resistance must hold this long
IV_MOVE_PCT = 5.0            # ATM IV move over 30 minutes worth reporting
IV_SPIKE_PCT = 10.0          # an IV move this large does not wait for the cycle
IV_LOOKBACK_BUCKETS = 6
QUIET_UNTIL = "09:25"        # opening auction noise; no behaviour before this

# Collection stays wide and analysis stays narrow, deliberately. Snapshots keep
# ATM +/-15 because storage is cheap and an unrecorded strike is gone forever,
# while detection reads only ATM +/-5 because thirty-one strikes of five-minute
# wobble is not something a person can read during a session. Widening the
# analysis later costs one constant; widening the collection later costs the
# data we never took.

STATES = ("started", "growing", "strong", "weakening", "finished")

# What the reader is shown. The five states are how the streak is TRACKED;
# three words are what a person needs during a session. Whether a behaviour is
# on its first bucket or its fourth changes nothing about where to look, and a
# label that changes while the thing it describes does not is noise wearing a
# state machine's clothes. The states stay in the payload for the studies this
# page exists to feed - they are simply not what the page says.
SHOWN = {"started": "active", "growing": "active", "strong": "active",
         "weakening": "fading", "finished": "finished"}

SEVERITY = {"started": "INFO", "growing": "WATCH", "strong": "IMPORTANT",
            "weakening": "WATCH", "finished": "INFO"}


# --- the folklore, quarantined -------------------------------------------
# Each entry is (what participants are doing, what it is conventionally taken to
# mean, what to watch next). None of it has been tested. It is phrased as a
# description of participants rather than a description of the market, because
# "call writers are adding" is checkable and "resistance is strengthening" is a
# forecast.

BEHAVIOUR_TEXT: dict[tuple[str, str], tuple[str, str, str]] = {
    ("ce", "writing"): (
        "Call writers are adding positions.",
        "Sellers are accepting the obligation above this strike while premium "
        "falls, which is conventionally read as writers being comfortable there.",
        "Whether price can trade above the strike and whether those writers stay.",
    ),
    ("ce", "short_covering"): (
        "Call writers are buying back positions.",
        "Open interest falling while premium rises is conventionally read as "
        "sellers closing rather than buyers arriving.",
        "Whether the strike keeps losing open interest or writers return.",
    ),
    ("ce", "long_buildup"): (
        "Call buyers are adding positions.",
        "Open interest and premium rising together is conventionally read as "
        "fresh buying rather than writing.",
        "Whether premium holds if the index stalls.",
    ),
    ("ce", "unwinding"): (
        "Call buyers are closing positions.",
        "Open interest and premium falling together is conventionally read as "
        "long holders leaving.",
        "Whether writers step in to replace them.",
    ),
    ("pe", "writing"): (
        "Put writers are adding positions.",
        "Sellers are accepting the obligation below this strike while premium "
        "falls, which is conventionally read as writers defending it.",
        "Whether price holds above the strike and whether those writers stay.",
    ),
    ("pe", "short_covering"): (
        "Put writers are buying back positions.",
        "Open interest falling while premium rises is conventionally read as "
        "put sellers closing, often in a hurry.",
        "Whether the strike keeps losing open interest.",
    ),
    ("pe", "long_buildup"): (
        "Put buyers are adding positions.",
        "Open interest and premium rising together is conventionally read as "
        "fresh downside hedging or speculation.",
        "Whether premium holds if the index stalls.",
    ),
    ("pe", "unwinding"): (
        "Put buyers are closing positions.",
        "Open interest and premium falling together is conventionally read as "
        "hedges being lifted.",
        "Whether writers step in to replace them.",
    ),
}

CHAIN_TEXT: dict[str, tuple[str, str, str]] = {
    "support_shift": (
        "The heaviest put open interest moved to a different strike.",
        "Writers relocating the strike they are willing to defend. Note that "
        "this repository has measured price behaviour at put-OI support: once "
        "price reached it, max pain was reached on fewer than three sessions "
        "in ten, and an equal distance the other way on seven.",
        "Whether the new strike keeps accumulating or the old one recovers.",
    ),
    "resistance_shift": (
        "The heaviest call open interest moved to a different strike.",
        "Writers relocating the strike they are willing to sell against. The "
        "one study run here found no measurable price behaviour at call-OI "
        "resistance in either direction.",
        "Whether the new strike keeps accumulating.",
    ),
    "iv_expansion": (
        "At-the-money implied volatility is rising.",
        "Options are being repriced for a wider outcome. Direction is not "
        "implied by this and never has been.",
        "Whether realised range follows or premium decays back.",
    ),
    "iv_crush": (
        "At-the-money implied volatility is falling.",
        "Options are being repriced for a narrower outcome, which erodes long "
        "premium regardless of direction.",
        "Whether the index range narrows to match.",
    ),
    "range_building": (
        "Call writers and put writers are both adding at the same time.",
        "Writers on both sides are collecting premium, conventionally read as "
        "an expectation that price stays between them.",
        "Whether either side starts covering, which would break the symmetry.",
    ),
}


def _pct(new: float, old: float) -> float | None:
    """Percent change, or None when the base is too small to divide by."""
    if old is None or new is None or abs(old) < 1e-9:
        return None
    return 100.0 * (new - old) / abs(old)


def buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reshape flat snapshot rows into ordered five-minute buckets."""
    grouped: dict[str, dict[float, dict]] = defaultdict(dict)
    spots: dict[str, float] = {}
    for r in rows:
        grouped[r["captured_at"]][float(r["strike"])] = r
        spots[r["captured_at"]] = float(r["spot"])
    return [{"at": at, "spot": spots[at], "strikes": grouped[at]}
            for at in sorted(grouped)]


def _quadrant(d_oi: float | None, d_ltp: float | None) -> str | None:
    """The four standard readings of open interest against premium.

    Open interest alone cannot tell a buyer from a seller - every contract has
    one of each. It is the direction of premium alongside it that is taken to
    separate them, which is why a flat premium yields no behaviour here rather
    than a guess.
    """
    if d_oi is None or d_ltp is None:
        return None
    if abs(d_oi) < MIN_OI_CHANGE_PCT or abs(d_ltp) < MIN_PREMIUM_CHANGE_PCT:
        return None
    if d_oi > 0:
        return "long_buildup" if d_ltp > 0 else "writing"
    return "short_covering" if d_ltp > 0 else "unwinding"


def _near(bucket: dict) -> set[float]:
    """The strikes close enough to the index to be worth a reader's attention.

    Heaviest open interest across the whole chain is not the level a trader
    means by support. NIFTY's round hundreds soak up open interest wherever
    price happens to be - a live read with the index at 24395 put the heaviest
    call at 25000 and the heaviest put at 24000, some 600 and 400 points away.
    Everything downstream of here reads the near window instead.

    The window is a distance, not a count of the nearest N. Counting would keep
    stretching outward on a thin chain until it swallowed exactly the far
    strikes this exists to exclude.
    """
    if not bucket["strikes"]:
        return set()
    atm = min(bucket["strikes"], key=lambda k: abs(k - bucket["spot"]))
    reach = FOCUS_HALF_WIDTH * _step(bucket)
    return {k for k in bucket["strikes"] if abs(k - atm) <= reach}


def _focus_strikes(bucket: dict, side: str) -> set[float]:
    """Strikes carrying enough open interest on this side to be worth watching.

    Without this the monitor reports thirty-one strikes of five-minute wobble.
    The busiest strike sets the bar and everything under a fifth of it is noise
    by construction, not by opinion.
    """
    col = f"{side}_oi"
    oi = {k: float(bucket["strikes"][k][col] or 0.0) for k in _near(bucket)}
    top = max(oi.values(), default=0.0)
    if top <= 0:
        return set()
    return {k for k, v in oi.items() if v >= MIN_OI_SHARE * top}


def _dominant(bucket: dict, side: str) -> float | None:
    """Heaviest open interest on this side, within the near window only."""
    col = f"{side}_oi"
    near = _near(bucket)
    if not near:
        return None
    return max(near, key=lambda k: float(bucket["strikes"][k][col] or 0.0))


def _atm_iv(bucket: dict) -> float | None:
    if not bucket["strikes"]:
        return None
    atm = min(bucket["strikes"], key=lambda k: abs(k - bucket["spot"]))
    row = bucket["strikes"][atm]
    ivs = [float(row[c]) for c in ("ce_iv", "pe_iv") if row[c]]
    return sum(ivs) / len(ivs) if ivs else None


def _transition(prev_state: str | None, streak: int) -> str | None:
    """Map a streak length onto a state, returning None when nothing changed."""
    if streak >= CONFIRM_BUCKETS:
        new = "strong"
    elif streak == 2:
        new = "growing"
    elif streak == 1:
        new = "started"
    else:
        new = "finished" if prev_state in ("weakening",) else "weakening"
    if new == prev_state:
        return None
    if new in ("weakening", "finished") and prev_state in (None, "finished"):
        return None
    return new


def _event(at: str, spot: float, key: str, label: str, state: str,
           evidence: list[str], text: tuple[str, str, str],
           strike: float | None = None, moment: bool = False) -> dict:
    """One line of the timeline.

    `moment` marks a thing that happened rather than a thing that is going on.
    A support strike migrating at 09:40 is over the instant it is reported; an
    episode of call writing is not. Only episodes can be currently active, which
    is what stops a morning IV expansion and an afternoon IV crush from sitting
    in the same list contradicting each other at 15:15.
    """
    what, meaning, watch = text
    return {
        "at": at[11:16],
        "timestamp": at,
        "spot": round(spot, 2),
        "key": key,
        "behaviour": label,
        "strike": strike,
        "state": state,
        "shown": SHOWN[state],
        "moment": moment,
        "severity": SEVERITY[state],
        "evidence": evidence,
        "what_changed": what,
        "commonly_read_as": meaning,
        "watch_next": watch,
        "tested": False,
    }


def _strike_events(seq: list[dict]) -> list[dict]:
    """Per-strike writer behaviour, tracked as streaks across buckets."""
    states: dict[str, str] = {}
    streaks: dict[str, int] = defaultdict(int)
    events: list[dict] = []

    for i in range(1, len(seq)):
        cur, prev = seq[i], seq[i - 1]
        if cur["at"][11:16] < QUIET_UNTIL:
            continue
        live: set[str] = set()

        for side in ("ce", "pe"):
            for strike in _focus_strikes(cur, side):
                if strike not in prev["strikes"]:
                    continue
                now, was = cur["strikes"][strike], prev["strikes"][strike]
                d_oi = _pct(float(now[f"{side}_oi"] or 0), float(was[f"{side}_oi"] or 0))
                d_ltp = _pct(float(now[f"{side}_ltp"] or 0), float(was[f"{side}_ltp"] or 0))
                kind = _quadrant(d_oi, d_ltp)
                if kind is None:
                    continue
                key = f"{side}:{kind}:{strike:.0f}"
                live.add(key)
                streaks[key] += 1
                state = _transition(states.get(key), streaks[key])
                if state:
                    states[key] = state
                    events.append(_event(
                        cur["at"], cur["spot"], key,
                        f"{side.upper()} {kind.replace('_', ' ')}", state,
                        [f"{strike:.0f} {side.upper()} open interest {d_oi:+.1f}%",
                         f"premium {d_ltp:+.1f}%",
                         f"index {cur['spot']:.2f}"],
                        BEHAVIOUR_TEXT[(side, kind)], strike))

        for key, was_state in list(states.items()):
            # Every tracked behaviour that is not currently live decays, not
            # just the ones with a live streak counter. Keying this off the
            # streak meant a behaviour reached `weakening` and stopped there
            # forever, so nothing ever ended and the active list only grew.
            if key in live or was_state == "finished":
                continue
            streaks[key] = 0
            state = _transition(was_state, 0)
            if state:
                states[key] = state
                side, kind, strike = key.split(":")
                events.append(_event(
                    cur["at"], cur["spot"], key,
                    f"{side.upper()} {kind.replace('_', ' ')}", state,
                    [f"{strike} {side.upper()} no longer moving"],
                    BEHAVIOUR_TEXT[(side, kind)], float(strike)))
    return events


def _chain_events(seq: list[dict]) -> list[dict]:
    """Behaviours of the chain as a whole rather than of one strike."""
    events: list[dict] = []
    held: dict[str, tuple[float | None, int]] = {"pe": (None, 0), "ce": (None, 0)}
    reported = {"pe": _dominant(seq[0], "pe") if seq else None,
                "ce": _dominant(seq[0], "ce") if seq else None}
    # Implied volatility is measured against the level at which it was last
    # reported, not against a rolling window. A rolling window re-reports the
    # same drift in every bucket it remains visible in - twelve buckets of one
    # slow climb became twelve identical lines, which is precisely the noise
    # the state machine exists to prevent everywhere else.
    anchor: tuple[int, float] | None = None

    for i in range(1, len(seq)):
        cur = seq[i]
        if cur["at"][11:16] < QUIET_UNTIL:
            continue

        for side, name in (("pe", "support_shift"), ("ce", "resistance_shift")):
            top = _dominant(cur, side)
            candidate, count = held[side]
            count = count + 1 if top == candidate else 1
            held[side] = (top, count)
            if (top is not None and top != reported[side]
                    and count >= SHIFT_HOLD_BUCKETS):
                direction = "higher" if top > (reported[side] or top) else "lower"
                events.append(_event(
                    cur["at"], cur["spot"], name, name.replace("_", " "),
                    "strong",
                    [f"heaviest {side.upper()} open interest {reported[side]:.0f}"
                     f" -> {top:.0f} ({direction})",
                     f"held for {count} buckets"],
                    CHAIN_TEXT[name], moment=True))
                reported[side] = top

        now_iv = _atm_iv(cur)
        if now_iv:
            if anchor is None:
                anchor = (i, now_iv)
            else:
                base_i, base_iv = anchor
                move = _pct(now_iv, base_iv)
                mins = (i - base_i) * BUCKET_MINUTES
                settled = (i - base_i) >= 2 or (move or 0) >= IV_SPIKE_PCT
                if move is not None and abs(move) >= IV_MOVE_PCT and settled:
                    name = "iv_expansion" if move > 0 else "iv_crush"
                    events.append(_event(
                        cur["at"], cur["spot"], name, name.replace("_", " "),
                        "strong",
                        [f"ATM implied volatility {base_iv:.1f} -> {now_iv:.1f}"
                         f" ({move:+.1f}% over {mins} minutes)"],
                        CHAIN_TEXT[name], moment=True))
                    anchor = (i, now_iv)
    return events


def _range_building(events: list[dict]) -> list[dict]:
    """Both sides writing in the same bucket - the one composite kept for now.

    The specification listed several composites. Only this one is implemented,
    because it is the only combination whose components were both confirmed in
    the same bucket by construction; the others require asserting that separate
    behaviours minutes apart belong to one story, which is a claim, not a join.
    """
    by_bucket: dict[str, set[str]] = defaultdict(set)
    spots: dict[str, float] = {}
    for e in events:
        if e["state"] == "strong" and e["key"].startswith(("ce:writing", "pe:writing")):
            by_bucket[e["timestamp"]].add(e["key"].split(":")[0])
            spots[e["timestamp"]] = e["spot"]
    return [
        _event(at, spots[at], "range_building", "range building", "strong",
               ["call writing and put writing both confirmed in the same bucket"],
               CHAIN_TEXT["range_building"], moment=True)
        for at, sides in sorted(by_bucket.items()) if len(sides) == 2
    ]


def summarise(bucket: dict) -> dict[str, Any]:
    """The headline numbers for one bucket. Descriptive only."""
    ce_total = sum(float(r["ce_oi"] or 0) for r in bucket["strikes"].values())
    pe_total = sum(float(r["pe_oi"] or 0) for r in bucket["strikes"].values())
    ladder = {k: {"ce": float(v["ce_oi"] or 0), "pe": float(v["pe_oi"] or 0)}
              for k, v in bucket["strikes"].items()}
    return {
        "at": bucket["at"],
        "spot": round(bucket["spot"], 2),
        "pcr": round(pe_total / ce_total, 3) if ce_total else None,
        "atm_iv": round(_atm_iv(bucket), 2) if _atm_iv(bucket) else None,
        "heaviest_call": _dominant(bucket, "ce"),
        "heaviest_put": _dominant(bucket, "pe"),
        "max_pain": _max_pain(ladder, bucket["spot"], half_width=10),
        "strikes": len(bucket["strikes"]),
    }


PHRASE: dict[tuple[str, str], str] = {
    ("pe", "writing"): "put writers added at {k:.0f}",
    ("pe", "short_covering"): "put writers bought back at {k:.0f}",
    ("pe", "long_buildup"): "put buyers added at {k:.0f}",
    ("pe", "unwinding"): "put buyers closed at {k:.0f}",
    ("ce", "writing"): "call writers added at {k:.0f}",
    ("ce", "short_covering"): "call writers bought back at {k:.0f}",
    ("ce", "long_buildup"): "call buyers added at {k:.0f}",
    ("ce", "unwinding"): "call buyers closed at {k:.0f}",
}

# The same eight behaviours named as ONGOING things rather than as events, for
# the sentences that say a behaviour stopped, is still running, or is worth
# keeping an eye on. "Call writing at 24600 has ended" reads; "call writers
# added at 24600 has ended" does not.
NOUN: dict[tuple[str, str], str] = {
    ("pe", "writing"): "put writing at {k:.0f}",
    ("pe", "short_covering"): "put short covering at {k:.0f}",
    ("pe", "long_buildup"): "put buying at {k:.0f}",
    ("pe", "unwinding"): "put unwinding at {k:.0f}",
    ("ce", "writing"): "call writing at {k:.0f}",
    ("ce", "short_covering"): "call short covering at {k:.0f}",
    ("ce", "long_buildup"): "call buying at {k:.0f}",
    ("ce", "unwinding"): "call unwinding at {k:.0f}",
}


def _say(key: str, table: dict[tuple[str, str], str]) -> str:
    side, kind, strike = key.split(":")
    return table[(side, kind)].format(k=float(strike))


def _step(bucket: dict) -> float:
    ks = sorted(bucket["strikes"])
    gaps = [b - a for a, b in zip(ks, ks[1:]) if b > a]
    return min(gaps) if gaps else 50.0


def _oi_total(bucket: dict, side: str) -> float:
    col = f"{side}_oi"
    return sum(float(bucket["strikes"][k][col] or 0.0) for k in _near(bucket))


def _oi_at(bucket: dict, strike: float | None, side: str) -> float | None:
    if strike is None or strike not in bucket["strikes"]:
        return None
    return float(bucket["strikes"][strike][f"{side}_oi"] or 0.0)


def scoreboard(seq: list[dict], lookback: int = IV_LOOKBACK_BUCKETS) -> list[dict]:
    """Current positioning at a glance. Every row is a measured quantity.

    There is deliberately no row combining these into a lean, and no row called
    "strength". A bar here is a percentage change that was computed, drawn to
    scale - it is the number, not an opinion about the number. The distinction
    matters because a bar chart looks like evidence even when it is a guess, and
    the one thing this repository has actually measured about call-OI resistance
    is that it carried no information at all.
    """
    cur = seq[-1]
    j = max(0, len(seq) - 1 - lookback)
    ref = seq[j]
    mins = (len(seq) - 1 - j) * BUCKET_MINUTES
    rows: list[dict] = []

    for side, name in (("pe", "Support"), ("ce", "Resistance")):
        strike = _dominant(cur, side)
        now, was = _oi_at(cur, strike, side), _oi_at(ref, strike, side)
        change = _pct(now, was) if (now is not None and was) else None
        rows.append({
            "name": name,
            "value": f"{strike:.0f}" if strike is not None else "-",
            "measure": f"heaviest {side.upper()} open interest within ATM +/-"
                       f"{FOCUS_HALF_WIDTH}",
            "change_pct": round(change, 1) if change is not None else None,
            "window_min": mins,
        })

    for side, name in (("ce", "Call writers"), ("pe", "Put writers")):
        now, was = _oi_total(cur, side), _oi_total(ref, side)
        change = _pct(now, was) if was else None
        rows.append({
            "name": name,
            "value": f"{now / 1e5:.1f}L" if now else "-",
            "measure": f"total {side.upper()} open interest across the near strikes",
            "change_pct": round(change, 1) if change is not None else None,
            "window_min": mins,
        })

    now_iv, was_iv = _atm_iv(cur), _atm_iv(ref)
    move = _pct(now_iv, was_iv) if (now_iv and was_iv) else None
    rows.append({
        "name": "Volatility",
        "value": f"{now_iv:.1f}" if now_iv else "-",
        "measure": "at-the-money implied volatility",
        "change_pct": round(move, 1) if move is not None else None,
        "window_min": mins,
    })
    return rows


def _importance(strong_new: int, total_new: int, moments: list[dict],
                iv_move: float | None, shift_steps: float) -> str:
    """How much of a change this update represents. Not how much it matters.

    This grades the update against the previous update, which is a statement
    about the data. It is not a ranking of trading relevance, because nothing
    here has earned the right to make one.
    """
    if (iv_move is not None and abs(iv_move) >= IV_SPIKE_PCT) or shift_steps >= 2:
        return "CRITICAL"
    if moments or strong_new >= 3:
        return "HIGH"
    if strong_new >= 1 or total_new >= 2:
        return "MEDIUM"
    return "LOW"


def _changed(a: dict, b: dict, newest: list[dict], moments: list[dict]) -> list[str]:
    """What is new in this window. Arithmetic only."""
    out: list[str] = []
    for e in newest[:4]:
        line = _say(e["key"], PHRASE)
        out.append(line[0].upper() + line[1:] + ".")
    for m in moments:
        line = m["evidence"][0].replace("->", "moved to")
        out.append(line[0].upper() + line[1:] + ".")
    if not out:
        out.append("No new positioning behaviour cleared the noise filters.")
    out.append(f"The index moved from {a['spot']:.2f} to {b['spot']:.2f}.")
    return out


def _stopped(gone: list[dict]) -> list[str]:
    """What was running and is not any more.

    Kept apart from what is new because they are opposite facts, and a reader
    scanning one paragraph for both will find neither.
    """
    return [_say(e["key"], NOUN).capitalize() + " has ended." for e in gone[:4]]


def _continuing(a: dict, b: dict, carried: list[str], moments: list[dict],
                iv_move: float | None) -> list[str]:
    """What was already running and still is - the background of the window."""
    out = [_say(k, NOUN).capitalize() + " continues." for k in carried[:4]]

    sup, res = _dominant(b, "pe"), _dominant(b, "ce")
    shifted = any(m["key"].endswith("_shift") for m in moments)
    if sup is not None and res is not None and not shifted:
        if sup == _dominant(a, "pe") and res == _dominant(a, "ce"):
            out.append(f"Heaviest put open interest still {sup:.0f}, heaviest "
                       f"call still {res:.0f}.")

    iv = _atm_iv(b)
    if iv is not None and not any(m["key"].startswith("iv_") for m in moments):
        if iv_move is None or abs(iv_move) < IV_MOVE_PCT:
            out.append(f"At-the-money implied volatility {iv:.1f}, little changed.")
    return out


def _observe(b: dict, newest: list[dict], carried: list[str],
             moments: list[dict], frm: str) -> list[str]:
    """Where to look next. Deliberately not what to do next.

    Every line is a QUESTION about something already on the page, phrased so
    that neither answer is the preferred one. "Whether call writing at 24600
    continues or reverses" points a reader at a strike; "resistance should
    hold" tells them what to think about it. The difference is the entire
    remit of this agent, so these are generated from live behaviour rather
    than written as advice with the verb softened.
    """
    out: list[str] = []
    seen: set[str] = set()

    for key in [e["key"] for e in newest] + list(carried):
        if key in seen:
            continue
        seen.add(key)
        out.append(f"Whether {_say(key, NOUN)} continues or reverses.")
        if len(out) >= 3:
            break

    for m in moments:
        if m["key"].endswith("_shift") and m["strike"] is not None:
            side = "put" if m["key"].startswith("support") else "call"
            out.append(f"Whether the heaviest {side} open interest stays at "
                       f"{m['strike']:.0f} or moves back.")
        elif m["key"].startswith("iv_"):
            out.append("Whether at-the-money implied volatility keeps moving "
                       "or settles.")

    # The strike the index is closest to being tested against. "Trades through"
    # names no direction: it is the same sentence whichever side price is on.
    heavy = [(s, "call") for s in [_dominant(b, "ce")] if s is not None]
    heavy += [(s, "put") for s in [_dominant(b, "pe")] if s is not None]
    if heavy:
        strike, side = min(heavy, key=lambda p: abs(p[0] - b["spot"]))
        out.append(f"Whether the index trades through {strike:.0f}, where "
                   f"{side} open interest is heaviest.")

    if not newest and not carried and not moments:
        out.append(f"Whether anything clears the noise filters at all - the "
                   f"chain has been quiet since {frm}.")
    return out[:5]


def stories(seq: list[dict], events: list[dict]) -> list[dict]:
    """Fifteen-minute market updates - the primary output of this agent.

    Detection runs every five minutes and loses nothing. Reporting runs every
    fifteen, because a person reading a live session cannot absorb a line every
    five minutes and will stop reading, which is the same as having no monitor.
    Updates graded LOW carry no new behaviour and are meant to be collapsed.

    Every update answers the same four questions in the same order - what
    changed, what stopped, what is continuing, what to observe next - because a
    live feed whose shape moves with its content cannot be skimmed, and skimming
    is the only way anyone reads one of these during a session.
    """
    if len(seq) <= SUMMARY_BUCKETS:
        return []

    bounds: list[tuple[int, int]] = []
    end = len(seq) - 1
    while end - SUMMARY_BUCKETS >= 0:
        bounds.append((end - SUMMARY_BUCKETS, end))
        end -= SUMMARY_BUCKETS
    bounds.reverse()

    out: list[dict] = []
    for i, j in bounds:
        a, b = seq[i], seq[j]
        window = [e for e in events if a["at"] < e["timestamp"] <= b["at"]]

        latest: dict[str, dict] = {}
        for e in window:
            if not e["moment"] and e["state"] in ("started", "growing", "strong"):
                latest[e["key"]] = e
        newest = sorted(latest.values(), key=lambda e: e["timestamp"])

        gone = {e["key"]: e for e in window
                if not e["moment"] and e["state"] == "finished"}
        moments = list({e["key"]: e for e in window if e["moment"]}.values())

        # Behaviours running at the end of the window that were not reported
        # inside it. Without this the feed loses everything that started
        # earlier and is still going, which is most of what a reader arriving
        # mid-session needs.
        live: dict[str, str] = {}
        for e in events:
            if e["moment"] or e["timestamp"] > b["at"]:
                continue
            if e["state"] in ("started", "growing", "strong"):
                live[e["key"]] = e["state"]
            else:
                live.pop(e["key"], None)
        carried = [k for k in live if k not in latest]

        iv_a, iv_b = _atm_iv(a), _atm_iv(b)
        iv_move = _pct(iv_b, iv_a) if (iv_a and iv_b) else None

        step = _step(b)
        shift_steps = 0.0
        for side in ("pe", "ce"):
            s0, s1 = _dominant(a, side), _dominant(b, side)
            if s0 is not None and s1 is not None:
                shift_steps = max(shift_steps, abs(s1 - s0) / step)

        tally: dict[str, int] = defaultdict(int)
        for e in newest:
            side, kind, _ = e["key"].split(":")
            tally[f"{side.upper()} {kind.replace('_', ' ')}"] += 1

        frm = a["at"][11:16]
        changed = _changed(a, b, newest, moments)
        stopped = _stopped(list(gone.values()))
        going = _continuing(a, b, carried, moments, iv_move)
        observe = _observe(b, newest, carried, moments, frm)

        out.append({
            "at": b["at"][11:16],
            "from": frm,
            "timestamp": b["at"],
            "spot": round(b["spot"], 2),
            "importance": _importance(
                sum(1 for e in newest if e["state"] == "strong"),
                len(newest), moments, iv_move, shift_steps),
            "changed": changed,
            "stopped": stopped,
            "continuing": going,
            "observe": observe,
            "text": " ".join(changed + stopped + going),
            "tally": dict(tally),
            "new_keys": [e["key"] for e in newest],
            "ended_keys": list(gone),
            "carried_keys": carried,
            "tested": False,
        })
    return out


def interpret(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Everything the monitor knows about one session, from snapshots alone."""
    seq = buckets(rows)
    if not seq:
        return {"buckets": 0, "timeline": [], "latest": None, "active": [],
                "scoreboard": [], "stories": [], "story": None}

    events = _strike_events(seq) + _chain_events(seq)
    events += _range_building(events)
    events.sort(key=lambda e: (e["timestamp"], e["key"]))

    # Only episodes can be active. A moment is over as soon as it is reported.
    active: dict[str, dict] = {}
    for e in events:
        if e["moment"]:
            continue
        if e["state"] in ("started", "growing", "strong"):
            active[e["key"]] = e
        else:
            active.pop(e["key"], None)

    told = stories(seq, events)
    return {
        "buckets": len(seq),
        "first_at": seq[0]["at"],
        "last_at": seq[-1]["at"],
        "latest": summarise(seq[-1]),
        "opening": summarise(seq[0]),
        "scoreboard": scoreboard(seq),
        "story": told[-1] if told else None,
        "stories": told,
        "timeline": events,
        "active": sorted(active.values(), key=lambda e: e["timestamp"]),
    }
