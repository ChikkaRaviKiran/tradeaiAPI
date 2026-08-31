"""Max pain over an option-interest ladder.

Extracted from the AgenticTrading max-pain study so the positioning agent and
view can share one implementation without dragging the research CLI along.
"""

from __future__ import annotations


def max_pain(ladder: dict[float, dict[str, float]], spot: float,
             half_width: int) -> float | None:
    """Argmin over total intrinsic value paid out, on a window around ATM.

    Only strikes inside the window contribute AND only they are candidates, so
    the number this returns is exactly as truncated as the window it was given.
    That is the whole point of calling it at several widths.
    """
    strikes = sorted(ladder)
    if not strikes:
        return None
    atm = min(strikes, key=lambda k: abs(k - spot))
    i = strikes.index(atm)
    window = strikes[max(0, i - half_width): i + half_width + 1]
    if len(window) < 3:
        return None

    best, best_pain = None, None
    for candidate in window:
        pain = 0.0
        for k in window:
            oi = ladder[k]
            pain += oi["ce"] * max(0.0, candidate - k)
            pain += oi["pe"] * max(0.0, k - candidate)
        if best_pain is None or pain < best_pain:
            best, best_pain = candidate, pain
    return best
