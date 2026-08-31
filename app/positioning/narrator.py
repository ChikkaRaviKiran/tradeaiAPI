"""Ollama connection settings and the fabrication guard.

The subset of the AgenticTrading narrator that the positioning writer actually
uses. The writer may only reword observations that were computed elsewhere, so
the one guard that has to travel with it is the one that checks every number in
the rewrite against the numbers it was given.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Configurable because "localhost" means the container itself once this runs
# under Docker. Point it at the host or at an ollama service instead.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")

KEEP_ALIVE = "15m"     # stay resident, so the second call is not a cold load
SEED = 7
TEMPERATURE = 0.0

# Bare small integers are skipped: "3 of the zones", "the first 2 levels". These
# are counts, not claims about the market.
#
# Decimals are NEVER skipped, whatever their size. An earlier version skipped
# everything under 10 and a test caught it waving through an invented "PCR of
# 0.94" - which is the entire class this guard exists for, since almost every
# fabricated option-chain figure (PCR, delta, IV, gamma) is a small decimal.
TRIVIAL_INT_MAX = 10


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for m in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            out.add(round(float(m), 4))
        except ValueError:
            continue
    return out


def _fact_numbers(node: Any, acc: set[float]) -> set[float]:
    """Every number anywhere in the fact sheet, including inside note strings."""
    if isinstance(node, bool):
        return acc
    if isinstance(node, (int, float)):
        acc.add(round(float(node), 4))
    elif isinstance(node, str):
        acc |= _numbers(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            acc |= _numbers(str(k))
            _fact_numbers(v, acc)
    elif isinstance(node, list):
        for v in node:
            _fact_numbers(v, acc)
    return acc


def _text_of(narrative: dict[str, Any]) -> str:
    """All prose in the narrative, including inside factor objects."""
    parts: list[str] = []
    for v in narrative.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.extend(str(x) for x in item.values())
    return " ".join(parts)


def _degroup(text: str) -> str:
    """Join digit groups so "10,000" is one number, not two.

    Without this the tokeniser split it into 10 and 000, and BOTH survived the
    trivial-integer skip - so an invented crore figure passed the guard clean.

    Only three-digit groups are joined, which leaves prose like "1, 2, 3"
    alone. Indian grouping ("2,40,000") is handled by repeating the pass.
    """
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(?<=\d),(?=\d{2,3}(?!\d))", "", text)
    return text


def check_fabrication(narrative: dict[str, Any],
                      facts: dict[str, Any]) -> list[str]:
    """Numbers in the narrative that appear nowhere in the fact sheet.

    A value and its percentage form count as the same number, so a sheet
    carrying 0.4571 permits "45.71%". So does a value and its magnitude: the
    sheet's -28.7 permits "a gap of 28.7 points". Neither is an invented figure,
    and this guard exists to catch INVENTION.
    """
    allowed = _fact_numbers(facts, set())
    allowed |= {round(a * 100, 4) for a in allowed if 0.0 < abs(a) <= 1.0}
    allowed |= {abs(a) for a in allowed}
    bad = []
    for token in re.findall(r"-?\d+(?:\.\d+)?", _degroup(_text_of(narrative))):
        n = round(float(token), 4)
        decimals = len(token.split(".")[1]) if "." in token else 0
        if decimals == 0 and abs(n) <= TRIVIAL_INT_MAX:
            continue
        if n in allowed:
            continue
        # A figure may be quoted to fewer decimals than the sheet carries, so
        # allow half a unit of the LAST DIGIT QUOTED - and no more. A flat
        # tolerance fails badly at small magnitudes.
        tol = 0.5 * (10 ** -decimals)
        if any(abs(n - a) < tol for a in allowed):
            continue
        bad.append(f"{n:g}")
    return sorted(set(bad))
