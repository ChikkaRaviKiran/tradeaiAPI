"""The LLM writes the positioning story; this module is the editor.

WHY THIS EXISTS
---------------
The computed story in `positioning_agent` is correct and reads like a machine,
because it is one - clauses concatenated in a fixed order. A person watching a
live session reads prose or reads nothing.

So the model gets one job, and it is the smallest job that still helps:

    facts -> computed four-part template -> model improves the reading

It is handed the template, not the measurements. It is told to summarize only:
do not infer, do not predict, do not rank, do not advise, do not explain. It
never sees a number it was not already told to reprint, and it is never asked a
question it could answer with an opinion.

WHY THE EDITOR IS THE IMPORTANT HALF
------------------------------------
A model rewriting numbers into prose will, sooner or later, write a number that
was not there. `check_fabrication` is reused from the narrator rather than
rewritten because it already survived the case that matters: an invented "PCR of
0.94" that an earlier, looser guard waved through.

Three rules, all enforced after generation and none negotiable:

  1. Every number in the output must appear in the fact sheet.
  2. No sentence may take a side. "More supportive of higher prices" is a
     forecast with the verb removed, and it is the single phrase this agent is
     most likely to be asked for and must never produce.
  3. No section may grow. A model breaks the "add nothing" rule by being
     helpful, not by being wrong - one extra bullet under "observe next" is the
     likeliest failure, and a request cannot stop it. A count can.

A rejected rewrite is not shown with a warning attached, because a warned
paragraph gets read anyway. It is replaced by the computed template, which was
always correct. The failure mode of this module is therefore "reads stiffly",
never "reads well and is wrong".
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from hashlib import sha256
from typing import Any

import httpx

from app.positioning.narrator import (
    KEEP_ALIVE,
    OLLAMA_URL,
    SEED,
    TEMPERATURE,
    check_fabrication,
)

# qwen2.5:7b-instruct rather than the narrator's gemma3:4b. The first live run
# was decided by measurement, not preference: gemma3:4b wrote a correct
# paragraph, closed the JSON string with a curly quote, and then emitted "47
# words." forty times. The paragraph was fine; the envelope was not.
DEFAULT_MODEL = "qwen2.5:7b-instruct"
TIMEOUT = 240.0
NUM_CTX = 2048

# A 90-word paragraph is roughly 130 tokens. The ceiling is deliberately close
# to it so a model that starts looping runs out of room quickly instead of
# spending two minutes repeating itself.
NUM_PREDICT = 500
REPEAT_PENALTY = 1.15

MAX_CHARS = 1600
MIN_CHARS = 40

# Phrases that state or imply a direction, a recommendation, or a forecast.
# This is deliberately blunt. These have no legitimate use in a paragraph whose
# entire remit is to restate what has already happened.
BANNED = (
    "bullish", "bearish", "supportive of higher", "supportive of lower",
    "more supportive", "upside", "downside", "rally", "sell-off",
    "expect", "likely to", "should ", "you should", "we should",
    "will move", "will rise", "will fall", "poised", "set up for",
    "breakout confirmed", "buy ", "sell the", "go long", "go short",
    "target", "stop loss", "book profit", "take profit",
    "suggests price", "indicates price", "points to a move",
    "strengthening", "weakening resistance", "resistance is weak",
)

SECTIONS = ("changed", "stopped", "continuing", "observe")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {s: {"type": "array", "items": {"type": "string"}}
                   for s in SECTIONS},
    "required": list(SECTIONS),
}

SYSTEM = """You rewrite option-chain observations so they read better. Nothing else.

Summarize only.

Do NOT infer.
Do NOT predict.
Do NOT rank.
Do NOT advise.
Do NOT explain.

HARD RULES:
1. Use ONLY the numbers given. Never introduce, round, or compute a number.
2. Never say or imply where price will go. No "bullish", "bearish", "supportive
   of higher prices", "expect", "likely", or any recommendation.
3. Keep every line in the section it came from. Do not move a fact between
   sections and do not add a line to any section.
4. Return the same number of lines you were given, or fewer if two say the same
   thing. An empty section stays empty.
5. Plain English, short sentences. The "observe" lines stay as questions about
   where to look, never as instructions about what to do.

You are an editor. You have no knowledge of markets and must behave as if you
have none."""

USER_TEMPLATE = """Rewrite the lines in each section. Follow the rules exactly.

Window: {frm} to {at}
Index level: {spot}

{facts}"""


def fact_sheet(story: dict[str, Any]) -> dict[str, Any]:
    """The only material the model is allowed to draw on.

    This is the computed template itself, not the raw measurements. The model
    never sees a number it was not already told to reprint, which makes the
    fabrication guard's allowed set exactly the set of figures on the page.
    Anything outside it was invented.
    """
    return {s: list(story.get(s) or []) for s in SECTIONS}


def check_language(text: str) -> list[str]:
    """Banned phrases present in the rewrite."""
    low = text.lower()
    return [w.strip() for w in BANNED if w in low]


def check_shape(draft: dict[str, list[str]],
                facts: dict[str, list[str]]) -> list[str]:
    """The model may reword a section. It may not grow one.

    Rule 3 of the prompt is the one a model breaks by being helpful rather than
    by being wrong - inventing an extra "observe" line, or promoting something
    from continuing into changed. A request cannot stop that; a count can.
    """
    bad = []
    for s in SECTIONS:
        got, want = draft.get(s) or [], facts.get(s) or []
        if len(got) > len(want):
            bad.append(f"{s} grew from {len(want)} to {len(got)} lines")
        if want and not got:
            bad.append(f"{s} was emptied")
    return bad


# Salvages the sections when the model breaks its own JSON. The first live run
# closed a string with a curly quote and then kept generating, so `json.loads`
# failed and the raw content - good prose plus forty lines of noise - went to
# the guard, which rejected it for a number that was only in the noise.
#
# Salvaging is safe here precisely BECAUSE the guard runs afterwards. Anything
# recovered still has to survive the number, language and shape checks, so a
# lenient parser cannot let a bad rewrite through - only a good one that was
# wrapped badly.
_OBJECT = re.compile(r"\{.*\}", re.S)


def _extract(content: str) -> dict[str, list[str]]:
    data: Any = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = _OBJECT.search(content)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for s in SECTIONS:
        v = data.get(s)
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            out[s] = [str(x).strip() for x in v if str(x).strip()]
    return out


def write(story: dict[str, Any], model: str = DEFAULT_MODEL,
          url: str = OLLAMA_URL) -> dict[str, Any]:
    """Rewrite one story's four sections. Always returns usable text.

    Never raises. A model that is slow, absent, or disobedient degrades this
    page to the wording it already had, which is the whole point of keeping the
    computed template rather than replacing it.
    """
    facts = fact_sheet(story)
    prompt = USER_TEMPLATE.format(
        frm=story.get("from", "?"), at=story.get("at", "?"),
        spot=story.get("spot", "?"),
        facts=json.dumps(facts, indent=2, default=str))

    record: dict[str, Any] = {
        "at": story.get("at"),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt_hash": sha256((SYSTEM + prompt).encode()).hexdigest()[:16],
        "source": "template",
        "sections": facts,
        "template": facts,
        "rejected": [],
        "tested": False,
    }

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{url}/api/chat", json={
                "model": model,
                "stream": False,
                "format": SCHEMA,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": TEMPERATURE, "seed": SEED,
                            "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
                            "repeat_penalty": REPEAT_PENALTY},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
            })
            resp.raise_for_status()
            draft = _extract(resp.json()["message"]["content"])
    except Exception as exc:
        record["rejected"] = [f"{type(exc).__name__}: {exc}"]
        return record

    record["draft"] = draft
    blob = " ".join(line for s in SECTIONS for line in draft.get(s, []))
    reasons: list[str] = []

    if not draft:
        reasons.append("model returned nothing usable")
    elif len(blob) < MIN_CHARS:
        reasons.append("rewrite was empty or truncated")
    if len(blob) > MAX_CHARS:
        reasons.append(f"rewrite ran to {len(blob)} characters")

    reasons += check_shape(draft, facts)

    banned = check_language(blob)
    if banned:
        reasons.append("forecast or instruction language: " + ", ".join(banned))

    invented = check_fabrication({"prose": blob}, facts)
    if invented:
        reasons.append("numbers absent from the facts: " + ", ".join(invented))

    record["rejected"] = reasons
    if not reasons:
        record["source"] = "llm"
        record["sections"] = draft
    return record


# --- the market story ------------------------------------------------------
# The four-section rewrite above serves the detail panel. This one serves the
# card the page opens with, and it is a different job: one short paragraph, not
# four lists. It is kept in the same module so both share one set of guards -
# a second copy of the fabrication check is a second thing to forget to tighten.

STORY_MAX_WORDS = 120
STORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"story": {"type": "string"}},
    "required": ["story"],
}

STORY_SYSTEM = """You rewrite option-chain observations into one short paragraph.

Rules:
Do NOT infer.
Do NOT predict.
Do NOT recommend trades.
Do NOT invent missing facts.
Only improve readability.
Maximum 120 words.

Use ONLY the observations given. Never introduce, round, or compute a number.
Never say or imply where price will go - no "bullish", "bearish", "expect",
"likely", "support should hold". Describe what participants have done and for
how long. You are an editor with no knowledge of markets."""

STORY_TEMPLATE = """Rewrite these observations as one paragraph.

Computed wording:
{lines}

Supporting facts:
{facts}"""


def _story_numbers(view_story: dict[str, Any]) -> dict[str, Any]:
    """Everything the paragraph is allowed to contain a number from."""
    return {"lines": list(view_story.get("lines") or []),
            "facts": list(view_story.get("facts") or [])}


def write_narrative(view_story: dict[str, Any], model: str = DEFAULT_MODEL,
                    url: str = OLLAMA_URL) -> dict[str, Any]:
    """Rewrite the market story. Falls back to the computed lines, never raises.

    The direction view is deliberately NOT passed in. The model is never shown
    the conclusion, so it cannot restate it, soften it, or argue with it - and
    a rewrite that cannot see a direction cannot leak one into prose.
    """
    facts = _story_numbers(view_story)
    computed = " ".join(facts["lines"])
    prompt = STORY_TEMPLATE.format(
        lines="\n".join(f"- {line}" for line in facts["lines"]),
        facts=json.dumps(facts["facts"], indent=2, default=str))

    record: dict[str, Any] = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "prompt_hash": sha256((STORY_SYSTEM + prompt).encode()).hexdigest()[:16],
        "source": "template",
        "story": computed,
        "template": computed,
        "rejected": [],
        "tested": False,
    }
    if not facts["lines"]:
        record["rejected"] = ["nothing computed to rewrite"]
        return record

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.post(f"{url}/api/chat", json={
                "model": model,
                "stream": False,
                "format": STORY_SCHEMA,
                "keep_alive": KEEP_ALIVE,
                "options": {"temperature": TEMPERATURE, "seed": SEED,
                            "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
                            "repeat_penalty": REPEAT_PENALTY},
                "messages": [{"role": "system", "content": STORY_SYSTEM},
                             {"role": "user", "content": prompt}],
            })
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
    except Exception as exc:
        record["rejected"] = [f"{type(exc).__name__}: {exc}"]
        return record

    draft = ""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            draft = str(parsed.get("story") or "").strip()
    except json.JSONDecodeError:
        m = _OBJECT.search(content)
        if m:
            try:
                draft = str(json.loads(m.group(0)).get("story") or "").strip()
            except (json.JSONDecodeError, AttributeError):
                draft = ""

    record["draft"] = draft
    reasons: list[str] = []
    words = len(draft.split())

    if not draft or len(draft) < MIN_CHARS:
        reasons.append("model returned nothing usable")
    if words > STORY_MAX_WORDS:
        reasons.append(f"rewrite ran to {words} words, over {STORY_MAX_WORDS}")

    banned = check_language(draft)
    if banned:
        reasons.append("forecast or instruction language: " + ", ".join(banned))

    invented = check_fabrication({"prose": draft}, facts)
    if invented:
        reasons.append("numbers absent from the facts: " + ", ".join(invented))

    record["rejected"] = reasons
    if not reasons:
        record["source"] = "llm"
        record["story"] = draft
    return record
