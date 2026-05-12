"""
Quote generation agent — LangGraph state machine.

Graph:
  select_mode → generate → validate
                  ↑             ↓ fail (retry < MAX)
                  └──────────── retry
                                ↓ exhausted
                           curated_fallback → END
"""
from __future__ import annotations

import json
import logging
import random
import re
from typing import Literal

from langgraph.graph import END, StateGraph

import wisdom.config as cfg
from wisdom import providers
from wisdom.schemas import PipelineState, Quote

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
MIN_UNIQUENESS = 7


def _clean(text: str) -> str:
    text = text.strip().strip('""\'\'„"«»‹›').strip()
    # Strip markdown bolding (double asterisks)
    text = text.replace("**", "")
    text = re.sub(r'\s*[-—–~]\s*[A-Z][^—–\n]{1,60}$', '', text).strip()
    return text.strip('"""\'').strip()


def _extract_highlight(text: str) -> str:
    parts = [p.strip() for p in re.split(r"[.!?—–]", text) if p.strip()]
    if len(parts) > 1:
        last = parts[-1].split()
        return " ".join(last[:5]) if len(last) > 5 else parts[-1]
    words = text.split()
    return " ".join(words[-5:]) if len(words) > 5 else text


def _parse_quote_json(raw: str) -> dict | None:
    # Strip markdown code fences Gemini often wraps around JSON
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except Exception:
            return None
    try:
        arr = json.loads(m.group())
        return arr[0] if isinstance(arr, list) and arr else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Topic + prompt helpers (delegate to config)
# ---------------------------------------------------------------------------

def _get_topic_block(theme_key: str) -> tuple[str, str]:
    """Returns (topic_block, image_hint)."""
    from wisdom.agents._topic_builder import build_topic_block
    return build_topic_block(theme_key)


def _build_prompt(theme_key: str, mode: str, topic_block: str,
                  max_words: int, recent_quotes: list[str]) -> str:
    from wisdom.agents._prompt_builder import build_quote_prompt
    return build_quote_prompt(theme_key, mode, topic_block, max_words, recent_quotes)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def select_mode(state: PipelineState) -> PipelineState:
    mode = "offline" if state.get("offline") else random.choice(["real_author", "internet_found"])
    logger.info(f"Quote mode: {mode}")
    return {**state, "_quote_mode": mode, "_quote_attempt": 0, "_quote_fallback": None}


def generate(state: PipelineState) -> PipelineState:
    theme_key = state["theme_key"]
    theme = state["theme"]
    mode = state.get("_quote_mode", "internet_found")
    attempt = state.get("_quote_attempt", 0) + 1
    logger.info(f"Quote attempt {attempt}/{MAX_RETRIES}…")

    topic_block, image_hint = _get_topic_block(theme_key)
    max_words = theme.max_words
    recent_quotes = state.get("recent_quotes", [])

    prompt = _build_prompt(theme_key, mode, topic_block, max_words, recent_quotes)
    errors = list(state.get("errors", []))

    try:
        raw = providers.llm.generate(prompt, role="quote_generation")
        data = _parse_quote_json(raw)
        if not data:
            return {**state, "_quote_attempt": attempt, "_raw_quote_data": None,
                    "_image_hint": image_hint, "errors": errors}

        text = _clean(data.get("quote", ""))
        author = (data.get("author") or "").strip()
        uniqueness = int(data.get("uniqueness", 0) or 0)

        if (text and author and len(text.split()) >= 3
                and len(text.split()) <= max_words + 5
                and uniqueness >= MIN_UNIQUENESS):
            
            # Use LLM-selected highlight if available, else fallback to math rule
            hl = data.get("highlight", "").strip()
            if not hl or len(hl.split()) < 2:
                hl = _extract_highlight(text)
                
            quote = Quote(
                text=text, author=author,
                highlight=hl,
                image_hint=image_hint,
                score=uniqueness,
                source=_source_name(mode),
            )
            logger.info(f'  ✓ "{text[:70]}" — {author}')
            return {**state, "_quote_attempt": attempt, "_valid_quote": quote,
                    "_image_hint": image_hint}

        # Below bar — keep as fallback
        fallback = state.get("_quote_fallback")
        if text and author and (not fallback or uniqueness > fallback.get("uniqueness", 0)):
            fallback = {**data, "image_hint": image_hint, "uniqueness": uniqueness}
        return {**state, "_quote_attempt": attempt, "_valid_quote": None,
                "_quote_fallback": fallback, "_image_hint": image_hint}

    except Exception as exc:
        errors.append(f"quote attempt {attempt}: {exc}")
        return {**state, "_quote_attempt": attempt, "_valid_quote": None, "errors": errors}


def curated_fallback(state: PipelineState) -> PipelineState:
    theme_key = state["theme_key"]
    below_bar = state.get("_quote_fallback")
    image_hint = state.get("_image_hint", "")

    # 1. Try below-bar candidate from generation
    if below_bar:
        text = _clean(below_bar.get("quote", ""))
        author = (below_bar.get("author") or "").strip()
        hl = below_bar.get("highlight", "").strip()
        if not hl:
            hl = _extract_highlight(text)
        if text and author:
            logger.warning("Using below-bar candidate as fallback")
            q = Quote(text=text, author=author, highlight=hl,
                      image_hint=image_hint, source="fallback")
            return {**state, "quote": q}

    # 2. Curated pool
    pool = cfg.curated_quotes().get(theme_key, [])
    available = pool
    if available:
        item = random.choice(available)
        q = Quote(text=item["text"], author=item.get("author", "Unknown"),
                  highlight=_extract_highlight(item["text"]),
                  image_hint=image_hint, source="fallback")
        logger.info(f"Curated fallback: \"{q.text[:60]}\"")
        return {**state, "quote": q}

    # 3. Hardcoded emergency
    hardcoded = {"text": "The wound is the place where the light enters you.", "author": "Rumi"}
    q = Quote(text=hardcoded["text"], author=hardcoded["author"],
              highlight="light enters you", image_hint=image_hint, source="fallback")
    return {**state, "quote": q}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_generate(state: PipelineState) -> Literal["done", "retry", "fallback"]:
    if state.get("_valid_quote"):
        return "done"
    attempt = state.get("_quote_attempt", 0)
    if attempt < MAX_RETRIES:
        return "retry"
    return "fallback"


def _set_quote(state: PipelineState) -> PipelineState:
    return {**state, "quote": state["_valid_quote"]}


def _source_name(mode: str) -> str:
    return "real_author" if mode == "real_author" else "internet_found"


def _route_mode(state: PipelineState) -> Literal["generate", "curated_fallback"]:
    if state.get("_quote_mode") == "offline":
        return "curated_fallback"
    return "generate"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build() -> any:
    g = StateGraph(PipelineState)

    g.add_node("select_mode", select_mode)
    g.add_node("generate", generate)
    g.add_node("set_quote", _set_quote)
    g.add_node("curated_fallback", curated_fallback)

    g.set_entry_point("select_mode")
    g.add_conditional_edges("select_mode", _route_mode, {
        "generate": "generate",
        "curated_fallback": "curated_fallback"
    })
    g.add_conditional_edges("generate", _route_generate, {
        "done": "set_quote",
        "retry": "generate",
        "fallback": "curated_fallback",
    })
    g.add_edge("set_quote", END)
    g.add_edge("curated_fallback", END)

    return g.compile()
