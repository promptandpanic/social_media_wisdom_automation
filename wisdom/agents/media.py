"""
Media agent — image generation, composition, judge, and retry loop.

Graph:
  generate_image → compose → judge → [accept | retry (max N) | use_best]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from langgraph.graph import END, StateGraph

import wisdom.config as cfg
from wisdom import providers
from wisdom.schemas import PipelineState

logger = logging.getLogger(__name__)

_RETRY_SUFFIXES = [
    "",
    "Try a completely different angle, time of day, or compositional approach.",
    "Go bolder — extreme contrast, dramatic scale, or stark minimalism.",
]

_THEME_PROMPT_PREFIXES: dict[str, str] = {
    "womenpower": "A woman — ",
}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def generate_image(state: PipelineState) -> PipelineState:
    brief = state.get("brief")
    theme_key = state["theme_key"]
    attempt = state.get("design_attempt", 0) + 1
    max_attempts = cfg.app().get("design_attempts", 3)
    logger.info(f"Image attempt {attempt}/{max_attempts}…")

    if state.get("offline"):
        from wisdom.providers.image import GradientFallback

        logger.info("  Offline mode: using gradient fallback image")
        image_bytes = GradientFallback().generate("offline fallback")
        provider_name = "gradient"
    else:
        base_prompt = (
            brief.image_prompt
            if brief
            else f"Beautiful inspirational {theme_key} image, 9:16."
        )
        prefix = _THEME_PROMPT_PREFIXES.get(theme_key, "")
        if prefix and not base_prompt.lower().startswith(
            prefix.lower().strip(" —").strip()
        ):
            base_prompt = f"{prefix}{base_prompt}"
        suffix = _RETRY_SUFFIXES[min(attempt - 1, len(_RETRY_SUFFIXES) - 1)]
        prompt = f"{base_prompt} {suffix}".strip() if suffix else base_prompt

        exclude = state.get("failed_providers", [])
        image_bytes, provider_name = providers.image.generate(prompt, exclude=exclude)
        if "model_usage" not in state:
            state["model_usage"] = {}
        state["model_usage"]["Image Generation"] = provider_name

    return {
        **state,
        "image_bytes": image_bytes,
        "design_attempt": attempt,
        "current_provider": provider_name,
    }


def compose(state: PipelineState) -> PipelineState:
    from wisdom.composers.card import compose_image

    image_bytes = state.get("image_bytes", b"")
    quote = state.get("quote")
    brief = state.get("brief")
    composed = compose_image(image_bytes, quote, brief)
    logger.info(f"  Composed ({len(composed) // 1024} KB)")
    return {**state, "composed_image": composed}


def judge(state: PipelineState) -> PipelineState:
    composed = state.get("composed_image", b"")
    quote = state.get("quote")
    score, accepted, hard_gate, issues = _judge_image(composed, quote, state)

    candidate = {
        "image": state.get("image_bytes"),
        "composed": composed,
        "score": score,
        "hard_gate": hard_gate,
        "provider": state.get("current_provider"),
    }

    best = state.get("best_state")
    if best is None:
        best = candidate
    elif best.get("hard_gate") and not hard_gate:
        best = candidate
    elif best.get("hard_gate") == hard_gate and score > best.get("score", 0):
        best = candidate

    logger.info(f"  Judge: {score}/10 {'✓ accepted' if accepted else '✗ rejected'}")
    if issues:
        logger.info(f"  Judge reason: {issues}")

    new_state = {
        **state,
        "best_score": score,
        "_accepted": accepted,
        "_hard_gate": hard_gate,
        "best_state": best,
    }

    if not accepted:
        failed = list(state.get("failed_providers", []))
        current = state.get("current_provider")
        if current and current not in failed and current != "gradient":
            failed.append(current)
            logger.info(
                f"  Provider '{current}' blacklisted for this run due to poor quality/gibberish"
            )
        new_state["failed_providers"] = failed

    return new_state


def use_best(state: PipelineState) -> PipelineState:
    """After retries exhausted, promote best_state to composed_image."""
    best = state.get("best_state", {})
    if best.get("hard_gate"):
        logger.warning("All attempts failed hard gates — using gradient fallback")
        from wisdom.providers.image import GradientFallback
        from wisdom.composers.card import compose_image

        raw = GradientFallback().generate("")
        composed = compose_image(raw, state.get("quote"), state.get("brief"))
    else:
        composed = best.get("composed", state.get("composed_image", b""))
    return {**state, "composed_image": composed}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_judge(state: PipelineState) -> Literal["accept", "retry", "use_best"]:
    if state.get("_accepted"):
        return "accept"
    attempt = state.get("design_attempt", 0)
    max_attempts = cfg.app().get("design_attempts", 3)
    if attempt < max_attempts:
        return "retry"
    return "use_best"


def _passthrough(state: PipelineState) -> PipelineState:
    return state


# ---------------------------------------------------------------------------
# Image judge (Gemini vision)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are judging a composed inspirational image for a global social media account.

Quote: "{text}"

Hard gates (automatic rejection — score 0):
  - Obvious gibberish text artifacts, signatures, or watermarks in the background
  - Anatomical anomalies (e.g., 3 legs, extra fingers, distorted eyes) or obvious AI glitches/warped bodies
  - Completely black, corrupted, or "not needed" artifacts
  - Anything that looks like a technical error or ruined reputation

Quality criteria (score 1-10):
  1. Text readability (contrast, font clarity)
  2. Visual impact (would stop scrolling)
  3. Emotional match to the quote
  4. Quote zone is clean and uncluttered
  5. Colour palette is rich and intentional

Return ONLY valid JSON:
{{"score": 7, "accept": true, "hard_gate_failure": false, "issues": ""}}
"""


def _judge_image(image_bytes: bytes, quote, state: PipelineState) -> tuple[int, bool, bool, str]:
    if not image_bytes:
        return 0, False, True, "No image bytes"
    threshold = cfg.app().get("judge_threshold", 7)
    try:
        prompt = _JUDGE_PROMPT.format(text=getattr(quote, "text", "") if quote else "")
        content, provider_info = providers.llm.judge_image(image_bytes, prompt, role="image_judge")
        if "model_usage" not in state:
            state["model_usage"] = {}
        state["model_usage"]["Vision Judge"] = provider_info

        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group())
            score = int(data.get("score", 5))
            hard_gate = bool(data.get("hard_gate_failure", False))
            accepted = bool(data.get("accept", False)) and score >= threshold
            issues = data.get("issues", "")
            return score, accepted, hard_gate, issues
    except Exception as exc:
        logger.warning(f"Judge failed ({exc}) — accepting with score 7")
    return 7, True, False, ""


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def build() -> any:
    g = StateGraph(PipelineState)
    g.add_node("generate_image", generate_image)
    g.add_node("compose", compose)
    g.add_node("judge", judge)
    g.add_node("use_best", use_best)
    g.add_node("accept", _passthrough)

    g.set_entry_point("generate_image")
    g.add_edge("generate_image", "compose")
    g.add_edge("compose", "judge")
    g.add_conditional_edges(
        "judge",
        _route_judge,
        {
            "accept": "accept",
            "retry": "generate_image",
            "use_best": "use_best",
        },
    )
    g.add_edge("accept", END)
    g.add_edge("use_best", END)
    return g.compile()
