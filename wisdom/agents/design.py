"""
Design agent — two-phase LangGraph state machine.

Phase 1 (pick_style): LLM picks the best visual style for the quote.
Phase 2 (generate_brief): LLM writes a vivid, quote-specific image prompt.
Rendering parameters (font, colors, overlay) come from the style's config — not LLM.
"""
from __future__ import annotations

import logging
import re

from langgraph.graph import END, StateGraph

import wisdom.config as cfg
from wisdom import providers
from wisdom.schemas import DesignBrief, Overlay, PipelineState

logger = logging.getLogger(__name__)

_ACCOUNT = "global inspirational quotes account (ages 18–35, large Indian following)"

_VALID_FONTS = frozenset({"playfair", "cormorant", "cinzel", "montserrat", "bebas"})


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _styles_for_theme(theme_key: str, locked: list[str] | None,
                      recent: list[str]) -> list[dict]:
    all_styles = cfg.styles()
    result = []
    for name, s in all_styles.items():
        if locked and name not in locked:
            continue
        if not locked and theme_key not in s.get("categories", []):
            continue
        result.append({"name": name, **s})
    weight_order = {"high": 3, "medium": 2, "low": 1}
    result.sort(key=lambda s: weight_order.get(s.get("weight", "medium"), 2), reverse=True)
    return result


def _picker_prompt(quote_text: str, theme_key: str,
                   styles: list[dict], recent: list[str]) -> str:
    lines = ["Available styles (pick the ONE that fits this quote best):"]
    for s in styles:
        avoid = " [used recently — avoid if possible]" if s["name"] in recent else ""
        star = " ★" if s.get("weight") == "high" else ""
        lines.append(f"  {s['name']}{star} — {s.get('summary', '')}{avoid}")

    return f"""\
You are the creative director for a {_ACCOUNT}.
Pick ONE visual style for this post.

QUOTE: "{quote_text}"
THEME: {theme_key}

{chr(10).join(lines)}

Choose the style whose aesthetic creates the strongest emotional match for this quote.
Favour styles not used recently.

Return ONLY valid JSON: {{"style": "chosen_style_name"}}"""


_IMAGE_PROMPT_TEMPLATE = """\
Write a vivid image generation prompt for this quote and visual style.

QUOTE: "{text}"
STYLE: {style_name}

{style_description}
{image_hint_block}
Write 4–6 rich sentences describing the scene:
  subject → setting → technique/medium → colour palette (use hex values) → lighting → composition

Constraints:
  - Scene must be emotionally specific to THIS quote — not a generic illustration of the theme
  - Leave the centre band completely clear and dark for the text overlay
  - No text, words, signs, logos, watermarks, or clearly recognisable faces
  - 9:16 portrait format

Reply with ONLY the image prompt — plain text, no JSON, no preamble."""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def pick_style(state: PipelineState) -> PipelineState:
    theme_key = state["theme_key"]
    theme = state["theme"]
    quote = state.get("quote")
    recent = state.get("recent_styles", [])

    styles = _styles_for_theme(theme_key, theme.styles, recent)
    if not styles:
        return {**state, "_chosen_style": "dark_academia_classical"}

    try:
        prompt = _picker_prompt(quote.text if quote else "", theme_key, styles, recent)
        raw = providers.llm.generate(prompt, role="style_picker")
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            import json
            style_name = json.loads(m.group()).get("style", "")
            if style_name and style_name in cfg.styles():
                logger.info(f"Style: {style_name}")
                return {**state, "_chosen_style": style_name}
    except Exception as exc:
        logger.warning(f"Style picker failed ({exc}) — using top style")

    fallback = next((s["name"] for s in styles if s["name"] not in recent), styles[0]["name"])
    return {**state, "_chosen_style": fallback}


def generate_brief(state: PipelineState) -> PipelineState:
    theme_key = state["theme_key"]
    quote = state.get("quote")
    style_name = state.get("_chosen_style", "dark_academia_classical")
    style_data = cfg.styles().get(style_name, {})
    style_desc = style_data.get("description", "").strip()
    image_hint = quote.image_hint if quote else ""
    text = quote.text if quote else ""

    try:
        prompt = _IMAGE_PROMPT_TEMPLATE.format(
            text=text,
            style_name=style_name,
            style_description=style_desc,
            image_hint_block=f"ADDITIONAL DIRECTION: {image_hint}\n" if image_hint else "",
        )
        image_prompt = providers.llm.generate(prompt, role="creative_brief").strip()
        if len(image_prompt.split()) >= 20:
            brief = _build_brief(image_prompt, style_name, style_data, text, quote)
            logger.info(f"Brief: style={style_name} layout={brief.layout} font={brief.font}")
            return {**state, "brief": brief}
    except Exception as exc:
        logger.warning(f"Brief generation failed ({exc}) — using style default")

    brief = _default_brief(style_name, style_data, text, quote)
    return {**state, "brief": brief}


# ---------------------------------------------------------------------------
# Brief construction
# ---------------------------------------------------------------------------

def _build_brief(image_prompt: str, style_name: str, style_data: dict,
                 text: str, quote) -> DesignBrief:
    r = style_data.get("rendering", {})
    ov = r.get("overlay", {})

    word_count = len(text.split())
    layout = "big_center" if word_count <= 12 else "full_card"
    font_size = (96 if layout == "big_center" and word_count <= 7
                 else 88 if layout == "big_center"
                 else max(60, 78 - max(0, word_count - 12)))

    font = r.get("font", "playfair")
    if font not in _VALID_FONTS:
        font = "playfair"

    return DesignBrief(
        image_prompt=image_prompt,
        style=style_name,
        font=font,
        text_color=r.get("text_color", "#FFFFFF"),
        highlight_color=r.get("highlight_color", "#FFD700"),
        author_color=r.get("author_color", "#FFD700"),
        overlay=Overlay(
            type=ov.get("type", "gradient_bottom"),
            opacity=int(ov.get("opacity", 150)),
            color=ov.get("color", "#000000"),
        ),
        text_zone="center",
        layout=layout,
        decoration=r.get("decoration", "none"),
        highlight=quote.highlight if quote else "",
        highlight_style=r.get("highlight_style", "color"),
        font_size=font_size,
        animation="fade",
        skip_kenburns=bool(r.get("skip_kenburns", False)),
        mood_note="",
        voice_gender=r.get("voice_gender", "male"),
    )


def _default_brief(style_name: str, style_data: dict, text: str, quote) -> DesignBrief:
    desc = style_data.get("description", "Beautiful inspirational scene.").strip()
    first_line = desc.split("\n")[0].strip()
    image_prompt = f"{first_line} Centre band left clear and dark for text overlay. 9:16 portrait."
    return _build_brief(image_prompt, style_name, style_data, text, quote)


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build() -> any:
    g = StateGraph(PipelineState)
    g.add_node("pick_style", pick_style)
    g.add_node("generate_brief", generate_brief)
    g.set_entry_point("pick_style")
    g.add_edge("pick_style", "generate_brief")
    g.add_edge("generate_brief", END)
    return g.compile()
