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

_VALID_FONTS = frozenset({"playfair", "montserrat", "bebas", "poppins", "inter", "outfit", "spectral"})


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
You are a world-class Creative Director specialized in emotional storytelling.
Your goal is to pick ONE visual style that perfectly matches the "SOUL" and "ENERGY" of the quote.

CRITICAL RULES:
1. Vibe-Matching: Do NOT pick a romantic style for a quote about business or discipline. Do NOT pick a minimalist style for a quote that feels chaotic or intense.
2. Emotional Resonance: The image should feel like a visual extension of the words. 
3. Avoid Clichés: Don't just pick the first style that fits the category; pick the one that fits the "feeling."

QUOTE: "{quote_text}"
THEME: {theme_key}

{chr(10).join(lines)}

Return ONLY valid JSON: {{"style": "chosen_style_name"}}"""


_IMAGE_PROMPT_TEMPLATE = """\
Write a vivid image generation prompt that captures the deep emotional essence of this quote.

QUOTE: "{text}"
STYLE: {style_name}

{style_description}

GUIDELINES FOR CREATIVITY:
1. SOUL-MATCHING: The scene MUST reflect the emotion of the quote. If the quote is about solitude, show a vast, peaceful landscape. If it's about struggle, show grit and determination.
2. NO MISMATCHES: Never show romantic connection for a quote about independent hustle. Never show high-energy fitness for a quote about quiet reflection.
3. VISUAL METAPHOR: Instead of being literal, use atmosphere, lighting, and composition to tell the story.
4. {image_hint_block}

Write 4–6 rich sentences describing the scene:
  subject → setting → technique/medium → colour palette (use hex values) → lighting → composition

Constraints:
  - DO NOT generate abstract or overly "AI-stylized" art unless specified.
  - Scene MUST feel intentional, high-end, and crystal clear.
  - COMPOSITION: The area for text must be naturally clean, high-contrast, and COMPLETELY UNCLUTTERED.
  - TEXT OVERLAY: {text_zone_instruction}
  - No text, words, signs, logos, watermarks, or explicitly recognizable faces.
  - 9:16 portrait format.

Reply with ONLY the image prompt — plain text, no JSON, no preamble.
"""


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
        r = style_data.get("rendering", {})
        ov = r.get("overlay", {})
        overlay_type = ov.get("type", "gradient_bottom")
        text_color = r.get("text_color", "#FFFFFF")
        zone_desc = {
            "top": "top third",
            "bottom": "bottom third",
            "center": "center area",
            "top_minimalist": "top area with lots of negative space",
            "center_minimalist": "center area with lots of negative space",
            "top_left": "top-left corner",
            "top_right": "top-right corner",
            "bottom_left": "bottom-left corner",
            "bottom_right": "bottom-right corner"
        }.get(r.get("text_zone", "center"), "center")
        
        text_zone_instruction = (
            f"The {zone_desc} of the frame will have {text_color} text overlaid on it. "
            f"That area MUST be naturally clean, shadowed, or low-contrast in the scene itself — "
            f"not bright or busy — so the text is legible."
        )
        prompt = _IMAGE_PROMPT_TEMPLATE.format(
            text=text,
            style_name=style_name,
            style_description=style_desc,
            image_hint_block=f"ADDITIONAL DIRECTION: {image_hint}\n" if image_hint else "",
            text_zone_instruction=text_zone_instruction,
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
    layout = r.get("layout", "big_center")
    if layout == "minimalist":
        font_size = 40
    elif layout == "asymmetric":
        font_size = 56
    else:
        # Ultra-elegant aesthetic: smaller text, maximum negative space
        font_size = (72 if layout == "big_center" and word_count <= 7
                     else 64 if layout == "big_center"
                     else max(48, 60 - max(0, word_count - 12)))

    font = r.get("font", "playfair")

    text_zone = r.get("text_zone", "center")
    overlay_type = ov.get("type", "gradient_bottom")

    return DesignBrief(
        image_prompt=image_prompt,
        style=style_name,
        font=font,
        text_color=r.get("text_color", "#FFFFFF"),
        highlight_color=r.get("highlight_color", "#FFD700"),
        author_color=r.get("author_color", "#FFD700"),
        overlay=Overlay(
            type=overlay_type,
            opacity=int(ov.get("opacity", 150)),
            color=ov.get("color", "#000000"),
            blur=int(ov.get("blur", 20)),
        ),
        text_zone=text_zone,
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
    r = style_data.get("rendering", {})
    ov = r.get("overlay", {})
    overlay_type = ov.get("type", "gradient_bottom")
    zone = "bottom third" if overlay_type == "gradient_bottom" else "top third" if overlay_type == "gradient_top" else "center band"
    text_color = r.get("text_color", "#FFFFFF")
    image_prompt = (
        f"{first_line} The {zone} must be naturally dark and uncluttered for {text_color} text overlay. 9:16 portrait."
    )
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
