"""
Design agent — two-phase LangGraph state machine.

Phase 1 (pick_style): LLM picks the best visual style for the quote.
Phase 2 (generate_brief): LLM writes a vivid, quote-specific image prompt.
Rendering parameters (font, colors, overlay) come from the style's config — not LLM.
"""

from __future__ import annotations

import logging
import random
import re

from langgraph.graph import END, StateGraph

import wisdom.config as cfg
from wisdom import providers
from wisdom.schemas import DesignBrief, Overlay, PipelineState

logger = logging.getLogger(__name__)

_ACCOUNT = "global inspirational quotes account (ages 18–35)"

_VALID_FONTS = frozenset(
    {
        "inter",
        "outfit",
        "spectral",
        "jost",
        "satisfy",
        "playfair",
        "cormorant",
        "bebas",
        "poppins",
        "anton",
        "cinzel",
        "great_vibes",
        "montserrat",
    }
)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def _styles_for_theme(
    theme_key: str, locked: list[str] | None, recent: list[str]
) -> list[dict]:
    all_styles = cfg.styles()
    result = []
    for name, s in all_styles.items():
        if locked and name not in locked:
            continue
        if not locked and theme_key not in s.get("categories", []):
            continue
        result.append({"name": name, **s})
    weight_order = {"high": 3, "medium": 2, "low": 1}
    result.sort(
        key=lambda s: weight_order.get(s.get("weight", "medium"), 2), reverse=True
    )
    return result


def _picker_prompt(
    quote_text: str, theme_key: str, styles: list[dict], recent: list[str]
) -> str:
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
You are a world-class photographer and Creative Director creating a single scroll-stopping 8K image for a global social media audience aged 18–35.

Your task: translate the emotional truth of this quote into one breathtaking photographic scene.

QUOTE: "{text}"

VISUAL GRAMMAR — how to shoot (technique, light, palette — not the scene):
STYLE: {style_name}
{style_description}

SCENE PARAMETERS — what world to build:
  - SCENE & SUBJECT: Imagine a highly creative, random, and unique scene that captures the SENTIMENT and EMOTION of the quote. 
    DO NOT translate the quote literally. Instead, find a visual metaphor. 
    Vary your subjects wildly: use nature, animals, human figures (no recognizable faces), architecture, or everyday objects. Every single prompt must feel completely different from the last.
  - EMOTIONAL ANCHOR: The quote's specific emotion shapes every detail. Let the words lead.
{image_hint_block}
RULES:
1. GENERAL AUDIENCE: Beautiful, relatable, emotionally resonant. Must stop someone scrolling.
2. 8K PHOTOREALISTIC: Hyper-real photography quality. Breathtaking natural detail. Cinematic color grade.
3. CREATIVE FREEDOM: The scene is entirely up to your imagination. The style only defines how it is shot (lighting, grading, technique).
4. VAST NEGATIVE SPACE: The text overlay zone must be naturally clean and uncluttered. Non-negotiable.{subject_constraint}

Write 4–6 sentences:
  location & setting → subject & emotional action → atmospheric detail → exact color palette (hex values) → lighting & composition.

Constraints:
  - TEXT ZONE: {text_zone_instruction}
  - No text, words, signs, logos, watermarks, or explicitly recognizable faces.
  - 9:16 portrait format, 8K resolution.

Reply with ONLY the image prompt — plain text, no JSON, no preamble.
"""

_THEME_SUBJECT_CONSTRAINTS: dict[str, str] = {
    "womenpower": (
        "\n  - THEMATIC MANDATE: Keep it like an editorial style: elegant, fashionable, well-dressed woman. "
        "Aspirational, confident, and put-together. Do not show recognizable faces, but focus on the "
        "attitude, clothing, posture, and sophisticated environment."
    ),
}


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
        return {**state, "_chosen_style": "golden_hour_epic"}

    try:
        prompt = _picker_prompt(quote.text if quote else "", theme_key, styles, recent)
        raw, provider_info = providers.llm.generate(prompt, role="style_picker")
        if "model_usage" not in state:
            state["model_usage"] = {}
        state["model_usage"]["Style Picker"] = provider_info

        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            import json

            style_name = json.loads(m.group()).get("style", "")
            if style_name and style_name in cfg.styles():
                logger.info(f"Style: {style_name}")
                return {**state, "_chosen_style": style_name}
    except Exception as exc:
        logger.warning(f"Style picker failed ({exc}) — using top style")

    fallback = next(
        (s["name"] for s in styles if s["name"] not in recent), styles[0]["name"]
    )
    return {**state, "_chosen_style": fallback}


def generate_brief(state: PipelineState) -> PipelineState:
    theme_key = state["theme_key"]
    quote = state.get("quote")
    style_name = state.get("_chosen_style", "cinematic_35mm")
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
            "bottom_right": "bottom-right corner",
        }.get(r.get("text_zone", "center"), "center")

        text_zone_instruction = (
            f"The {zone_desc} of the frame will have text overlaid on it. "
            f"That area MUST be naturally clean, shadowed, or low-contrast in the scene itself — "
            f"not bright or busy — so the text is legible."
        )
        subject_constraint = _THEME_SUBJECT_CONSTRAINTS.get(theme_key, "")

        prompt = _IMAGE_PROMPT_TEMPLATE.format(
            text=text,
            style_name=style_name,
            style_description=style_desc,
            image_hint_block=f"ADDITIONAL DIRECTION: {image_hint}\n"
            if image_hint
            else "",
            text_zone_instruction=text_zone_instruction,
            subject_constraint=subject_constraint,
        )
        image_prompt, provider_info = providers.llm.generate(prompt, role="creative_brief")
        if "model_usage" not in state:
            state["model_usage"] = {}
        state["model_usage"]["Creative Brief"] = provider_info

        image_prompt = image_prompt.strip()
        if len(image_prompt.split()) >= 20:
            brief = _build_brief(image_prompt, style_name, style_data, text, quote)
            logger.info(
                f"Brief: style={style_name} layout={brief.layout} font={brief.font}"
            )
            return {**state, "brief": brief}
    except Exception as exc:
        logger.warning(f"Brief generation failed ({exc}) — using style default")

    brief = _default_brief(style_name, style_data, text, quote)
    return {**state, "brief": brief}


# ---------------------------------------------------------------------------
# Brief construction
# ---------------------------------------------------------------------------


def _build_brief(
    image_prompt: str, style_name: str, style_data: dict, text: str, quote
) -> DesignBrief:
    r = style_data.get("rendering", {})
    ov = r.get("overlay", {})

    word_count = len(text.split())
    layout = r.get("layout", "big_center")
    if layout == "minimalist":
        font_size = 40
    elif layout == "asymmetric":
        font_size = 46
    else:
        font_size = (
            52
            if layout == "big_center" and word_count <= 7
            else 46
            if layout == "big_center"
            else max(38, 50 - max(0, word_count - 12))
        )

    variants = r.get("font_variants") or [r.get("font", "playfair")]
    font = random.choice(variants)
    logger.info(f"Font: {font} (pool: {variants})")

    text_zone = r.get("text_zone", "center")
    overlay_type = ov.get("type", "gradient_bottom")

    return DesignBrief(
        image_prompt=image_prompt,
        style=style_name,
        font=font,
        text_color=r.get("text_color", "#FFFFFF"),
        highlight_color=r.get("highlight_color", "#D4AF37"),
        author_color=r.get("author_color", "#D4AF37"),
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
        skip_kenburns=True,
        mood_note="",
        voice_gender=r.get("voice_gender", "male"),
    )


def _default_brief(style_name: str, style_data: dict, text: str, quote) -> DesignBrief:
    desc = style_data.get("description", "Beautiful inspirational scene.").strip()
    first_line = desc.split("\n")[0].strip()
    r = style_data.get("rendering", {})
    ov = r.get("overlay", {})
    overlay_type = ov.get("type", "gradient_bottom")
    zone = (
        "bottom third"
        if overlay_type == "gradient_bottom"
        else "top third"
        if overlay_type == "gradient_top"
        else "center band"
    )
    text_color = r.get("text_color", "#FFFFFF")
    image_prompt = f"{first_line} The {zone} must be naturally dark and uncluttered for {text_color} text overlay. 9:16 portrait."
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
