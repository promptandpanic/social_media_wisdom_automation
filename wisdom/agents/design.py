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
You are an unhinged, visionary Creative Director building high-stakes, scroll-stopping social media art from scratch. 
Write a vivid, avant-garde image generation prompt that captures the absolute emotional essence of this quote, but in a completely unpredictable, surreal, or dramatic way.

QUOTE: "{text}"
STYLE: {style_name}

{style_description}

CORE CREATIVE DIRECTIVES:
1. SCROLL-STOPPING ORIGINALITY: Break the rules. No clichés. No "person walking on a path" or "tree in a field". Invent a wildly unpredictable, high-stakes visual metaphor. Use extreme scale contrast, impossible architecture, gravity-defying objects, or surreal minimalism.
2. MEDIUM FLUIDITY: Do not just use photography. Depending on the quote's energy, mandate striking mediums: Brutalist 3D renders, neo-noir illustrations, hyper-macro textures, glitch-art aesthetics, liminal space photography, dark renaissance paintings, or retro-futurism.
3. SHOCK VALUE & MOOD: The image must evoke an immediate gasp or deep emotional resonance. Use aggressive or ethereal lighting (e.g., neon-lit darkness, blinding bioluminescence, void-like shadows, harsh brutalist sunlight).
4. THEMATIC RADICALISM: Do NOT repeat the exact example concepts listed in the constraints. You must invent a completely fresh, unique, and HIGHLY creative concept of your own.
5. {image_hint_block}

Write 4–6 intensely descriptive sentences describing the scene:
  medium/style → subject & action → extreme setting/environment → bold color palette (with hex values) → dramatic lighting & composition.

Constraints:
  - COMPOSITION: The area for text must be naturally clean, high-contrast, and contain VAST NEGATIVE SPACE (voids, massive skies, flat brutalist walls, empty dark waters). This is non-negotiable.
  - TEXT OVERLAY: {text_zone_instruction}
  - No text, words, signs, logos, watermarks, or explicitly recognizable faces.
  - 9:16 portrait format.{subject_constraint}
  - CREATIVE CHAOS SEED: {random_seed}
  - DRAMATIC TWIST: {atmospheric_twist}

Reply with ONLY the image prompt — plain text, no JSON, no preamble.
"""

_THEME_SUBJECT_CONSTRAINTS: dict[str, str] = {
    "womenpower": (
        "\n  - THEMATIC MANDATE: A fierce, highly conceptual representation of feminine dominance, resilience, or divine energy. "
        "Think avant-garde fashion mixed with surrealism. "
        "Examples: A colossal marble statue of a woman fracturing to reveal glowing gold beneath, a figure floating in a gravity-defying storm of crimson silk, or a sleek futuristic silhouette standing untouched amid total destruction. "
        "Do NOT use basic 'woman in a suit' tropes. Go mythic, brutalist, or hyper-modern."
    ),
    "darkacademia": (
        "\n  - THEMATIC MANDATE: A dark, labyrinthine visual of forbidden knowledge, obsessive intellect, or gothic surrealism. "
        "Examples: A sprawling library where the books are entirely made of glowing crystal, an endless spiral staircase sinking into an ink-black void, or macro-photography of a crumbling marble bust weeping molten bronze. "
        "Make it atmospheric, obsessive, and visually overwhelming."
    ),
    "latenight": (
        "\n  - THEMATIC MANDATE: The psychological weight of 3 AM. A visually striking, liminal, or neo-noir metaphor for isolation and realization. "
        "Examples: A single glowing red doorway suspended in an endless ocean at midnight, a brutalist concrete room lit only by the glare of a monolithic neon monolith, or a distorted reflection in shattered black glass. "
        "Keep it mysterious, lonely, and deeply cinematic."
    ),
    "morning": (
        "\n  - THEMATIC MANDATE: An aggressive, high-energy burst of awakening, discipline, or raw potential. "
        "Examples: A sun literally exploding from the chest of an abstract geometric figure, extreme macro of a single bead of sweat shattering concrete upon impact, or a blindingly bright hyper-minimalist ascent into pure light. "
        "Avoid basic 'gym/runner at dawn' concepts. Make it feel like an unstoppable force of nature."
    ),
    "wisdom": (
        "\n  - THEMATIC MANDATE: A clean, highly abstract, mind-bending visual metaphor for consciousness, time, or truth. "
        "Examples: A surreal tesseract of glass floating over a perfectly mirrored black desert, a giant iris embedded in a cliff face watching a storm, or a minimalist zen garden where the stones are floating monoliths. "
        "Break reality. Use extreme minimalism or mind-bending scale."
    ),
    "mindfulness": (
        "\n  - THEMATIC MANDATE: An impossible serenity. A visually stunning, hyper-calm environment that defies physics. "
        "Examples: A perfectly smooth sphere of water hovering silently in a sterile white brutalist gallery, a vast field of bioluminescent grass waving in slow-motion without wind, or an endless pastel sky reflected on a liquid metal floor. "
        "Evoke a feeling of profound, almost unsettling peace and stillness."
    ),
    "love": (
        "\n  - THEMATIC MANDATE: A visceral, high-stakes metaphor for soul-deep connection, sacrifice, or warmth. "
        "Examples: Two supernovas colliding to form a single quiet glowing core, a macro shot of two hands turning into intertwined smoke, or a stark dark room where two light beams violently bend towards each other. "
        "Avoid basic romantic clichés. Make the emotion feel powerful, cosmic, or deeply structural."
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

    fallback = next(
        (s["name"] for s in styles if s["name"] not in recent), styles[0]["name"]
    )
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
            "bottom_right": "bottom-right corner",
        }.get(r.get("text_zone", "center"), "center")

        text_zone_instruction = (
            f"The {zone_desc} of the frame will have text overlaid on it. "
            f"That area MUST be naturally clean, shadowed, or low-contrast in the scene itself — "
            f"not bright or busy — so the text is legible."
        )
        subject_constraint = _THEME_SUBJECT_CONSTRAINTS.get(theme_key, "")

        import random

        variation_seeds = [
            "Glitch-core Surrealism",
            "Baroque Futurism",
            "Liminal Horror / Void",
            "Ethereal Renaissance",
            "Neon-Brutalism",
            "Hyper-Macro Abstract",
            "Cyber-Mysticism",
            "Atmospheric Monolithic",
            "Retro-Futuristic Noir",
            "Dreamcore / Weirdcore",
        ]
        atmospheric_twists = [
            "Time frozen at the exact moment of a shattering impact.",
            "Submerged entirely under ethereal, bioluminescent water.",
            "Harsh, blinding strobe lighting freezing dynamic motion.",
            "Melted reality with extreme chromatic aberration and distortion.",
            "A chilling, oppressive liminal emptiness with zero shadows.",
            "Volcanic ash falling softly like snow in a hyper-colored light.",
            "An overwhelming sense of monumental scale and insignificance.",
            "Hyper-saturated infrared colors making nature look alien.",
        ]

        prompt = _IMAGE_PROMPT_TEMPLATE.format(
            text=text,
            style_name=style_name,
            style_description=style_desc,
            image_hint_block=f"ADDITIONAL DIRECTION: {image_hint}\n"
            if image_hint
            else "",
            text_zone_instruction=text_zone_instruction,
            subject_constraint=subject_constraint,
            random_seed=random.choice(variation_seeds),
            atmospheric_twist=random.choice(atmospheric_twists),
        )
        image_prompt = providers.llm.generate(prompt, role="creative_brief").strip()
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
        font_size = 38
    elif layout == "asymmetric":
        font_size = 48
    else:
        # Ultra-elegant aesthetic: smaller text, maximum negative space
        font_size = (
            60
            if layout == "big_center" and word_count <= 7
            else 52
            if layout == "big_center"
            else max(42, 56 - max(0, word_count - 12))
        )

    font = r.get("font", "playfair")

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
