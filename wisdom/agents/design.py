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
You are a world-class photographer and Creative Director creating a single scroll-stopping 8K image for a global social media audience aged 18–35.

Your task: translate the emotional truth of this quote into one breathtaking photographic scene.

QUOTE: "{text}"

VISUAL GRAMMAR — how to shoot (technique, light, palette — not the scene):
STYLE: {style_name}
{style_description}

SCENE PARAMETERS — what world to build:
  - LOCATION: {random_seed}
  - CONDITIONS: {atmospheric_twist}
  - EMOTIONAL ANCHOR: The quote's specific emotion shapes every detail — subject, action, scale, mood.
    Same style + different quote = completely different scene. Let the words lead.
{image_hint_block}
RULES:
1. GENERAL AUDIENCE: Beautiful, relatable, emotionally resonant. Must stop someone scrolling at 7am.
   Real-world cinematography — not conceptual art, not niche aesthetics, not fashion editorial.
2. 8K PHOTOREALISTIC: Hyper-real photography quality. Breathtaking natural detail. Cinematic color grade.
3. QUOTE-DRIVEN SCENE: The scene is entirely shaped by the quote's emotion. The style only defines how it is shot.
4. VAST NEGATIVE SPACE: The text overlay zone must be naturally clean and dark. Non-negotiable.{subject_constraint}

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
        "\n  - THEMATIC MANDATE: A scene that radiates quiet, earned power and freedom — aspirational but real. "
        "A woman (seen from behind, in silhouette, or as an anonymous presence) in a setting that embodies the quote's specific emotion. "
        "Beautiful, cinematic, and universally relatable. Not runway, not editorial, not conceptual art."
    ),
    "darkacademia": (
        "\n  - THEMATIC MANDATE: The quiet beauty of deep focus and solitary thought. "
        "Warm, intimate, lived-in environments where intellectual life happens — "
        "the kind of scene the viewer wants to step into. "
        "Driven entirely by the emotional truth of the quote."
    ),
    "latenight": (
        "\n  - THEMATIC MANDATE: The specific emotional weight of late-night solitude — honest, beautiful, deeply relatable. "
        "A scene the viewer has experienced themselves: that 2am moment of clarity, longing, or quiet truth. "
        "Cinematic, intimate, emotionally precise."
    ),
    "morning": (
        "\n  - THEMATIC MANDATE: The raw energy and possibility of a new day — specific and visceral, not generic. "
        "Light arriving, the world waking, potential made visible. "
        "The scene must feel alive with beginning. Driven entirely by the quote's emotional truth."
    ),
    "wisdom": (
        "\n  - THEMATIC MANDATE: A visual that makes the viewer stop and feel the weight of something true. "
        "Timeless, humbling in scale or detail. Must look like it belongs in National Geographic or a museum. "
        "Driven entirely by the emotional truth of the quote."
    ),
    "mindfulness": (
        "\n  - THEMATIC MANDATE: Genuine, breathtaking calm — a real moment of natural beauty so perfect it quiets the mind. "
        "The image must make the viewer exhale. Not spiritual cliché — actual beauty. "
        "Driven entirely by the emotional truth of the quote."
    ),
    "love": (
        "\n  - THEMATIC MANDATE: The real, human truth of connection — warmth, longing, tenderness, or heartbreak rendered beautifully. "
        "The image must feel deeply personal, like a memory the viewer has lived. "
        "Cinematic and emotionally precise. Driven entirely by the quote's specific feeling."
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
            "Himalayan ridge at first light",
            "Pacific coastal cliffside",
            "Saharan sand dune sea",
            "Nordic fjord valley",
            "ancient Japanese cedar forest",
            "Patagonian open steppe",
            "Icelandic volcanic moss plains",
            "Vietnamese terraced rice fields",
            "Scottish highland moor",
            "Atacama desert salt flat",
            "Norwegian mountain plateau",
            "Tuscan rolling hillside at harvest",
            "Alaskan wilderness tundra",
            "New Zealand south island coastline",
            "Moroccan desert at dusk",
        ]
        atmospheric_twists = [
            "Crystal clear air after heavy rain — everything hyper-sharp, colors deeply saturated.",
            "Morning mist slowly burning off — soft diffused layers, depth created by receding fog.",
            "Approaching storm on the horizon — dramatic contrast between dark sky and a single shaft of brilliant light.",
            "Low fog layer blanketing the ground — landscape or figure rising above a sea of cloud.",
            "Epic wide establishing shot — any human presence is made tiny against the vast landscape.",
            "Intimate ground-level framing looking up — the sky fills most of the frame.",
            "Golden backlight — subject silhouetted with a glowing rim-light halo, sky ablaze behind.",
            "Storm light breaking — one shaft of sunlight cutting through dark clouds onto a single point in the landscape.",
            "Pre-dawn blue hour — the world holds its breath in deep blue before the first light arrives.",
            "Last light of the day — everything dipped in deep orange and long purple shadows.",
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
