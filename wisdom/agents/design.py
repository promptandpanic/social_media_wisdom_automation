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
You are a Creative Director writing a prompt for an AI image generator (Flux Pro / Imagen 4). Your output feeds directly into the model — write for the model, not for a human reader.

QUOTE: "{text}"

VISUAL GRAMMAR: {style_name}
{style_description}

COLOR SCIENCE: {color_grade}

SCENE SEED: {random_seed}
LENS TECHNIQUE: {atmospheric_twist}
{image_hint_block}RULES:
1. SUBJECT FREEDOM: The subject can be human, animal, vehicle, weather phenomenon, landscape, or object — whichever carries the quote's emotion most powerfully. A resilience quote → lone wolf on a frozen plain, or a soldier in mud. A peace quote → still lake at dawn, or a deer drinking. A discipline quote → military humvee on a brutal road. Let the quote's FEELING choose the subject — never default to a person standing somewhere.
2. HERO DETAIL: Include one unexpected, visually arresting detail the viewer notices only on second look — something that couldn't be predicted from the quote alone. Make it specific and physical.
3. TEXT ZONE: {text_zone_instruction}
4. No text, words, signs, logos, watermarks, or explicitly recognizable faces.
5. 9:16 portrait format, 8K photorealistic quality.{subject_constraint}

OUTPUT — write a single dense image-generation prompt using comma-separated descriptors only. NO prose sentences. NO explanation. NO preamble:
[subject + emotional action], [specific environment], [lighting: type + direction + color temperature], [color grade: {color_grade_short}], [lens + aperture + depth of field], [film texture or grain], [mood + atmosphere], [hero detail], photorealistic, 8K, 9:16 portrait
"""

_THEME_SUBJECT_CONSTRAINTS: dict[str, str] = {
    "womenpower": (
        "\n  - THEMATIC MANDATE: A stunning, stylish woman — impeccably dressed, confident, aspirational. "
        "She can be in motion or still, but her presence commands the frame. Fashion-forward without being costume: "
        "think tailored coat on a windswept cliff, silk dress in a sunlit doorway, or structured outfit in a minimal architectural space. "
        "The image should make someone stop scrolling and think 'I want to feel like that.' "
        "Do not show faces. Convey power, elegance, and intention through posture, clothing, light, and environment."
    ),
    "darkacademia": (
        "\n  - THEMATIC MANDATE: The quiet beauty of deep focus and solitary thought. "
        "Use your boundless creativity to imagine a warm, intimate, lived-in environment. "
        "Do not rely on clichés; think completely out of the box and let the quote's emotional truth drive the scene."
    ),
    "latenight": (
        "\n  - THEMATIC MANDATE: The specific emotional weight of late-night solitude. "
        "Use your boundless creativity to imagine a cinematic, intimate, and emotionally precise scene. "
        "Do not rely on clichés; think completely out of the box."
    ),
    "morning": (
        "\n  - THEMATIC MANDATE: The raw energy and possibility of a new day. "
        "Use your boundless creativity to imagine a specific, visceral scene where light arrives and potential is made visible. "
        "Do not rely on clichés; think completely out of the box."
    ),
    "wisdom": (
        "\n  - THEMATIC MANDATE: A visual that makes the viewer stop and feel the weight of something true. "
        "Use your boundless creativity to imagine a timeless, humbling scene. "
        "Do not rely on clichés; think completely out of the box."
    ),
    "mindfulness": (
        "\n  - THEMATIC MANDATE: Genuine, breathtaking calm — a real moment of natural beauty. "
        "Use your boundless creativity to imagine a scene so perfect it quiets the mind. "
        "Do not rely on clichés or spiritual stereotypes; think completely out of the box."
    ),
    "love": (
        "\n  - THEMATIC MANDATE: The real, human truth of connection — warmth, longing, tenderness, or heartbreak. "
        "Use your boundless creativity to imagine a deeply personal, cinematic scene. "
        "Do not rely on clichés; think completely out of the box."
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
        return {**state, "_chosen_style": "cinematic_35mm"}

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

        variation_seeds = [
            # Wildlife & nature
            "lone wolf standing at the edge of a frozen tundra at pre-dawn blue hour",
            "herd of elephants crossing an African savanna in harsh midday heat, dust rising",
            "bald eagle in mid-dive through storm clouds above a mountain range",
            "deer drinking from a still forest lake at sunrise, mist rising off the water",
            "pride of lions resting on a sun-scorched rocky kopje at golden hour",
            "great white shark breaching through dark ocean water under a storm sky",
            "wild horse galloping through a desert dust storm at dusk",
            # Vehicles & conflict
            "military humvee pushing through a flooded jungle road in monsoon rain",
            "commercial aircraft cutting through towering cumulonimbus clouds at altitude",
            "armoured convoy on a rutted dirt road through a conflict-scarred landscape at dawn",
            "old cargo ship moving through a thick Arctic fog, ice floes on either side",
            # Weather & sky phenomena
            "double rainbow arching over a rain-soaked valley after a violent storm",
            "lightning bolt splitting a bruised purple sky over open ocean at night",
            "aurora borealis in full bloom reflecting perfectly in a glassy arctic lake",
            "wall of red dust storm rolling across a cracked desert plain at dusk",
            "tornado forming on the horizon of a flat Midwestern prairie at golden hour",
            # Dramatic landscapes
            "ancient jungle temple being slowly consumed by massive tree roots and vines",
            "sunlit wheat field, camera low, golden stalks filling the frame, sky vast above",
            "snow-covered mountain lookout, vast stillness, breath visible in cold air",
            "African savanna waterhole at dusk, multiple species gathered under a burning sky",
            "cliff edge overlooking a vast ocean at golden hour, waves crashing far below",
            # Intimate & architectural
            "minimalist white room with a single arched window, morning light pooling on the floor",
            "linen-curtained window seat, soft diffused morning light, a cup of tea steaming",
            "rain-soaked cobblestone alley with warm amber glow from a single doorway",
            "still lake perfectly mirroring a dramatic painted sky at sunrise",
        ]
        atmospheric_twists = [
            "Anamorphic prime lens — wide open at f/1.4, background dissolves to creamy bokeh, horizontal lens flare cuts across frame.",
            "Golden hour backlight — subject rim-lit or silhouetted against warm amber sun, atmospheric haze fills the air.",
            "Cinematic 35mm film grain — Kodak Portra warmth, lifted blacks, natural halation bleeding around every light source.",
            "Tilt-shift razor focus — one element pin-sharp, everything else dissolves to soft breath, disorienting and beautiful.",
            "Long exposure motion — water or clouds rendered silky smooth, subject perfectly still, time made visible.",
            "Magic hour silhouette — strong clean silhouette against a sky that looks painted by hand, form over detail.",
            "Extreme macro — background completely dissolved, one tiny physical detail rendered in breathtaking clarity.",
            "Film halation — warm orange-red glow bleeding around bright highlights, as if light is alive and restless.",
            "Overcast diffused — no harsh shadows, colours deeply saturated from within, the world glowing without a sun.",
            "Stormy dramatic backlight — dark turbulent sky, single beam of light breaking through, subject caught in the beam.",
        ]

        r = style_data.get("rendering", {})
        color_grade = r.get("color_grade", "natural cinematic color grade")
        color_grade_short = color_grade.split(",")[0]  # first clause only for inline hint

        prompt = _IMAGE_PROMPT_TEMPLATE.format(
            text=text,
            style_name=style_name,
            style_description=style_desc,
            color_grade=color_grade,
            color_grade_short=color_grade_short,
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
        font_size = 33
    elif layout == "asymmetric":
        font_size = 41
    else:
        font_size = (
            50
            if layout == "big_center" and word_count <= 7
            else 43
            if layout == "big_center"
            else max(37, 48 - max(0, word_count - 12))
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
