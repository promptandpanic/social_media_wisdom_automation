"""
PIL-based image composer.

Text layout is 100% math-driven — pixel_wrap() uses font metrics so no
line ever exceeds the canvas width regardless of font or quote length.

Public API:
  compose_image(image_bytes, quote, brief)             → JPEG bytes
  compose_partial(image_bytes, quote, brief, n_lines)  → JPEG bytes
  compose_base(image_bytes, brief)                     → PIL Image
  compose_text_layer(image_bytes, quote, brief)        → PNG bytes (RGBA)
  get_reveal_counts(quote, brief)                      → list[int]
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import wisdom.config as cfg
from wisdom.schemas import DesignBrief, Quote

logger = logging.getLogger(__name__)

FONTS_DIR = Path("assets/fonts")

# ----- canvas constants (from config) -----
_img = cfg.image_cfg()
IMAGE_WIDTH = _img["width"]
IMAGE_HEIGHT = _img["height"]

# Instagram mobile safe zones (1080×1920):
#   Right: action icons ~150px, Bottom: caption bar ~420px
MARGIN_X = 72
MARGIN_X_R = 180
TEXT_MAX_W = IMAGE_WIDTH - MARGIN_X - MARGIN_X_R
TEXT_ZONE_CX = (MARGIN_X + IMAGE_WIDTH - MARGIN_X_R) // 2
INSTAGRAM_SAFE_BOTTOM = IMAGE_HEIGHT - 420

_ZONE_MAX_H = {
    "top": int(IMAGE_HEIGHT * 0.55),
    "center": int(IMAGE_HEIGHT * 0.72),
    "bottom": INSTAGRAM_SAFE_BOTTOM - int(IMAGE_HEIGHT * 0.28),
}

# ---------------------------------------------------------------------------
# Font registry
# ---------------------------------------------------------------------------

_FONT_URLS: dict[str, tuple[str, str]] = {
    "poppins": (
        "poppins.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Regular.ttf",
    ),
    "inter": (
        "inter.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/static/Inter-Regular.ttf",
    ),
    "poppins_bold": (
        "poppins_bold.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Bold.ttf",
    ),
    "outfit": (
        "outfit.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/outfit/static/Outfit-Regular.ttf",
    ),
    "spectral": (
        "spectral.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/spectral/Spectral-Regular.ttf",
    ),
    "jost": (
        "jost.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/jost/static/Jost-Regular.ttf",
    ),
    "satisfy": (
        "satisfy.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/satisfy/Satisfy-Regular.ttf",
    ),
    "playfair": (
        "playfair.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/static/PlayfairDisplay-Regular.ttf",
    ),
    "cormorant": (
        "cormorant.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/static/CormorantGaramond-Regular.ttf",
    ),
    "dancing": (
        "dancing.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/dancingscript/static/DancingScript-Regular.ttf",
    ),
    "caveat": (
        "caveat.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/static/Caveat-Regular.ttf",
    ),
    "bebas": (
        "bebas.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/bebasneue/BebasNeue-Regular.ttf",
    ),
    "anton": (
        "anton.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
    ),
    "cinzel": (
        "cinzel.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/cinzel/static/Cinzel-Regular.ttf",
    ),
    "great_vibes": (
        "great_vibes.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/greatvibes/GreatVibes-Regular.ttf",
    ),
    "shadows_into_light": (
        "shadows_into_light.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/shadowsintolight/ShadowsIntoLight-Regular.ttf",
    ),
    "montserrat": (
        "montserrat_bold.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/montserrat/static/Montserrat-Bold.ttf",
    ),
    "oswald": (
        "oswald_bold.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/oswald/static/Oswald-Bold.ttf",
    ),
    "raleway": (
        "raleway_bold.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/raleway/static/Raleway-Bold.ttf",
    ),
    "patrick_hand": (
        "patrick_hand.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    ),
    "comfortaa": (
        "comfortaa.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/comfortaa/static/Comfortaa-Regular.ttf",
    ),
    "indie_flower": (
        "indie_flower.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/indieflower/IndieFlower-Regular.ttf",
    ),
    "fredoka": (
        "fredoka.ttf",
        "https://raw.githubusercontent.com/google/fonts/main/ofl/fredoka/static/Fredoka-Regular.ttf",
    ),
}

_font_cache: dict = {}


def _ensure_fonts() -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, (filename, url) in _FONT_URLS.items():
        path = FONTS_DIR / filename
        if not path.exists():
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    path.write_bytes(r.content)
                    logger.info(f"  ✓ font: {filename}")
            except Exception as exc:
                logger.warning(f"  ✗ font {filename}: {exc}")


def _font(key: str, size: int) -> ImageFont.FreeTypeFont:
    ck = f"{key}_{size}"
    if ck not in _font_cache:
        _ensure_fonts()
        filename = _FONT_URLS.get(key, ("",))[0]
        path = FONTS_DIR / filename
        try:
            _font_cache[ck] = ImageFont.truetype(str(path), size)
        except Exception:
            fallback_path = FONTS_DIR / _FONT_URLS["poppins"][0]
            try:
                logger.warning(f"Font '{key}' unavailable — falling back to poppins")
                _font_cache[ck] = ImageFont.truetype(str(fallback_path), size)
            except Exception:
                _font_cache[ck] = ImageFont.truetype(
                    str(FONTS_DIR / _FONT_URLS["poppins_bold"][0]), size
                )
    return _font_cache[ck]


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple:
    m = re.search(r"#([0-9A-Fa-f]{6})", str(hex_color))
    if m:
        h = m.group(1)
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (255, 200, 50)


def _tw(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return draw.textbbox((0, 0), text, font=font)[2]


def _wrap_words(
    words: list[str], font: ImageFont.FreeTypeFont, max_width: int
) -> list[str]:
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        candidate = " ".join(cur + [word])
        bb = font.getbbox(candidate)
        w = bb[2] - bb[0]
        if w <= max_width:
            cur.append(word)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def pixel_wrap(
    text: str, font: ImageFont.FreeTypeFont, max_width: int = TEXT_MAX_W
) -> list[str]:
    words = text.split()
    return _wrap_words(words, font, max_width) or [text]


def _layout_lines(
    disp_text: str, font: ImageFont.FreeTypeFont, layout: str
) -> list[str]:
    if layout == "sentence_reveal":
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", disp_text.strip())
            if s.strip()
        ]
        if len(sentences) > 1:
            lines: list[str] = []
            for sent in sentences:
                lines.extend(pixel_wrap(sent, font))
            return lines
    return pixel_wrap(disp_text, font)


# Fonts that need bigger starting sizes for legibility
_FONT_SIZE_SCALE: dict[str, float] = {
    "inter": 1.00,
    "inter_bold": 1.00,
    "outfit": 1.02,
    "spectral": 1.05,
    "jost": 1.00,
    "satisfy": 1.12,
    "playfair": 1.08,
    "cormorant": 1.15,
    "dancing": 1.15,
    "caveat": 1.12,
    "bebas": 1.00,
    "poppins": 1.00,
}


def _fit_text(
    disp_text: str, font_key: str, font_size: int, layout: str, zone: str
) -> tuple[list[str], ImageFont.FreeTypeFont, int]:
    scale = _FONT_SIZE_SCALE.get(font_key, 1.0)
    font_size = max(36, int(font_size * scale))
    max_h = _ZONE_MAX_H.get(zone, int(IMAGE_HEIGHT * 0.70))
    for size in range(font_size, 34, -2):
        f = _font(font_key, size)
        lines = _layout_lines(disp_text, f, layout)
        if len(lines) * int(size * 1.28) <= max_h:
            return lines, f, size
    size = 36
    f = _font(font_key, size)
    return _layout_lines(disp_text, f, layout), f, size


def _sanitize(text: str) -> str:
    import unicodedata

    text = (
        text.replace("—", " - ")
        .replace("–", "-")
        .replace("…", "...")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ô", "o")
        .replace("û", "u")
    )
    return "".join(
        ch
        for ch in text
        if ord(ch) < 0x10000 and unicodedata.category(ch) not in ("So", "Cs")
    )


_CURSIVE_FONTS = {
    "dancing",
    "satisfy",
    "pacifico",
    "caveat",
    "kalam",
    "indieflower",
    "great_vibes",
}


def _drop_shadow_text(
    draw,
    xy,
    text,
    font,
    fill,
    shadow_color=(0, 0, 0, 180),
    offset=(4, 6),
    stroke_width=0,
    stroke_fill=None,
):
    x, y = xy
    # For a sharp 3D shadow (which looks modern), we just draw multiple offset layers
    if shadow_color and len(shadow_color) >= 4 and shadow_color[3] > 0:
        for i in range(1, 4):
            draw.text(
                (x + (offset[0] * i / 3), y + (offset[1] * i / 3)),
                text,
                font=font,
                fill=(*shadow_color[:3], int(shadow_color[3] * (4 - i) / 3)),
            )
    # Draw main text with optional translucent outline
    if stroke_width > 0 and stroke_fill:
        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
    else:
        draw.text((x, y), text, font=font, fill=fill)


def _render_line(
    draw,
    img_width: int,
    y: int,
    line: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    stroke: int = 3,
    highlight_text: str = "",
    hi_color: tuple = None,
    text_zone: str = "center",
    hl_style: str = "color",
) -> None:
    bb = font.getbbox(line)
    line_w = bb[2] - bb[0]

    if text_zone == "center":
        x = (img_width - line_w) // 2
    elif "left" in text_zone:
        x = MARGIN_X
    elif "right" in text_zone:
        x = img_width - MARGIN_X_R - line_w
    else:
        x = (img_width - line_w) // 2

    # Smart shadow & outline: white outline for dark text, dark outline for light text
    is_dark_text = (fill[0] * 0.299 + fill[1] * 0.587 + fill[2] * 0.114) < 128
    base_alpha = 180 if stroke > 0 else 80

    stroke_width = min(2, stroke) if stroke > 0 else 0
    stroke_fill = None

    if stroke_width > 0:
        if is_dark_text:
            stroke_fill = (250, 249, 246, 160)  # Soft Alabaster glow
            shadow_color = (250, 249, 246, 60)
        else:
            stroke_fill = (26, 26, 26, 120)  # Soft Charcoal outline
            shadow_color = (26, 26, 26, 60)
    else:
        shadow_color = (
            (255, 255, 255, base_alpha) if is_dark_text else (0, 0, 0, base_alpha)
        )

    letter_spacing = 0
    if "minimalist" in text_zone or font.path.lower().endswith(
        ("poppins", "montserrat")
    ):
        letter_spacing = 4 if "minimalist" in text_zone else 1

    if letter_spacing == 0:
        _drop_shadow_text(
            draw,
            (x, y),
            line,
            font,
            fill=fill,
            shadow_color=shadow_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return

    words = line.split(" ")
    current_x = x

    # Custom render function to handle letter spacing
    def _draw_text_custom(draw, xy, text, font, fill, ls):
        curr_x, curr_y = xy
        for char in text:
            _drop_shadow_text(
                draw,
                (curr_x, curr_y),
                char,
                font,
                fill=fill,
                shadow_color=shadow_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            curr_x += draw.textlength(char, font=font) + ls
        return curr_x

    hl_words = [hw.strip(".,!?\\\"'").lower() for hw in highlight_text.split()] if highlight_text else []

    for w in words:
        if w:
            word_clean = w.strip(".,!?\\\"'").lower()
            is_hl = word_clean in hl_words and word_clean
            word_fill = hi_color if is_hl else fill

            current_x = _draw_text_custom(
                draw, (current_x, y), w, font, fill=word_fill, ls=letter_spacing
            )

        # Add space with letter spacing
        current_x += draw.textlength(" ", font=font) + letter_spacing


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def _gradient_rect(
    img: Image.Image, y0: int, y1: int, color: tuple = (0, 0, 0), max_alpha: int = 180
) -> Image.Image:
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    height = y1 - y0
    for y in range(y0, y1):
        a = int(max_alpha * ((y - y0) / height) ** 0.55)
        draw.line([(0, y), (IMAGE_WIDTH, y)], fill=(*color, a))
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _gradient_rect_from_top(
    img: Image.Image, y0: int, y1: int, color: tuple = (0, 0, 0), max_alpha: int = 180
) -> Image.Image:
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    height = y1 - y0
    for y in range(y0, y1):
        frac = (y - y0) / height
        a = int(max_alpha * (1 - frac) ** 0.55)
        draw.line([(0, y), (IMAGE_WIDTH, y)], fill=(*color, a))
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _solid_overlay(
    img: Image.Image, opacity: int, color: tuple = (0, 0, 0)
) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (*color, opacity))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _vignette(img: Image.Image, intensity: int = 160) -> Image.Image:
    w, h = img.size
    rgba = img.convert("RGBA")
    vig = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vig)
    for i in range(min(w, h) // 2, 0, -1):
        alpha = int(intensity * (1 - (i / (min(w, h) / 2))) ** 1.8)
        draw.ellipse(
            [w // 2 - i, h // 2 - i, w // 2 + i, h // 2 + i],
            outline=(0, 0, 0, alpha),
            width=4,
        )
    return Image.alpha_composite(rgba, vig).convert("RGB")


def _apply_overlay(img: Image.Image, brief: DesignBrief) -> Image.Image:
    ov = brief.overlay
    otype = ov.type
    opacity = ov.opacity
    color = _hex_to_rgb(ov.color)

    if otype == "gradient_bottom":
        return _gradient_rect(
            img, int(IMAGE_HEIGHT * 0.20), IMAGE_HEIGHT, color=color, max_alpha=opacity
        )
    elif otype == "gradient_top":
        return _gradient_rect_from_top(
            img, 0, int(IMAGE_HEIGHT * 0.60), color=color, max_alpha=opacity
        )
    elif otype == "gradient_center":
        img = _gradient_rect(
            img,
            int(IMAGE_HEIGHT * 0.25),
            int(IMAGE_HEIGHT * 0.78),
            color=color,
            max_alpha=opacity,
        )
        return _gradient_rect_from_top(
            img,
            int(IMAGE_HEIGHT * 0.22),
            int(IMAGE_HEIGHT * 0.52),
            color=color,
            max_alpha=opacity // 2,
        )
    elif otype == "solid":
        return _solid_overlay(img, opacity, color=color)
    elif otype == "vignette":
        return _vignette(img, intensity=opacity)
    elif otype == "glass":
        # Handled in _draw_text for precise bounding box
        return img
    return img  # "none"


def _bg_luminance(img: Image.Image, zone: str) -> float:
    W, H = IMAGE_WIDTH, IMAGE_HEIGHT
    if "top" in zone:
        box = (MARGIN_X, int(H * 0.06), W - MARGIN_X, int(H * 0.45))
    elif "center" in zone:
        box = (MARGIN_X, int(H * 0.18), W - MARGIN_X, int(H * 0.82))
    else:
        box = (MARGIN_X, int(H * 0.38), W - MARGIN_X, H - 80)
    small = img.crop(box).resize((16, 16), Image.LANCZOS).convert("RGB")
    pixels = list(small.getdata())
    return sum(0.299 * r + 0.587 * g + 0.114 * b for r, g, b in pixels) / len(pixels)


def _contrast_ratio(lum_a: float, lum_b: float) -> float:
    a = (lum_a / 255) ** 2.2 + 0.05
    b = (lum_b / 255) ** 2.2 + 0.05
    lighter, darker = max(a, b), min(a, b)
    return lighter / darker


def _ensure_readable(color: tuple, bg_lum: float) -> tuple:
    txt_lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    if _contrast_ratio(txt_lum, bg_lum) >= 4.5:
        return color
    white_cr = _contrast_ratio(255, bg_lum)
    dark_cr = _contrast_ratio(20, bg_lum)
    return (255, 255, 255) if white_cr >= dark_cr else (20, 20, 20)


# ---------------------------------------------------------------------------
# Core text renderer
# ---------------------------------------------------------------------------


def _draw_text(
    img: Image.Image,
    quote: Quote,
    brief: DesignBrief,
    n_lines: int | None = None,
    lum_img: Image.Image | None = None,
) -> Image.Image:
    _QC = "\"'“”''«»„‟"
    text = _sanitize(quote.text).strip(_QC).strip()
    author = quote.author
    font_key = brief.font
    if font_key == "playfair_it":
        font_key = "playfair"
    txt_color = _hex_to_rgb(brief.text_color)
    hi_color = _hex_to_rgb(brief.highlight_color)
    text_zone = brief.text_zone
    decoration = brief.decoration
    layout = brief.layout

    upper = font_key == "bebas"
    disp_text = text.upper() if upper else text

    bg_lum = _bg_luminance(lum_img if lum_img is not None else img, text_zone)
    txt_color = _ensure_readable(txt_color, bg_lum)
    hi_color = _ensure_readable(hi_color, bg_lum)

    # Force text contrast for glass overlays
    if brief.overlay.type == "glass":
        if bg_lum > 128:
            # Bright BG -> Dark glass box -> Light text
            txt_color = (255, 255, 255)
        else:
            # Dark BG -> Light glass box -> Dark text
            txt_color = (30, 30, 30)

    font_size = max(36, brief.font_size)
    all_lines, f, font_size = _fit_text(
        disp_text, font_key, font_size, layout, text_zone
    )

    lines = all_lines[:n_lines] if n_lines is not None else all_lines
    line_h = int(font_size * 1.28)
    block_h = len(all_lines) * line_h

    if text_zone == "top" or text_zone == "top_minimalist":
        y = int(IMAGE_HEIGHT * 0.15)
    elif text_zone == "center" or text_zone == "center_minimalist":
        y = (IMAGE_HEIGHT - block_h) // 2
    elif text_zone.startswith("bottom_"):
        y = INSTAGRAM_SAFE_BOTTOM - block_h
    else:
        y = (IMAGE_HEIGHT - block_h) // 2

    # Glass effect - draw a blurred box behind the text area
    if brief.overlay.type == "glass":
        # Draw glass box slightly larger than text block
        box_padding = 100
        box_w = TEXT_MAX_W + (box_padding * 2)
        box_h = block_h + (box_padding * 2)

        if text_zone == "center":
            bx = (IMAGE_WIDTH - box_w) // 2
            by = y - box_padding
        elif "left" in text_zone:
            bx = MARGIN_X - box_padding
            by = y - box_padding
        elif "right" in text_zone:
            bx = IMAGE_WIDTH - MARGIN_X_R - TEXT_MAX_W - box_padding
            by = y - box_padding
        else:
            bx = (IMAGE_WIDTH - box_w) // 2
            by = y - box_padding

        # 1. Crop the region and blur it
        crop_box = (
            max(0, bx),
            max(0, by),
            min(IMAGE_WIDTH, bx + box_w),
            min(IMAGE_HEIGHT, by + box_h),
        )
        region = img.crop(crop_box)
        region = region.filter(ImageFilter.GaussianBlur(radius=brief.overlay.blur))

        # 2. Add a semi-transparent tint to the blurred region
        # If background is bright, use a dark glass box. If dark, use a light glass box.
        alpha = int(brief.overlay.opacity * 0.8)
        if bg_lum > 128:
            tint = Image.new("RGBA", region.size, (0, 0, 0, alpha))
        else:
            tint = Image.new("RGBA", region.size, (255, 255, 255, alpha))

        region_rgba = region.convert("RGBA")
        region_rgba = Image.alpha_composite(region_rgba, tint)

        # 3. Paste it back with rounded corners
        mask = Image.new("L", region.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle(
            [0, 0, region.size[0], region.size[1]], radius=40, fill=255
        )

        # If the destination image is RGBA (video layer), we want the box to be translucent
        if img.mode == "RGBA":
            # For video layers, we don't paste the blurred RGB, we paste a translucent tinted box
            # because we can't blur the moving video frames with a static PNG.
            glass_layer = Image.new("RGBA", region.size, (0, 0, 0, 0))
            glass_layer.paste(region_rgba, (0, 0), mask=mask)
            img.alpha_composite(glass_layer, (int(bx), int(by)))
        else:
            # For static images, we paste the blurred and tinted region
            img.paste(region_rgba.convert("RGB"), (int(bx), int(by)), mask=mask)

        # 4. Draw a subtle rounded border
        draw_box = ImageDraw.Draw(img, "RGBA")
        border_color = (255, 255, 255, 80) if bg_lum < 128 else (0, 0, 0, 80)
        draw_box.rounded_rectangle(
            [bx, by, bx + box_w, by + box_h], radius=40, outline=border_color, width=2
        )

    draw = ImageDraw.Draw(img)

    if decoration == "rule":
        draw.line(
            [(MARGIN_X, y - 20), (IMAGE_WIDTH - MARGIN_X_R, y - 20)],
            fill=(*hi_color, 220),
            width=3,
        )
    elif decoration == "quote_mark":
        dq_font = _font("playfair", 450)
        draw.text((MARGIN_X - 60, y - 100), "“", font=dq_font, fill=(*hi_color, 80))

    text_stroke = 0 if font_key in _CURSIVE_FONTS else 3

    for line in lines:
        _render_line(
            draw,
            TEXT_ZONE_CX * 2,
            y,
            line,
            font=f,
            fill=txt_color,
            stroke=text_stroke,
            highlight_text=quote.highlight,
            hi_color=hi_color,
            text_zone=text_zone,
            hl_style=brief.highlight_style,
        )
        y += line_h

    _SKIP_AUTHOR = {"unknown", "anonymous", "original"}
    all_visible = n_lines is None or n_lines >= len(all_lines)
    if (
        all_visible
        and author
        and author.lower() not in _SKIP_AUTHOR
        and not author.startswith("@")
    ):
        author_spaced = " ".join(list(author.upper().replace(" ", "  ")))
        a_font = _font("poppins", 26)
        dash_font = _font("poppins", 26)
        dash = "—  "
        d_bb = dash_font.getbbox(dash)
        n_bb = a_font.getbbox(author_spaced)
        total_w = (d_bb[2] - d_bb[0]) + (n_bb[2] - n_bb[0])
        ax = (TEXT_ZONE_CX * 2 - total_w) // 2
        ay = y + 16
        a_color = _hex_to_rgb(brief.author_color)
        draw.text((ax, ay), dash, font=dash_font, fill=a_color)
        draw.text(
            (ax + (d_bb[2] - d_bb[0]), ay), author_spaced, font=a_font, fill=a_color
        )

    # High-Fashion Editorial Header
    import hashlib
    try:
        # Generate a unique 4-character hex ID based on the quote text
        unique_id = hashlib.md5(quote.text.encode('utf-8')).hexdigest()[:4].upper()
        
        # Fallback to poppins if inter isn't available
        try:
            tag_font = _font("inter", 18)
        except Exception:
            tag_font = _font("poppins", 18)
            
        tag_text = f"A R C H I V E   //   N O .  {unique_id}"
        tag_bb = tag_font.getbbox(tag_text)
        tag_w = tag_bb[2] - tag_bb[0]
        # Draw at top center with low opacity
        draw.text(((IMAGE_WIDTH - tag_w) // 2, 70), tag_text, font=tag_font, fill=(255, 255, 255, 90))
    except Exception:
        pass

    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_reveal_counts(quote: Quote, brief: DesignBrief) -> list[int]:
    font_key = brief.font
    if font_key == "playfair_it":
        font_key = "playfair"
    font_size = max(36, brief.font_size)
    text = _sanitize(quote.text)
    upper = font_key == "bebas"
    disp_text = text.upper() if upper else text
    zone = brief.text_zone

    all_lines, f, _ = _fit_text(disp_text, font_key, font_size, "sentence_reveal", zone)

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", disp_text.strip()) if s.strip()
    ]

    if len(sentences) <= 1:
        n = len(all_lines)
        if n <= 3:
            return [n]
        mid = (n + 1) // 2
        return [mid, n]

    cumulative, total = [], 0
    for sent in sentences:
        total += len(pixel_wrap(sent, f))
        cumulative.append(total)
    return cumulative


def compose_image(image_bytes: bytes, quote: Quote, brief: DesignBrief) -> bytes:
    img = _load(image_bytes)
    img = _apply_overlay(img, brief)
    img = _draw_text(img, quote, brief)
    return _to_jpeg(img)


def compose_partial(
    image_bytes: bytes, quote: Quote, brief: DesignBrief, n_lines: int
) -> bytes:
    img = _load(image_bytes)
    img = _apply_overlay(img, brief)
    img = _draw_text(img, quote, brief, n_lines=n_lines)
    return _to_jpeg(img)


def compose_base(image_bytes: bytes, brief: DesignBrief) -> Image.Image:
    """Raw image only — no overlay, no text. Used as the Ken Burns zoom layer."""
    return _load(image_bytes)


def compose_overlay_layer(brief: DesignBrief) -> bytes:
    """Gradient/vignette overlay as a static RGBA PNG — no image, no text."""
    ov = brief.overlay
    otype = ov.type
    opacity = ov.opacity
    color = _hex_to_rgb(ov.color)
    W, H = IMAGE_WIDTH, IMAGE_HEIGHT

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    if otype == "gradient_bottom":
        y0, y1 = int(H * 0.20), H
        h = y1 - y0
        for y in range(y0, y1):
            a = int(opacity * ((y - y0) / h) ** 0.55)
            draw.line([(0, y), (W, y)], fill=(*color, a))
    elif otype == "gradient_top":
        y0, y1 = 0, int(H * 0.60)
        h = y1 - y0
        for y in range(y0, y1):
            a = int(opacity * (1 - (y - y0) / h) ** 0.55)
            draw.line([(0, y), (W, y)], fill=(*color, a))
    elif otype == "gradient_center":
        y0, y1 = int(H * 0.25), int(H * 0.78)
        h = y1 - y0
        for y in range(y0, y1):
            a = int(opacity * ((y - y0) / h) ** 0.55)
            draw.line([(0, y), (W, y)], fill=(*color, a))
        y0b, y1b = int(H * 0.22), int(H * 0.52)
        hb = y1b - y0b
        for y in range(y0b, y1b):
            a = int((opacity // 2) * (1 - (y - y0b) / hb) ** 0.55)
            draw.line([(0, y), (W, y)], fill=(*color, a))
    elif otype == "solid":
        canvas = Image.new("RGBA", (W, H), (*color, opacity))
    elif otype == "vignette":
        for i in range(min(W, H) // 2, 0, -1):
            a = int(opacity * (1 - (i / (min(W, H) / 2))) ** 1.8)
            draw.ellipse(
                [W // 2 - i, H // 2 - i, W // 2 + i, H // 2 + i],
                outline=(0, 0, 0, a),
                width=4,
            )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def compose_text_layer(image_bytes: bytes, quote: Quote, brief: DesignBrief) -> bytes:
    base = _load(image_bytes)
    base = _apply_overlay(base, brief)
    canvas = Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0, 0))
    canvas = _draw_text(canvas, quote, brief, lum_img=base)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)

    # Apply subtle film grain for a more premium, cinematic feel
    return _apply_film_grain(img)


def _apply_film_grain(img: Image.Image, intensity: float = 0.015) -> Image.Image:
    """Adds an extremely subtle high-end film grain texture. Reduced for crystal clarity."""
    import numpy as np

    arr = np.array(img).astype(np.float32)
    grain = np.random.normal(0, 255 * intensity, arr.shape).astype(np.float32)
    arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _to_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()
