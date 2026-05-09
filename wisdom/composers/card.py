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
from PIL import Image, ImageDraw, ImageFont

import wisdom.config as cfg
from wisdom.schemas import DesignBrief, Quote

logger = logging.getLogger(__name__)

FONTS_DIR = Path("assets/fonts")

# ----- canvas constants (from config) -----
_img = cfg.image_cfg()
IMAGE_WIDTH  = _img["width"]
IMAGE_HEIGHT = _img["height"]

# Instagram mobile safe zones (1080×1920):
#   Right: action icons ~150px, Bottom: caption bar ~420px
MARGIN_X          = 72
MARGIN_X_R        = 180
TEXT_MAX_W        = IMAGE_WIDTH - MARGIN_X - MARGIN_X_R
TEXT_ZONE_CX      = (MARGIN_X + IMAGE_WIDTH - MARGIN_X_R) // 2
INSTAGRAM_SAFE_BOTTOM = IMAGE_HEIGHT - 420

_ZONE_MAX_H = {
    "top":    int(IMAGE_HEIGHT * 0.55),
    "center": int(IMAGE_HEIGHT * 0.72),
    "bottom": INSTAGRAM_SAFE_BOTTOM - int(IMAGE_HEIGHT * 0.28),
}

# ---------------------------------------------------------------------------
# Font registry
# ---------------------------------------------------------------------------

_FONT_URLS: dict[str, tuple[str, str]] = {
    "bebas":        ("bebas.ttf",             "https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf"),
    "oswald":       ("oswald_bold.ttf",       "https://fonts.gstatic.com/s/oswald/v57/TK3_WkUHHAIjg75cFRf3bXL8LICs1xZogUE.ttf"),
    "montserrat":   ("montserrat_bold.ttf",   "https://fonts.gstatic.com/s/montserrat/v31/JTUHjIg1_i6t8kCHKm4532VJOt5-QNFgpCuM70w-.ttf"),
    "raleway":      ("raleway_bold.ttf",      "https://fonts.gstatic.com/s/raleway/v37/1Ptxg8zYS_SKggPN4iEgvnHyvveLxVs9pYCP.ttf"),
    "anton":        ("anton.ttf",             "https://fonts.gstatic.com/s/anton/v27/1Ptgg87LROyAm0K0.ttf"),
    "cinzel":       ("cinzel_bold.ttf",       "https://fonts.gstatic.com/s/cinzel/v26/8vIU7ww63mVu7gtR-kwKxNvkNOjw-jHgTYo.ttf"),
    "josefin":      ("josefin_bold.ttf",      "https://fonts.gstatic.com/s/josefinsans/v34/Qw3PZQNVED7rKGKxtqIqX5E-AVSJrOCfjY46_N_XXME.ttf"),
    "fjalla":       ("fjalla.ttf",            "https://github.com/google/fonts/raw/main/ofl/fjallaone/FjallaOne-Regular.ttf"),
    "poppins":      ("poppins_bold.ttf",      "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf"),
    "poppins_light":("poppins_light.ttf",     "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Light.ttf"),
    "nunito":       ("nunito_bold.ttf",       "https://fonts.gstatic.com/s/nunito/v32/XRXI3I6Li01BKofiOc5wtlZ2di8HDOUhRTM.ttf"),
    "nunito_light": ("nunito_light.ttf",      "https://fonts.gstatic.com/s/nunito/v32/XRXI3I6Li01BKofiOc5wtlZ2di8HDFwmRTM.ttf"),
    "jost":         ("jost_bold.ttf",         "https://fonts.gstatic.com/s/jost/v20/92zPtBhPNqw79Ij1E865zBUv7mxEIgVG.ttf"),
    "worksans":     ("worksans_bold.ttf",     "https://fonts.gstatic.com/s/worksans/v24/QGY_z_wNahGAdqQ43RhVcIgYT2Xz5u32K67QNig.ttf"),
    "abril":        ("abril_fatface.ttf",     "https://fonts.gstatic.com/s/abrilfatface/v25/zOL64pLDlL1D99S8g8PtiKchm-A.ttf"),
    "playfair":     ("playfair_bold.ttf",     "https://fonts.gstatic.com/s/playfairdisplay/v40/nuFRD-vYSZviVYUb_rj3ij__anPXDTnCjmHKM4nYO7KN_qiTbtY.ttf"),
    "playfair_it":  ("playfair_it.ttf",       "https://fonts.gstatic.com/s/playfairdisplay/v40/nuFvD-vYSZviVYUb_rj3ij__anPXJzDwcbmjWBN2PKeiukDQ.ttf"),
    "merriweather": ("merriweather_bold.ttf", "https://fonts.gstatic.com/s/merriweather/v33/u-4D0qyriQwlOrhSvowK_l5UcA6zuSYEqOzpPe3HOZJ5eX1WtLaQwmYiScCmDxhtNOKl8yDrOSAqEw.ttf"),
    "cormorant":    ("cormorant_bold.ttf",    "https://fonts.gstatic.com/s/cormorantgaramond/v21/co3smX5slCNuHLi8bLeY9MK7whWMhyjYrGFEsdtdc62E6zd5LDfOjw.ttf"),
    "spectral":     ("spectral_bold.ttf",     "https://github.com/google/fonts/raw/main/ofl/spectral/Spectral-Bold.ttf"),
    "vollkorn":     ("vollkorn_bold.ttf",     "https://fonts.gstatic.com/s/vollkorn/v30/0ybgGDoxxrvAnPhYGzMlQLzuMasz6Df213auGQ.ttf"),
    "dancing":      ("dancing.ttf",           "https://fonts.gstatic.com/s/dancingscript/v29/If2cXTr6YS-zF4S-kcSWSVi_sxjsohD9F50Ruu7B1i0HTQ.ttf"),
    "satisfy":      ("satisfy.ttf",           "https://fonts.gstatic.com/s/satisfy/v22/rP2Hp2yn6lkG50LoOZQ.ttf"),
    "pacifico":     ("pacifico.ttf",          "https://github.com/google/fonts/raw/main/ofl/pacifico/Pacifico-Regular.ttf"),
    "caveat":       ("caveat_bold.ttf",       "https://fonts.gstatic.com/s/caveat/v23/WnznHAc5bAfYB2QRah7pcpNvOx-pjRV6SII.ttf"),
    "kalam":        ("kalam.ttf",             "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf"),
    "indieflower":  ("indieflower.ttf",       "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf"),
    "specialelite": ("specialelite.ttf",      "https://fonts.gstatic.com/s/specialelite/v20/XLYgIZbkc4JPUL5CVArUVL0nhnc.ttf"),
    "lato":         ("lato.ttf",              "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf"),
    "lato_bold":    ("lato_bold.ttf",         "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf"),
    "lato_light":   ("lato_light.ttf",        "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Light.ttf"),
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
            lato_path = FONTS_DIR / _FONT_URLS["lato"][0]
            try:
                logger.warning(f"Font '{key}' unavailable — falling back to lato")
                _font_cache[ck] = ImageFont.truetype(str(lato_path), size)
            except Exception:
                _font_cache[ck] = ImageFont.truetype(str(FONTS_DIR / _FONT_URLS["lato_bold"][0]), size)
    return _font_cache[ck]


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple:
    m = re.search(r'#([0-9A-Fa-f]{6})', str(hex_color))
    if m:
        h = m.group(1)
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (255, 200, 50)


def _tw(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return draw.textbbox((0, 0), text, font=font)[2]


def _wrap_words(words: list[str], font: ImageFont.FreeTypeFont,
                max_width: int) -> list[str]:
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


def pixel_wrap(text: str, font: ImageFont.FreeTypeFont,
               max_width: int = TEXT_MAX_W,
               keep_phrase: str = "") -> list[str]:
    words = text.split()
    lines = _wrap_words(words, font, max_width) or [text]

    if not keep_phrase:
        return lines

    pl = keep_phrase.lower()
    if any(pl in ln.lower() for ln in lines):
        return lines

    import string
    _strip = string.punctuation
    pwords = [w.lower().strip(_strip) for w in keep_phrase.split()]
    wnorm  = [w.lower().strip(_strip) for w in words]
    n = len(pwords)
    start = -1
    for i in range(len(wnorm) - n + 1):
        if wnorm[i:i + n] == pwords:
            start = i
            break
    if start <= 0:
        return lines

    phrase_line = " ".join(words[start:start + n])
    pb = font.getbbox(phrase_line)
    if pb[2] - pb[0] > max_width:
        return lines

    left  = _wrap_words(words[:start], font, max_width)
    right = _wrap_words(words[start:], font, max_width)
    if right and pl in right[0].lower():
        return left + right
    return lines


def _layout_lines(disp_text: str, font: ImageFont.FreeTypeFont,
                  layout: str, keep_phrase: str = "") -> list[str]:
    if layout == "sentence_reveal":
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', disp_text.strip())
                     if s.strip()]
        if len(sentences) > 1:
            lines: list[str] = []
            for sent in sentences:
                kp = keep_phrase if (keep_phrase and keep_phrase.lower() in sent.lower()) else ""
                lines.extend(pixel_wrap(sent, font, keep_phrase=kp))
            return lines
    return pixel_wrap(disp_text, font, keep_phrase=keep_phrase)


# Fonts that need bigger starting sizes for legibility
_FONT_SIZE_SCALE: dict[str, float] = {
    "poppins_light": 1.25,
    "nunito_light":  1.25,
    "lato_light":    1.20,
    "cormorant":     1.15,
    "dancing":       1.15,
    "satisfy":       1.12,
    "indieflower":   1.15,
    "caveat":        1.12,
    "kalam":         1.10,
    "playfair":      1.08,
    "playfair_it":   1.12,
    "merriweather":  1.05,
    "spectral":      1.05,
    "vollkorn":      1.05,
    "cinzel":        1.05,
    "specialelite":  1.05,
    "pacifico":      0.92,
    "montserrat":    0.95,
}


def _fit_text(disp_text: str, font_key: str, font_size: int,
              layout: str, zone: str,
              keep_phrase: str = "") -> tuple[list[str], ImageFont.FreeTypeFont, int]:
    scale = _FONT_SIZE_SCALE.get(font_key, 1.0)
    font_size = max(64, int(font_size * scale))
    max_h = _ZONE_MAX_H.get(zone, int(IMAGE_HEIGHT * 0.70))
    for size in range(font_size, 62, -2):
        f = _font(font_key, size)
        lines = _layout_lines(disp_text, f, layout, keep_phrase=keep_phrase)
        if len(lines) * int(size * 1.28) <= max_h:
            return lines, f, size
    size = 64
    f = _font(font_key, size)
    return _layout_lines(disp_text, f, layout, keep_phrase=keep_phrase), f, size


def _sanitize(text: str) -> str:
    import unicodedata
    text = (text
            .replace('—', ' - ').replace('–', '-').replace('…', '...')
            .replace('’', "'").replace('‘', "'")
            .replace('“', '"').replace('”', '"')
            .replace('é', 'e').replace('è', 'e').replace('ê', 'e')
            .replace('à', 'a').replace('â', 'a')
            .replace('ô', 'o').replace('û', 'u'))
    return "".join(
        ch for ch in text
        if ord(ch) < 0x10000 and unicodedata.category(ch) not in ("So", "Cs")
    )


def _split_at_phrase(line: str, phrase: str) -> tuple[str, str, str] | None:
    idx = line.lower().find(phrase.lower())
    if idx == -1:
        return None
    return line[:idx], line[idx:idx + len(phrase)], line[idx + len(phrase):]


_CURSIVE_FONTS = {"dancing", "satisfy", "pacifico", "caveat", "kalam", "indieflower"}


def _highlight_font(base_key: str, hi_style: str, size: int) -> ImageFont.FreeTypeFont:
    if hi_style in ("italic", "caps_italic"):
        return _font("playfair_it", size)
    if hi_style == "script":
        return _font("dancing", size)
    return _font(base_key, size)


def _stroke_text(draw, xy, text, font, fill, stroke_color=(0, 0, 0), stroke=3):
    x, y = xy
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    draw.text((x, y), text, font=font, fill=fill)


def _render_line(draw, img_width: int, y: int, line: str,
                 font: ImageFont.FreeTypeFont, fill: tuple,
                 hi_font: ImageFont.FreeTypeFont, hi_fill: tuple,
                 hi_phrase: str, hi_style: str, stroke: int = 3) -> None:
    segs = _split_at_phrase(line, hi_phrase) if hi_phrase else None

    if segs is None:
        bb = font.getbbox(line)
        x = (img_width - (bb[2] - bb[0])) // 2
        _stroke_text(draw, (x, y), line, font, fill=fill, stroke=stroke)
        return

    before, match, after = segs
    hi_disp = match.upper() if hi_style == "caps" else match

    bw = (font.getbbox(before)[2] - font.getbbox(before)[0]) if before else 0
    hw = (hi_font.getbbox(hi_disp)[2] - hi_font.getbbox(hi_disp)[0]) if hi_disp else 0
    aw = (font.getbbox(after)[2] - font.getbbox(after)[0]) if after else 0
    x = (img_width - bw - hw - aw) // 2

    hi_stroke = min(1, stroke)
    if before:
        _stroke_text(draw, (x, y), before, font, fill=fill, stroke=stroke)
        x += bw
    if hi_disp:
        _stroke_text(draw, (x, y), hi_disp, hi_font, fill=hi_fill, stroke=hi_stroke)
        if hi_style == "underline":
            uy = y + hi_font.getbbox(hi_disp)[3] + 2
            draw.rectangle([(x, uy), (x + hw, uy + 3)], fill=(*hi_fill, 255))
        x += hw
    if after:
        _stroke_text(draw, (x, y), after, font, fill=fill, stroke=stroke)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _gradient_rect(img: Image.Image, y0: int, y1: int,
                   color: tuple = (0, 0, 0), max_alpha: int = 180) -> Image.Image:
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    height = y1 - y0
    for y in range(y0, y1):
        a = int(max_alpha * ((y - y0) / height) ** 0.55)
        draw.line([(0, y), (IMAGE_WIDTH, y)], fill=(*color, a))
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _gradient_rect_from_top(img: Image.Image, y0: int, y1: int,
                             color: tuple = (0, 0, 0), max_alpha: int = 180) -> Image.Image:
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    height = y1 - y0
    for y in range(y0, y1):
        frac = (y - y0) / height
        a = int(max_alpha * (1 - frac) ** 0.55)
        draw.line([(0, y), (IMAGE_WIDTH, y)], fill=(*color, a))
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _solid_overlay(img: Image.Image, opacity: int,
                   color: tuple = (0, 0, 0)) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (*color, opacity))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _vignette(img: Image.Image, intensity: int = 160) -> Image.Image:
    w, h = img.size
    rgba = img.convert("RGBA")
    vig = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vig)
    for i in range(min(w, h) // 2, 0, -1):
        alpha = int(intensity * (1 - (i / (min(w, h) / 2))) ** 1.8)
        draw.ellipse([w//2 - i, h//2 - i, w//2 + i, h//2 + i],
                     outline=(0, 0, 0, alpha), width=4)
    return Image.alpha_composite(rgba, vig).convert("RGB")


def _apply_overlay(img: Image.Image, brief: DesignBrief) -> Image.Image:
    ov = brief.overlay
    otype   = ov.type
    opacity = ov.opacity
    color   = _hex_to_rgb(ov.color)

    if otype == "gradient_bottom":
        return _gradient_rect(img, int(IMAGE_HEIGHT * 0.20), IMAGE_HEIGHT,
                              color=color, max_alpha=opacity)
    elif otype == "gradient_top":
        return _gradient_rect_from_top(img, 0, int(IMAGE_HEIGHT * 0.60),
                                       color=color, max_alpha=opacity)
    elif otype == "gradient_center":
        img = _gradient_rect(img, int(IMAGE_HEIGHT * 0.25), int(IMAGE_HEIGHT * 0.78),
                             color=color, max_alpha=opacity)
        return _gradient_rect_from_top(img, int(IMAGE_HEIGHT * 0.22),
                                       int(IMAGE_HEIGHT * 0.52),
                                       color=color, max_alpha=opacity // 2)
    elif otype == "solid":
        return _solid_overlay(img, opacity, color=color)
    elif otype == "vignette":
        return _vignette(img, intensity=opacity)
    return img  # "none"


def _bg_luminance(img: Image.Image, zone: str) -> float:
    W, H = IMAGE_WIDTH, IMAGE_HEIGHT
    if zone == "top":
        box = (MARGIN_X, int(H * 0.06), W - MARGIN_X, int(H * 0.45))
    elif zone == "center":
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
    dark_cr  = _contrast_ratio(20, bg_lum)
    return (255, 255, 255) if white_cr >= dark_cr else (20, 20, 20)




# ---------------------------------------------------------------------------
# Core text renderer
# ---------------------------------------------------------------------------

def _draw_text(img: Image.Image, quote: Quote, brief: DesignBrief,
               n_lines: int | None = None,
               lum_img: Image.Image | None = None) -> Image.Image:
    _QC = '"\'""\'\'«»„‟'
    text       = _sanitize(quote.text).strip(_QC).strip()
    author     = quote.author
    font_key   = brief.font
    if font_key == "playfair_it":
        font_key = "playfair"
    txt_color  = _hex_to_rgb(brief.text_color)
    hi_color   = _hex_to_rgb(brief.highlight_color)
    text_zone  = brief.text_zone
    decoration = brief.decoration
    layout     = brief.layout
    hi_style   = brief.highlight_style
    hi_phrase  = _sanitize(brief.highlight or "").strip(_QC).strip().lower()

    upper     = font_key == "bebas"
    disp_text = text.upper() if upper else text

    bg_lum    = _bg_luminance(lum_img if lum_img is not None else img, text_zone)
    txt_color = _ensure_readable(txt_color, bg_lum)

    font_size = max(64, brief.font_size)
    all_lines, f, font_size = _fit_text(disp_text, font_key, font_size, layout, text_zone,
                                        keep_phrase=hi_phrase)
    hi_f = _highlight_font(font_key, hi_style, font_size)

    if hi_phrase:
        on_line = any(hi_phrase in ln.lower() for ln in all_lines)
        if not on_line:
            logger.warning(f'  highlight not rendered: "{hi_phrase}"')

    lines = all_lines[:n_lines] if n_lines is not None else all_lines
    line_h = int(font_size * 1.28)
    block_h = len(all_lines) * line_h

    if text_zone == "top":
        y = int(IMAGE_HEIGHT * 0.08)
    elif text_zone == "center":
        y = (IMAGE_HEIGHT - block_h) // 2
    else:
        y = INSTAGRAM_SAFE_BOTTOM - block_h

    draw = ImageDraw.Draw(img)

    if decoration == "rule":
        draw.line([(MARGIN_X, y - 20), (IMAGE_WIDTH - MARGIN_X_R, y - 20)],
                  fill=(*hi_color, 220), width=3)
    elif decoration == "quote_mark":
        dq_font = _font("playfair", 260)
        draw.text((MARGIN_X - 44, y - 40), "“", font=dq_font, fill=(*hi_color, 30))

    text_stroke = 0 if font_key in _CURSIVE_FONTS else 3

    for line in lines:
        has_hi = bool(hi_phrase and hi_phrase in line.lower())
        _render_line(
            draw, TEXT_ZONE_CX * 2, y, line,
            font=f, fill=txt_color,
            hi_font=hi_f, hi_fill=hi_color,
            hi_phrase=hi_phrase if has_hi else "",
            hi_style=hi_style,
            stroke=text_stroke,
        )
        y += line_h

    _SKIP_AUTHOR = {"unknown", "anonymous", "original"}
    all_visible = n_lines is None or n_lines >= len(all_lines)
    if (all_visible and author
            and author.lower() not in _SKIP_AUTHOR
            and not author.startswith("@")):
        try:
            ac = tuple(int(brief.author_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            ac = (204, 204, 204)
        a_font    = _font("lato_light", 38)
        dash_font = _font("lato_light", 38)
        dash = "— "
        d_bb = dash_font.getbbox(dash)
        n_bb = a_font.getbbox(author)
        total_w = (d_bb[2] - d_bb[0]) + (n_bb[2] - n_bb[0])
        ax = (TEXT_ZONE_CX * 2 - total_w) // 2
        ay = y + 16
        draw.text((ax, ay), dash, font=dash_font, fill=hi_color)
        draw.text((ax + (d_bb[2] - d_bb[0]), ay), author, font=a_font, fill=ac)


    return img


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_reveal_counts(quote: Quote, brief: DesignBrief) -> list[int]:
    font_key  = brief.font
    if font_key == "playfair_it":
        font_key = "playfair"
    font_size = max(64, brief.font_size)
    text      = _sanitize(quote.text)
    upper     = font_key == "bebas"
    disp_text = text.upper() if upper else text
    zone      = brief.text_zone

    all_lines, f, _ = _fit_text(disp_text, font_key, font_size, "sentence_reveal", zone)

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', disp_text.strip())
                 if s.strip()]

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


def compose_partial(image_bytes: bytes, quote: Quote, brief: DesignBrief,
                    n_lines: int) -> bytes:
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
            draw.ellipse([W//2 - i, H//2 - i, W//2 + i, H//2 + i],
                         outline=(0, 0, 0, a), width=4)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def compose_text_layer(image_bytes: bytes, quote: Quote, brief: DesignBrief) -> bytes:
    base   = _load(image_bytes)
    base   = _apply_overlay(base, brief)
    canvas = Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HEIGHT), (0, 0, 0, 0))
    canvas = _draw_text(canvas, quote, brief, lum_img=base)
    buf    = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
    return img


def _to_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()
