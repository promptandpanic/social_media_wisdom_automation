"""
FFmpeg-based Reel composer — Ken Burns zoom + static text overlay + background music.

Layout:
  Background image → Ken Burns continuous zoom throughout
  Text layer       → static RGBA PNG overlaid on top, fades out 2s before end

Audio: background music loop, ducked at end, no TTS.

Public API:
  create_reel(image_bytes, quote, brief, audio_file, duration_sec, music_volume)
      → (video_bytes, thumbnail_bytes)  — both can be None on ffmpeg failure
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import wisdom.config as cfg
from wisdom.composers.card import (
    IMAGE_HEIGHT, IMAGE_WIDTH,
    compose_base, compose_overlay_layer, compose_text_layer,
)
from wisdom.schemas import DesignBrief, Quote

logger = logging.getLogger(__name__)

FPS           = cfg.reel_cfg().get("fps", 30)
FADE_DUR_SEC  = 2.0
BASE_HOLD_SEC = 0.3


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _scale_crop() -> str:
    W, H = IMAGE_WIDTH, IMAGE_HEIGHT
    return f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"


def _zoompan_at(total_frames: int, start_frame: int) -> str:
    W, H = IMAGE_WIDTH, IMAGE_HEIGHT
    return (
        f"zoompan=z='min(1+0.0005*(on+{start_frame}),1.12)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={total_frames}:s={W}x{H}:fps={FPS}"
    )


# ---------------------------------------------------------------------------
# Core reel builder
# ---------------------------------------------------------------------------

def _build_reel(image_bytes: bytes, quote: Quote, brief: DesignBrief,
                audio_file: str, duration_sec: float,
                music_volume: float) -> bytes | None:
    total = duration_sec
    W, H  = IMAGE_WIDTH, IMAGE_HEIGHT

    raw_pil      = compose_base(image_bytes, brief)
    overlay_png  = compose_overlay_layer(brief)
    text_png     = compose_text_layer(image_bytes, quote, brief)
    total_frames = int(total * FPS)

    text_fade_start = total - FADE_DUR_SEC - BASE_HOLD_SEC
    skip_kenburns   = brief.skip_kenburns
    sc = _scale_crop()

    has_audio = Path(audio_file).exists() if audio_file else False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bg_path      = str(tmp / "bg.jpg")
        overlay_path = str(tmp / "overlay.png")
        text_path    = str(tmp / "text.png")
        raw_pil.save(bg_path, format="JPEG", quality=95)
        (tmp / "overlay.png").write_bytes(overlay_png)
        (tmp / "text.png").write_bytes(text_png)

        out_p = str(tmp / "reel.mp4")
        cmd   = ["ffmpeg", "-y"]
        cmd  += ["-loop", "1", "-t", f"{total:.4f}", "-i", bg_path]       # 0: raw bg
        cmd  += ["-loop", "1", "-t", f"{total:.4f}", "-i", overlay_path]  # 1: overlay
        cmd  += ["-loop", "1", "-t", f"{total:.4f}", "-i", text_path]     # 2: text

        audio_idx = None
        if has_audio:
            audio_idx = 3
            cmd += ["-stream_loop", "-1", "-i", str(Path(audio_file).resolve())]

        parts = []
        if not skip_kenburns:
            # Added temporal noise for high-end cinematic 'motion texture' (film grain)
            parts.append(f"[0:v]{sc},{_zoompan_at(total_frames, 0)},noise=alls=12:allf=t+u[bg]")
        else:
            parts.append(f"[0:v]{sc},setsar=1,fps={FPS},noise=alls=12:allf=t+u[bg]")

        # Overlay and text are both fully static — no zoom
        parts.append(f"[1:v]format=rgba,setsar=1,fps={FPS}[ov_static]")
        parts.append(
            f"[2:v]format=rgba,setsar=1,fps={FPS},"
            f"fade=t=out:st={text_fade_start:.3f}:d={FADE_DUR_SEC:.3f}:alpha=1[text_fade]"
        )
        parts.append("[bg][ov_static]overlay=0:0[bg_ov]")
        parts.append("[bg_ov][text_fade]overlay=0:0[vout]")

        if audio_idx is not None:
            fade_st = max(0.0, total - 1.5)
            parts.append(
                f"[{audio_idx}:a]volume={music_volume:.3f},"
                f"afade=t=out:st={fade_st:.2f}:d=1.5[aout]"
            )

        filt = ";".join(parts)
        cmd += ["-filter_complex", filt, "-map", "[vout]"]
        if audio_idx is not None:
            cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-t", str(total), out_p]

        logger.info(f"Reel: {total}s | kenburns={'yes' if not skip_kenburns else 'no'} | music={'yes' if has_audio else 'no'}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                logger.error(f"ffmpeg error:\n{result.stderr[-2000:]}")
                return None
            out = Path(out_p)
            logger.info(f"✓ Reel ready ({out.stat().st_size // 1024}KB)")
            return out.read_bytes()
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out")
            return None
        except Exception as exc:
            logger.error(f"ffmpeg exception: {exc}")
            return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def create_reel(image_bytes: bytes, quote: Quote, brief: DesignBrief,
                audio_file: str = "", duration_sec: float = 23,
                music_volume: float = 0.15) -> tuple[bytes | None, bytes | None]:
    """
    Returns (video_bytes, thumbnail_bytes).
    thumbnail_bytes is always the composed JPEG — used as fallback cover.
    """
    if not _ffmpeg_available():
        logger.warning("ffmpeg not found — skipping Reel creation")
        return None, None

    from wisdom.composers.card import compose_image
    thumbnail = compose_image(image_bytes, quote, brief)

    video = _build_reel(image_bytes, quote, brief, audio_file, duration_sec, music_volume)
    return video, thumbnail
