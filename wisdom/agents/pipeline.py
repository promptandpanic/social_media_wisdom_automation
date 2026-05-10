"""
Top-level pipeline — orchestrates quote → design → media → store → post.

generate_only=True  → stops after store (content ready, not posted)
dry_run=True        → saves locally, never posts
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
import re
import smtplib
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import wisdom.config as cfg
from wisdom.agents import design, media, quote
from wisdom.schemas import PipelineState, PostMeta, ThemeConfig

logger = logging.getLogger(__name__)


def run(theme_key: str, dry_run: bool = False,
        generate_only: bool = False, offline: bool = False) -> PipelineState:
    theme = cfg.theme(theme_key)
    if not theme.enabled:
        logger.info(f"Theme '{theme_key}' is disabled — skipping")
        return {}

    from wisdom.storage.db import ContentDB
    db = ContentDB()
    db.load()

    initial: PipelineState = {
        "theme_key": theme_key,
        "theme": theme,
        "dry_run": dry_run,
        "generate_only": generate_only,
        "offline": offline,
        "quote": None,
        "brief": None,
        "image_bytes": None,
        "composed_image": None,
        "video_bytes": None,
        "thumbnail_bytes": None,
        "design_attempt": 0,
        "best_score": 0,
        "best_state": None,
        "meta": None,
        "pending_id": None,
        "platform_results": [],
        "errors": [],
        "recent_quotes": db.recent_quotes(),
        "recent_styles": db.recent_styles(),
    }

    # ── Step 1: Quote ────────────────────────────────────────────────────────
    logger.info("── Quote generation ──────────────────────────")
    quote_graph = quote.build()
    state = quote_graph.invoke(initial)
    if not state.get("quote"):
        logger.error("Quote generation failed completely")
        return state

    q = state["quote"]
    logger.info(f'Quote: "{q.text[:70]}" — {q.author}')

    # ── Step 2: Design ───────────────────────────────────────────────────────
    logger.info("── Design brief ──────────────────────────────")
    design_graph = design.build()
    state = design_graph.invoke(state)

    # ── Step 3: Media ────────────────────────────────────────────────────────
    logger.info("── Image + compose + judge ───────────────────")
    media_graph = media.build()
    state = media_graph.invoke(state)

    # ── Step 4: Video (if reel format) ──────────────────────────────────────
    if theme.format == "reel":
        logger.info("── Video creation ────────────────────────────")
        state = _create_video(state, theme)

    # ── Step 5: Build post metadata ─────────────────────────────────────────
    state = _build_meta(state, theme)

    # ── Step 6: Store ────────────────────────────────────────────────────────
    logger.info("── Storing assets ────────────────────────────")
    state = _store(state, theme, db)

    if dry_run:
        _save_dry_run(state, theme_key)
        logger.info("✅ Dry-run complete — check output/")
        return state

    if generate_only:
        db.save()
        logger.info("✅ Generated and stored — run 'task post' to publish")
        return state

    # ── Step 7: Post to all platforms ───────────────────────────────────────
    logger.info("── Posting ───────────────────────────────────")
    state = _post(state, theme, db)
    db.save()

    results = state.get("platform_results", [])
    for r in results:
        status_icon = "✅" if r.status == "posted" else "❌" if r.status == "failed" else "⏭"
        logger.info(f"{status_icon} {r.platform}: {r.status} {r.post_id or r.error}")

    _send_email_report(state, theme.name)
    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_audio_file(theme_key: str) -> str:
    audio_dir = Path("assets/audio")
    if not audio_dir.exists():
        return ""
    # Try theme-specific audio first
    candidates = list(audio_dir.glob(f"{theme_key}_*.mp3"))
    if candidates:
        return str(random.choice(candidates))
    
    # Fallback to ANY available audio for variety, instead of just one file
    all_audio = list(audio_dir.glob("*.mp3"))
    if all_audio:
        return str(random.choice(all_audio))
        
    return ""


def _create_video(state: PipelineState, theme: ThemeConfig) -> PipelineState:
    from wisdom.composers.reel import create_reel
    image_bytes = state.get("image_bytes", b"")   # raw image — no overlay, no text
    composed = state.get("composed_image", b"")    # full composite — thumbnail only
    brief = state.get("brief")
    quote = state.get("quote")
    reel_cfg = cfg.reel_cfg()

    try:
        video_bytes, thumbnail_bytes = create_reel(
            image_bytes=image_bytes,
            quote=quote,
            brief=brief,
            audio_file=_select_audio_file(theme.key),
            duration_sec=reel_cfg.get("duration_sec", 23),
            music_volume=reel_cfg.get("music_volume", 0.15),
        )
        return {**state, "video_bytes": video_bytes,
                "thumbnail_bytes": thumbnail_bytes or composed}
    except Exception as exc:
        logger.warning(f"Video creation failed ({exc}) — will post image only")
        return {**state, "thumbnail_bytes": composed}


def _generate_caption_and_tags(quote, theme: ThemeConfig) -> tuple[str, list[str]]:
    if not quote:
        return "", theme.hashtags
    prompt = f"""
You are the social media manager for an inspirational channel.
Quote: "{quote.text}" - {quote.author}
Theme: {theme.name}

Write a highly engaging Instagram caption based on this quote.
Format requirements:
1. Start with a strong, relatable hook (e.g., "Save this for when you need a reminder 📌", "Read this if you're feeling lost right now ⬇️").
2. 2-3 short, spaced-out sentences expanding on the quote (use emojis naturally).
3. End with a strong Call-To-Action (CTA) (e.g., "Drop a 💯 if you agree", "Tag someone who needs to hear this 👇").
4. Provide exactly 5 highly relevant, niche hashtags.

Return ONLY valid JSON in this format (use \\n for newlines):
{{"caption": "Hook\\n\\nBody sentence 1.\\n\\nBody sentence 2.\\n\\nCTA", "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]}}
"""
    try:
        from wisdom import providers
        raw = providers.llm.generate(prompt, role="quote_generation")
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return data.get("caption", ""), data.get("hashtags", theme.hashtags)
    except Exception as exc:
        logger.warning(f"Caption generation failed: {exc}")
    return "", theme.hashtags


def _build_meta(state: PipelineState, theme: ThemeConfig) -> PipelineState:
    quote = state.get("quote")
    text = quote.text if quote else ""
    author = quote.author if quote else ""

    caption_lines = [text]
    if author and author.lower() not in ("original", "unknown"):
        caption_lines.append(f"— {author}")

    if state.get("offline"):
        llm_caption, hashtags = "", theme.hashtags
    else:
        llm_caption, hashtags = _generate_caption_and_tags(quote, theme)

    if llm_caption:
        caption_lines.append(f"\n{llm_caption}")

    caption = "\n".join(caption_lines)
    snippet = text.split(".")[0][:80]
    title = f"{snippet} | {theme.name}"

    meta = PostMeta(
        caption=caption,
        title=title,
        hashtags=hashtags,
        tags=[t.lstrip("#") for t in hashtags],
        theme=theme.key,
    )
    return {**state, "meta": meta, "llm_caption": llm_caption}


def _store(state: PipelineState, theme: ThemeConfig, db) -> PipelineState:
    pending_dir = Path(cfg.app()["pending_dir"]) / str(uuid.uuid4())
    pending_dir.mkdir(parents=True, exist_ok=True)

    composed = state.get("composed_image", b"")
    video = state.get("video_bytes")
    thumbnail = state.get("thumbnail_bytes", composed)
    quote = state.get("quote")
    meta = state.get("meta")

    (pending_dir / "image.jpg").write_bytes(composed)
    if video:
        (pending_dir / "video.mp4").write_bytes(video)
    (pending_dir / "thumb.jpg").write_bytes(thumbnail)

    record_id = db.create_pending(
        theme=theme.key,
        quote={"text": quote.text, "author": quote.author,
               "highlight": quote.highlight, "source": quote.source} if quote else {},
        caption=meta.caption if meta else "",
        asset_dir=str(pending_dir),
    )
    return {**state, "pending_id": record_id}


def _post(state: PipelineState, theme: ThemeConfig, db) -> PipelineState:
    from wisdom.platforms.instagram import InstagramPlatform
    from wisdom.platforms.youtube import YouTubePlatform

    composed = state.get("composed_image", b"")
    video = state.get("video_bytes")
    thumbnail = state.get("thumbnail_bytes", composed)
    meta = state.get("meta")
    pending_id = state.get("pending_id")
    results = []

    platform_map = {
        "instagram": InstagramPlatform(),
        "youtube": YouTubePlatform(),
    }

    for platform_name in theme.platforms:
        platform = platform_map.get(platform_name)
        if not platform:
            logger.warning(f"Unknown platform: {platform_name}")
            continue
        if not platform.available():
            logger.warning(f"{platform_name}: credentials not set — skipping")
            continue

        logger.info(f"Posting to {platform_name}…")
        if video and platform_name == "youtube":
            result = platform.post_video(video, thumbnail, meta, theme=theme)
        elif video:
            result = platform.post_video(video, thumbnail, meta)
        else:
            result = platform.post_image(composed, meta)

        results.append(result)
        if pending_id:
            db.update_platform_status(
                pending_id, platform_name, result.status,
                post_id=result.post_id, error=result.error,
            )

    if pending_id and all(r.status == "posted" for r in results):
        db.promote_to_posted(pending_id)
        quote = state.get("quote")
        brief = state.get("brief")
        db.mark_posted(
            {"text": quote.text if quote else "", "author": getattr(quote, "author", "")},
            theme.key,
            style=brief.style if brief else "",
        )

    return {**state, "platform_results": results}


def _save_dry_run(state: PipelineState, theme_key: str) -> None:
    out = Path(cfg.app()["output_dir"])
    out.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    composed = state.get("composed_image", b"")
    video = state.get("video_bytes")
    quote = state.get("quote")
    brief = state.get("brief")
    meta = state.get("meta")

    if composed:
        (out / f"{theme_key}_{stamp}.jpg").write_bytes(composed)
    if video:
        (out / f"{theme_key}_{stamp}.mp4").write_bytes(video)

    logger.info(f'[DRY_RUN] Quote: "{getattr(quote, "text", "")}"')
    if brief:
        logger.info(f"[DRY_RUN] Style={brief.style} Font={brief.font} Layout={brief.layout}")
    if meta:
        print("\n" + "─" * 60)
        print("CAPTION:")
        print(meta.caption)
        print("─" * 60)


def _send_email_report(state: PipelineState, theme_name: str) -> None:
    """Send a premium, aesthetic HTML report of the post results."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    
    sender = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    recipient = os.environ.get("EMAIL_RECIPIENT") or sender

    if not (sender and password and recipient):
        logger.info("Skipping email report (SMTP_USER or SMTP_PASS not set).")
        return

    quote = state.get("quote")
    quote_text = quote.text if quote else "N/A"
    author = quote.author if quote else "Unknown"
    meta = state.get("meta")
    caption = meta.caption if meta else "N/A"
    results = state.get("platform_results", [])
    
    # Status Badge Logic
    has_success = any(r.status == "posted" for r in results)
    has_failure = any(r.status == "failed" for r in results)
    
    status_color = "#27ae60" if not has_failure else "#e67e22" if has_success else "#c0392b"
    status_text = "SUCCESS" if not has_failure else "PARTIAL SUCCESS" if has_success else "FAILED"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f7f9; margin: 0; padding: 40px 20px; }}
        .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.05); }}
        .header {{ background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d); padding: 30px; text-align: center; color: white; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; letter-spacing: 1px; background: rgba(255,255,255,0.2); margin-top: 10px; border: 1px solid rgba(255,255,255,0.3); }}
        .content {{ padding: 40px; }}
        .section-title {{ font-size: 14px; font-weight: bold; color: #888; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
        .quote-card {{ background: #f8faff; border-left: 4px solid #4A90E2; padding: 25px; margin-bottom: 30px; border-radius: 0 8px 8px 0; }}
        .quote-text {{ font-size: 20px; line-height: 1.5; color: #2c3e50; font-style: italic; margin: 0; }}
        .quote-author {{ margin-top: 15px; font-weight: bold; color: #7f8c8d; font-size: 15px; }}
        .caption-box {{ background: #ffffff; border: 1px solid #e1e8ed; border-radius: 8px; padding: 20px; color: #333; font-size: 14px; line-height: 1.6; white-space: pre-wrap; }}
        .platform-row {{ display: flex; align-items: center; justify-content: space-between; padding: 15px 0; border-bottom: 1px solid #f0f0f0; }}
        .platform-info {{ display: flex; align-items: center; }}
        .platform-icon {{ font-size: 20px; margin-right: 12px; }}
        .btn {{ display: inline-block; padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: bold; text-decoration: none; transition: all 0.2s; }}
        .btn-insta {{ background: #E1306C; color: white; }}
        .btn-youtube {{ background: #FF0000; color: white; }}
        .footer {{ text-align: center; padding: 30px; font-size: 12px; color: #aaa; background: #fafafa; }}
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1 style="margin:0; font-size: 24px; letter-spacing: 1px; font-weight: 300;">WISDOM DISPATCH</h1>
          <div class="status-badge">{status_text} | {theme_name.upper()}</div>
        </div>
        
        <div class="content">
          <div class="section-title">The Insight</div>
          <div class="quote-card">
            <p class="quote-text">"{quote_text}"</p>
            <div class="quote-author">— {author}</div>
          </div>

          <div class="section-title">Content Manifest</div>
          <div class="caption-box">{caption}</div>

          <div class="section-title" style="margin-top: 30px;">Digital Assets</div>
          <div style="margin-top: 10px;">
    """

    for r in results:
        btn_class = "btn-insta" if r.platform == "instagram" else "btn-youtube"
        label = "INSTAGRAM" if r.platform == "instagram" else "YOUTUBE"
        
        if r.status == "posted":
            html += f"""
            <div class="platform-row">
              <div class="platform-info">
                <span style="font-weight:bold; color:#444; font-size: 12px; letter-spacing: 1px;">{r.platform.upper()}</span>
              </div>
              <a href="{r.url}" class="btn {btn_class}">{label}</a>
            </div>
            """
        else:
            html += f"""
            <div class="platform-row" style="opacity: 0.6;">
              <div class="platform-info">
                <span style="font-weight:bold; color:#c0392b; font-size: 12px; letter-spacing: 1px;">{r.platform.upper()} FAILED</span>
              </div>
              <span style="font-size: 11px; color: #c0392b;">{r.error}</span>
            </div>
            """

    html += """
          </div>
        </div>
        <div class="footer">
          WISDOM ENGINE &bull; AUTOMATED INTELLIGENCE &bull; PUBLISHING LOG
        </div>
      </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    date_str = datetime.datetime.now().strftime("%d-%b-%y")
    msg["Subject"] = f"Wisdom Dispatch ({date_str}) | {theme_name}"
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=15)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        logger.info(f"High-grade email report sent to {recipient}")
    except Exception as e:
        logger.error(f"Failed to send email report: {e}")
