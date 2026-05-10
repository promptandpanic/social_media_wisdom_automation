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
You are the social media manager for an inspirational quotes channel.
Theme: {theme.name}

The post already opens with this quote and attribution — do NOT repeat or echo it.

Write the body of the Instagram caption that goes BELOW the quote:
1. One strong hook sentence that captures the feeling of the quote (e.g. "Save this for when you feel stuck.", "This one hits different at 2am."). No emojis in the hook.
2. 2 short sentences expanding the idea. Emojis are fine here.
3. One CTA line (e.g. "Tag someone who needs this.", "Drop a comment if this is you.").

Keep it tight — 4 lines total, each separated by a blank line.
Provide exactly 5 niche hashtags relevant to this quote and theme.

Return ONLY valid JSON (use \\n for newlines):
{{"caption": "Hook\\n\\nBody 1.\\n\\nBody 2.\\n\\nCTA", "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]}}

Quote context (do NOT include in output): "{quote.text}" — {quote.author}
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

    if state.get("offline"):
        llm_caption, hashtags = "", theme.hashtags
    else:
        llm_caption, hashtags = _generate_caption_and_tags(quote, theme)

    # Build caption: quote + attribution + body. No quote = body only.
    parts = []
    if text:
        attribution = f"— {author}" if author and author.lower() not in ("original", "unknown") else ""
        parts.append(text + (f"\n{attribution}" if attribution else ""))
    if llm_caption:
        parts.append(llm_caption)

    caption = "\n\n".join(parts)
    snippet = text.split(".")[0][:80] if text else theme.name
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


def _build_email_html(state: PipelineState, theme_name: str) -> str:
    """Build the HTML email body."""
    quote = state.get("quote")
    quote_text = quote.text if quote else "N/A"
    author = quote.author if quote else "Unknown"
    llm_caption = state.get("llm_caption", "")
    results = state.get("platform_results", [])

    has_success = any(r.status == "posted" for r in results)
    has_failure = any(r.status == "failed" for r in results)
    status_text = "LIVE" if not has_failure else "PARTIAL" if has_success else "FAILED"
    status_color = "#2ecc71" if not has_failure else "#e67e22" if has_success else "#e74c3c"
    ist = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
    date_str = f"{ist.strftime('%B %d, %Y').upper()} &nbsp;&bull;&nbsp; {ist.strftime('%A').upper()} &nbsp;&bull;&nbsp; {ist.strftime('%I:%M %p')} IST"

    platforms_html = ""
    for r in results:
        if r.status == "posted":
            label = "Instagram" if r.platform == "instagram" else "YouTube"
            platforms_html += f"""
            <tr>
              <td style="padding: 14px 0; border-bottom: 1px solid #1e3a5f;">
                <span style="font-size:11px; letter-spacing:2px; color:#4a9eba; text-transform:uppercase;">{label}</span>
              </td>
              <td style="padding: 14px 0; border-bottom: 1px solid #1e3a5f; text-align:right;">
                <a href="{r.url}" style="font-size:11px; letter-spacing:1.5px; color:#c9a96e; text-decoration:none; text-transform:uppercase; border-bottom: 1px solid #c9a96e; padding-bottom:2px;">View Post</a>
              </td>
            </tr>"""
        else:
            platforms_html += f"""
            <tr>
              <td style="padding: 14px 0; border-bottom: 1px solid #1e3a5f;">
                <span style="font-size:11px; letter-spacing:2px; color:#4a9eba; text-transform:uppercase;">{r.platform}</span>
              </td>
              <td style="padding: 14px 0; border-bottom: 1px solid #1e3a5f; text-align:right;">
                <span style="font-size:11px; color:#e74c3c; letter-spacing:1px;">Failed</span>
              </td>
            </tr>"""

    caption_html = llm_caption.replace("\n", "<br>") if llm_caption else ""

    live_dot = (
        '<span class="live-dot" style="display:inline-block; width:7px; height:7px; '
        'background:#e74c3c; border-radius:50%; margin-right:6px; vertical-align:middle;"></span>'
        if status_text == "LIVE" else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://fonts.googleapis.com/css2?family=Architects+Daughter&display=swap" rel="stylesheet">
  <style>
    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.15; }} }}
    .live-dot {{ animation: pulse 1.4s ease-in-out infinite; }}
  </style>
</head>
<body style="margin:0; padding:0; background-color:#0a0a0a; font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a; padding: 40px 20px;">
    <tr><td align="center">
      <table width="580" cellpadding="0" cellspacing="0" style="max-width:580px; width:100%;">

        <!-- Top rule -->
        <tr><td style="padding-bottom: 28px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td style="border-top: 1px solid #1e3a5f;"></td>
              <td width="40" style="border-top: 3px solid #c9a96e;"></td>
            </tr>
          </table>
        </td></tr>

        <!-- Masthead -->
        <tr><td style="padding-bottom: 8px;">
          <table width="100%" cellpadding="0" cellspacing="0">
            <tr>
              <td>
                <div style="font-size:28px; letter-spacing:6px; color:#f0f0f0; font-weight:300; text-transform:uppercase; line-height:1;">Wisdom <span style="color:#c9a96e;">Dispatch</span></div>
              </td>
              <td align="right" valign="bottom">
                <div style="font-size:9px; letter-spacing:2px; color:#4a6a7a; text-align:right; line-height:1.8;">
                  {date_str}<br>
                  {live_dot}<span style="color:{status_color}; font-weight:bold; letter-spacing:3px;">{status_text}</span>
                  &nbsp;&bull;&nbsp;
                  <span style="color:#4a9eba;">{theme_name.upper()}</span>
                </div>
              </td>
            </tr>
          </table>
        </td></tr>

        <!-- Rule under masthead -->
        <tr><td style="padding: 20px 0 40px 0;">
          <table width="100%" cellpadding="0" cellspacing="0"><tr>
            <td style="border-top: 1px solid #1e3a5f;"></td>
          </tr></table>
        </td></tr>

        <!-- Quote block -->
        <tr><td style="padding-bottom: 48px;">
          <div style="font-size:9px; letter-spacing:3px; color:#4a9eba; text-transform:uppercase; margin-bottom:24px;">The Insight</div>
          <div style="font-family:'Architects Daughter', cursive; font-size:24px; line-height:1.7; color:#e8e8e8; padding-left:20px; border-left: 2px solid #c9a96e;">{quote_text}</div>
          <div style="margin-top:20px; padding-left:20px; font-size:11px; letter-spacing:3px; color:#c9a96e; text-transform:uppercase;">{author}</div>
        </td></tr>

        <!-- Caption block (only if present) -->
        {'<tr><td style="padding-bottom: 48px;"><div style="font-size:9px; letter-spacing:3px; color:#4a9eba; text-transform:uppercase; margin-bottom:20px;">Caption</div><div style="background:#0f1a24; border: 1px solid #1e3a5f; padding: 24px; font-size:13px; line-height:1.8; color:#8aafbf;">' + caption_html + '</div></td></tr>' if caption_html else ''}

        <!-- Published to -->
        <tr><td style="padding-bottom: 0;">
          <div style="font-size:9px; letter-spacing:3px; color:#4a9eba; text-transform:uppercase; margin-bottom:4px;">Published To</div>
          <table width="100%" cellpadding="0" cellspacing="0">
            {platforms_html if platforms_html else '<tr><td style="padding:14px 0; color:#555; font-size:12px;">No platforms posted.</td></tr>'}
          </table>
        </td></tr>

        <!-- Bottom rule -->
        <tr><td style="padding-top: 28px;">
          <table width="100%" cellpadding="0" cellspacing="0"><tr>
            <td style="border-top: 1px solid #1e3a5f;"></td>
          </tr></table>
          <div style="padding-top:20px; font-size:9px; letter-spacing:4px; color:#4a6a7a; text-transform:uppercase; text-align:center;">
            Wisdom Engine &nbsp;&nbsp;&bull;&nbsp;&nbsp; Publishing Log
          </div>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _send_email_report(state: PipelineState, theme_name: str) -> None:
    """Send the HTML report email."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    sender = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    recipient = os.environ.get("EMAIL_RECIPIENT") or sender

    if not (sender and password and recipient):
        logger.info("Skipping email report (SMTP_USER or SMTP_PASS not set).")
        return

    from email.utils import formataddr

    html = _build_email_html(state, theme_name)
    ist = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
    date_str = ist.strftime("%d-%b-%y")
    quote = state.get("quote")
    author = quote.author if quote else ""
    results = state.get("platform_results", [])
    has_success = any(r.status == "posted" for r in results)
    has_failure = any(r.status == "failed" for r in results)

    if has_failure and not has_success:
        subject = f"[POST FAILED] {theme_name} — Wisdom Dispatch ({date_str})"
    elif has_failure:
        subject = f"[Partial] Wisdom Dispatch ({date_str}) | {theme_name}{' — ' + author if author else ''}"
    else:
        subject = f"Wisdom Dispatch ({date_str}) | {theme_name}{' — ' + author if author else ''}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr(("Wisdom Dispatch", sender))
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
        logger.info(f"Email report sent to {recipient}")
    except Exception as e:
        logger.error(f"Failed to send email report: {e}")
