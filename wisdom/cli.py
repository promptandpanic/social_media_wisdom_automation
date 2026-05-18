"""
Click CLI for the social_media_wisdom_automation pipeline.

Commands:
  run         full pipeline (generate + post)
  generate    generate + store, skip posting
  post        post a pending record by ID (or oldest pending)
  dry-run     generate locally, save to output/, never post
  validate    check config files are valid
  youtube-auth  one-time OAuth2 flow to get refresh token
  themes      list enabled themes
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@click.group()
def cli():
    """social_media_wisdom_automation — automated quote Reel pipeline."""


@cli.command()
@click.argument("theme")
@click.option("--dry-run", is_flag=True, help="Save locally, never post")
@click.option("--generate-only", is_flag=True, help="Stop after storing, don't post")
@click.option("--offline", is_flag=True, help="Bypass LLM, use local curated quotes")
def run(theme: str, dry_run: bool, generate_only: bool, offline: bool):
    """Full pipeline for THEME (generate → design → media → post)."""
    from wisdom.agents.pipeline import run as _run

    state = _run(theme, dry_run=dry_run, generate_only=generate_only, offline=offline)
    results = state.get("platform_results", [])
    if not results and not dry_run and not generate_only:
        click.echo("No platforms posted — check logs", err=True)
        sys.exit(1)


@cli.command("generate")
@click.argument("theme")
@click.option("--offline", is_flag=True, help="Bypass LLM, use local curated quotes")
def generate_cmd(theme: str, offline: bool):
    """Generate content for THEME and store as pending (don't post)."""
    from wisdom.agents.pipeline import run as _run

    _run(theme, generate_only=True, offline=offline)
    click.echo("✓ Generated and stored. Run 'task post' to publish.")


@cli.command()
@click.argument("theme")
@click.option("--pending-id", default=None, help="Specific pending record UUID")
def post(theme: str, pending_id: str | None):
    """Post a pending record for THEME (or oldest if no ID given)."""
    from wisdom.storage.db import ContentDB
    from wisdom.agents.pipeline import _post as _do_post
    import wisdom.config as cfg

    db = ContentDB()
    db.load()

    if pending_id:
        record = db.get_pending_by_id(pending_id)
    else:
        records = db.get_pending(theme)
        if not records:
            click.echo(f"No pending records for theme '{theme}'", err=True)
            sys.exit(1)
        record = records[0]

    theme_cfg = cfg.theme(theme)
    state = {
        "theme_key": theme,
        "theme": theme_cfg,
        "pending_id": record["id"],
        "composed_image": Path(record["asset_dir"] + "/image.jpg").read_bytes(),
        "video_bytes": _read_optional(record["asset_dir"] + "/video.mp4"),
        "thumbnail_bytes": _read_optional(record["asset_dir"] + "/thumb.jpg"),
        "meta": _meta_from_record(record, theme_cfg),
        "platform_results": [],
    }
    _do_post(state, theme_cfg, db)
    db.save()


@cli.command("dry-run")
@click.argument("theme")
@click.option("--offline", is_flag=True, help="Bypass LLM, use local curated quotes")
def dry_run_cmd(theme: str, offline: bool):
    """Generate for THEME, save locally to output/, never post."""
    # Use the full fallback chain unless overridden by the shell environment

    from wisdom.agents.pipeline import run as _run

    _run(theme, dry_run=True, offline=offline)


@cli.command()
def validate():
    """Parse all YAML config files and report any errors."""
    import wisdom.config as cfg

    errors = []
    checks = [
        ("app", lambda: cfg.app()),
        ("themes", lambda: cfg.themes()),
        ("llm", lambda: cfg.llm_providers()),
        ("image", lambda: cfg.image_providers()),
        ("topics", lambda: cfg.topics()),
        ("styles", lambda: cfg.styles()),
    ]
    for name, fn in checks:
        try:
            result = fn()
            click.echo(
                f"  ✓ {name}: {len(result) if hasattr(result, '__len__') else 'ok'}"
            )
        except Exception as exc:
            click.echo(f"  ✗ {name}: {exc}", err=True)
            errors.append(name)

    if errors:
        click.echo(f"\n{len(errors)} config error(s) found", err=True)
        sys.exit(1)
    click.echo("\nAll configs valid.")


@cli.command("youtube-auth")
def youtube_auth():
    """Run one-time OAuth2 flow to obtain a YouTube refresh token."""
    from wisdom.platforms.youtube import run_oauth_flow

    token = run_oauth_flow()
    click.echo(f"\nYOUTUBE_REFRESH_TOKEN={token}")
    click.echo("Add this to your .env / GitHub Secrets.")


@cli.command()
def themes():
    """List all enabled themes and their schedule."""
    import wisdom.config as cfg

    for key, t in cfg.enabled_themes().items():
        click.echo(f"  {key:15s}  {t.format:5s}  platforms={t.platforms}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_optional(path: str) -> bytes | None:
    p = Path(path)
    return p.read_bytes() if p.exists() else None


def _meta_from_record(record: dict, theme_cfg) -> object:
    from wisdom.schemas import PostMeta

    return PostMeta(
        caption=record.get("caption", ""),
        title="",
        hashtags=theme_cfg.hashtags,
        tags=[],
        theme=theme_cfg.key,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
