"""
Content database — SQLite.
Tracks posted quotes (for dedup) and pending posts.

In CI (GITHUB_ACTIONS=true): wisdom.db is downloaded from the 'db' GitHub Release
at load() and uploaded back at save(). The file is never committed to git.

Locally: data/wisdom.db is gitignored. Each local run builds its own history.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

import wisdom.config as cfg

logger = logging.getLogger(__name__)

_API = "https://api.github.com"
_DB_RELEASE_TAG = "db"
_DB_ASSET_NAME = "wisdom.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_ci() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true"


def _gh_headers() -> dict:
    token = os.environ.get("GITPROVIDER_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}


def _repo() -> str:
    return os.environ.get("GITPROVIDER_REPO") or os.environ.get("GITHUB_REPOSITORY", "")


def _release_id() -> int | None:
    repo = _repo()
    if not repo:
        return None
    r = requests.get(
        f"{_API}/repos/{repo}/releases/tags/{_DB_RELEASE_TAG}",
        headers=_gh_headers(), timeout=15,
    )
    return r.json()["id"] if r.status_code == 200 else None


def _download_db(path: Path) -> None:
    if not _is_ci() or not _repo() or not (os.environ.get("GITPROVIDER_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        return

    release_id = _release_id()
    if not release_id:
        logger.info("DB release not found — starting fresh")
        return

    r = requests.get(
        f"{_API}/repos/{_repo()}/releases/{release_id}/assets",
        headers=_gh_headers(), timeout=15,
    )
    if r.status_code != 200:
        return

    for asset in r.json():
        if asset["name"] == _DB_ASSET_NAME:
            dl = requests.get(
                asset["url"],
                headers={**_gh_headers(), "Accept": "application/octet-stream"},
                timeout=30,
            )
            if dl.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(dl.content)
                logger.info(f"Downloaded DB ({len(dl.content):,} bytes)")
            return

    logger.info("No DB asset in release — starting fresh")


def _upload_db(path: Path) -> None:
    if not _is_ci() or not _repo() or not os.environ.get("GITHUB_TOKEN") or not path.exists():
        return

    release_id = _release_id()
    if not release_id:
        logger.warning("DB release tag not found — cannot upload DB")
        return

    # Delete existing asset before uploading the new one
    r = requests.get(
        f"{_API}/repos/{_repo()}/releases/{release_id}/assets",
        headers=_gh_headers(), timeout=15,
    )
    if r.status_code == 200:
        for asset in r.json():
            if asset["name"] == _DB_ASSET_NAME:
                requests.delete(
                    f"{_API}/repos/{_repo()}/releases/assets/{asset['id']}",
                    headers=_gh_headers(), timeout=15,
                )

    data = path.read_bytes()
    r = requests.post(
        f"https://uploads.github.com/repos/{_repo()}/releases/{release_id}/assets"
        f"?name={_DB_ASSET_NAME}",
        headers={**_gh_headers(), "Content-Type": "application/octet-stream"},
        data=data, timeout=60,
    )
    if r.status_code in (200, 201):
        logger.info(f"Uploaded DB ({len(data):,} bytes)")
    else:
        logger.error(f"DB upload failed {r.status_code}: {r.text[:200]}")


def _row_to_pending(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["quote"] = json.loads(d["quote"])
    d["platforms"] = json.loads(d["platforms"])
    return d


def _all_posted(record: dict) -> bool:
    platforms = record.get("platforms", {})
    return bool(platforms) and all(p.get("status") == "posted" for p in platforms.values())


# ---------------------------------------------------------------------------
# ContentDB
# ---------------------------------------------------------------------------

class ContentDB:
    def __init__(self, path: str | None = None):
        self._path = Path(path or cfg.app()["db_path"])
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS posted (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                quote_text TEXT NOT NULL,
                author     TEXT,
                theme      TEXT,
                style      TEXT,
                posted_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pending (
                id         TEXT PRIMARY KEY,
                theme      TEXT NOT NULL,
                quote      TEXT NOT NULL,
                caption    TEXT,
                asset_dir  TEXT,
                platforms  TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

    def load(self) -> None:
        _download_db(self._path)
        self._connect()
        posted = self._conn.execute("SELECT COUNT(*) FROM posted").fetchone()[0]
        pending = self._conn.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        logger.info(f"DB: {posted} posted, {pending} pending")

    def save(self) -> None:
        if self._conn:
            self._conn.commit()
        _upload_db(self._path)

    # ── Posted quote dedup ────────────────────────────────────────────────────

    def recent_quotes(self, days: int | None = None) -> list[str]:
        """Return quote texts posted OR pending in the last N days (for LLM context)."""
        days = days if days is not None else cfg.app().get("recent_posts_window_days", 30)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        # 1. From posted table
        rows_posted = self._conn.execute(
            "SELECT quote_text FROM posted WHERE posted_at > ? ORDER BY posted_at DESC",
            (cutoff,),
        ).fetchall()
        posted = [r["quote_text"] for r in rows_posted]
        
        # 2. From pending table (LLM should avoid what we've already generated but not yet 'promoted')
        rows_pending = self._conn.execute(
            "SELECT quote FROM pending WHERE created_at > ?", (cutoff,)
        ).fetchall()
        pending = []
        for r in rows_pending:
            try:
                q_data = json.loads(r["quote"])
                if isinstance(q_data, dict) and q_data.get("text"):
                    pending.append(q_data["text"])
            except Exception:
                continue
                
        return list(set(posted + pending))

    def recent_styles(self, max_entries: int = 10) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT style FROM posted WHERE style != '' "
            "ORDER BY posted_at DESC LIMIT ?",
            (max_entries,),
        ).fetchall()
        return [r["style"] for r in rows]

    def mark_posted(self, quote: dict, theme: str, style: str = "") -> None:
        self._conn.execute(
            "INSERT INTO posted (quote_text, author, theme, style, posted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (quote["text"], quote.get("author", ""), theme, style, _now()),
        )
        self._conn.commit()

    # ── Pending posts (generate → post decoupling) ────────────────────────────

    def create_pending(self, theme: str, quote: dict, caption: str, asset_dir: str) -> str:
        record_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO pending (id, theme, quote, caption, asset_dir, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record_id, theme, json.dumps(quote), caption, asset_dir, _now()),
        )
        self._conn.commit()
        return record_id

    def get_pending(self, theme: str | None = None) -> list[dict]:
        if theme:
            rows = self._conn.execute(
                "SELECT * FROM pending WHERE theme = ? ORDER BY created_at", (theme,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM pending ORDER BY created_at"
            ).fetchall()
        return [_row_to_pending(r) for r in rows if not _all_posted(_row_to_pending(r))]

    def get_pending_by_id(self, record_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM pending WHERE id = ?", (record_id,)
        ).fetchone()
        return _row_to_pending(row) if row else None

    def update_platform_status(self, record_id: str, platform: str,
                               status: str, post_id: str = "", error: str = "") -> None:
        row = self._conn.execute(
            "SELECT platforms FROM pending WHERE id = ?", (record_id,)
        ).fetchone()
        if not row:
            return
        platforms = json.loads(row["platforms"])
        platforms[platform] = {
            "status": status,
            "post_id": post_id,
            "error": error,
            "posted_at": _now() if status == "posted" else "",
        }
        self._conn.execute(
            "UPDATE pending SET platforms = ? WHERE id = ?",
            (json.dumps(platforms), record_id),
        )
        self._conn.commit()

    def promote_to_posted(self, record_id: str) -> None:
        row = self._conn.execute(
            "SELECT * FROM pending WHERE id = ?", (record_id,)
        ).fetchone()
        if not row:
            return
        record = _row_to_pending(row)
        self.mark_posted(record["quote"], record["theme"], style=record.get("style", ""))
        self._conn.execute("DELETE FROM pending WHERE id = ?", (record_id,))
        self._conn.commit()
