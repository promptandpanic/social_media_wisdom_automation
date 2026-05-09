"""GitHub Releases as a public CDN for media assets."""
from __future__ import annotations

import logging
import os
import time
import uuid

import requests

logger = logging.getLogger(__name__)

_RELEASE_TAG = "media-pool"
_API = "https://api.github.com"


class GitHubUploader:
    def __init__(self):
        self._token = os.environ.get("GITPROVIDER_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
        self._repo = os.environ.get("GITPROVIDER_REPO") or os.environ.get("GITHUB_REPO") or os.environ.get("GITHUB_REPOSITORY", "")
        self._uploaded: list[str] = []

    def _headers(self) -> dict:
        return {"Authorization": f"token {self._token}",
                "Accept": "application/vnd.github.v3+json"}

    def _release_id(self) -> int | None:
        r = requests.get(
            f"{_API}/repos/{self._repo}/releases/tags/{_RELEASE_TAG}",
            headers=self._headers(), timeout=15,
        )
        if r.status_code == 200:
            return r.json()["id"]
        return None

    def upload(self, data: bytes, filename: str | None = None) -> str | None:
        if not self._token or not self._repo:
            logger.warning("GitHub uploader: GITPROVIDER_TOKEN or GITPROVIDER_REPO not set")
            return None

        release_id = self._release_id()
        if not release_id:
            logger.error("GitHub: media-pool release not found")
            return None

        fname = filename or f"{uuid.uuid4().hex[:8]}.mp4"
        ext = fname.rsplit(".", 1)[-1]
        mime = {"mp4": "video/mp4", "jpg": "image/jpeg",
                "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "application/octet-stream")

        upload_url = (
            f"https://uploads.github.com/repos/{self._repo}"
            f"/releases/{release_id}/assets?name={fname}"
        )
        r = requests.post(
            upload_url,
            headers={**self._headers(), "Content-Type": mime},
            data=data, timeout=120,
        )
        if r.status_code in (200, 201):
            url = r.json()["browser_download_url"]
            self._uploaded.append(r.json()["id"])
            logger.info(f"Uploaded {fname} → {url}")
            return url

        logger.error(f"Upload failed {r.status_code}: {r.text[:200]}")
        return None

    def cleanup(self) -> None:
        """Delete uploaded assets from GitHub Releases after posting."""
        for asset_id in self._uploaded:
            try:
                requests.delete(
                    f"{_API}/repos/{self._repo}/releases/assets/{asset_id}",
                    headers=self._headers(), timeout=15,
                )
            except Exception as exc:
                logger.debug(f"Cleanup failed for asset {asset_id}: {exc}")
        self._uploaded.clear()
