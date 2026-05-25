"""Facebook Graph API platform."""

from __future__ import annotations

import logging
import os
import requests

from wisdom.platforms.base import BasePlatform
from wisdom.schemas import PlatformResult, PostMeta
from wisdom.storage.uploader import GitHubUploader

logger = logging.getLogger(__name__)

_GRAPH = "https://graph.facebook.com/v19.0"

class FacebookPlatform(BasePlatform):
    name = "facebook"

    def available(self) -> bool:
        return bool(
            os.environ.get("FACEBOOK_ACCESS_TOKEN")
            and os.environ.get("FACEBOOK_PAGE_ID")
        )

    def _token(self) -> str:
        return os.environ["FACEBOOK_ACCESS_TOKEN"]

    def _page_id(self) -> str:
        return os.environ["FACEBOOK_PAGE_ID"]

    def _caption(self, meta: PostMeta) -> str:
        tags = " ".join(meta.hashtags)
        return f"{meta.caption}\n\n{tags}" if tags else meta.caption

    def _permalink(self, post_id: str, token: str) -> str:
        try:
            r = requests.get(
                f"{_GRAPH}/{post_id}",
                params={"fields": "permalink_url", "access_token": token},
                timeout=15,
            )
            return r.json().get("permalink_url", "")
        except Exception:
            return ""

    def post_video(self, video: bytes, thumbnail: bytes, meta: PostMeta) -> PlatformResult:
        uploader = GitHubUploader()
        try:
            video_url = uploader.upload(video, filename="fb_reel.mp4")
            if not video_url:
                return PlatformResult("facebook", "failed", error="Upload failed")

            caption = self._caption(meta)
            page = self._page_id()
            token = self._token()

            # For Facebook Pages, the /videos endpoint handles URL uploads seamlessly.
            # Vertical videos under 90s are distributed as Reels.
            r = requests.post(
                f"{_GRAPH}/{page}/videos",
                data={
                    "file_url": video_url,
                    "description": caption,
                    "access_token": token,
                },
                timeout=60,
            )
            if r.status_code != 200:
                logger.error(f"Facebook video publish failed: {r.text}")
                return PlatformResult("facebook", "failed", error=f"FB Error: {r.text}")

            post_id = r.json()["id"]
            # Video processing might mean the permalink isn't immediately resolvable,
            # but we can return the ID. Let's try to get the permalink.
            url = f"https://www.facebook.com/{page}/videos/{post_id}"
            
            return PlatformResult("facebook", "posted", post_id=post_id, url=url)

        except Exception as exc:
            logger.error(f"Facebook post_video failed: {exc}")
            return PlatformResult("facebook", "failed", error=str(exc))
        finally:
            uploader.cleanup()

    def post_image(self, image: bytes, meta: PostMeta) -> PlatformResult:
        uploader = GitHubUploader()
        try:
            image_url = uploader.upload(image, filename="fb_cover.jpg")
            if not image_url:
                return PlatformResult("facebook", "failed", error="Upload failed")

            caption = self._caption(meta)
            page = self._page_id()
            token = self._token()

            r = requests.post(
                f"{_GRAPH}/{page}/photos",
                data={
                    "url": image_url,
                    "message": caption,
                    "access_token": token,
                },
                timeout=30,
            )
            if r.status_code != 200:
                logger.error(f"Facebook image publish failed: {r.text}")
                return PlatformResult("facebook", "failed", error=f"FB Error: {r.text}")

            post_id = r.json()["id"]
            url = self._permalink(post_id, token)
            return PlatformResult("facebook", "posted", post_id=post_id, url=url)

        except Exception as exc:
            logger.error(f"Facebook post_image failed: {exc}")
            return PlatformResult("facebook", "failed", error=str(exc))
        finally:
            uploader.cleanup()
