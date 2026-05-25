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
        if hasattr(self, "_effective_token"):
            return self._effective_token
            
        token = os.environ["FACEBOOK_ACCESS_TOKEN"]
        page_id = self._page_id()
        
        try:
            r = requests.get(
                f"{_GRAPH}/{page_id}",
                params={"fields": "access_token", "access_token": token},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                if "access_token" in data:
                    token = data["access_token"]
        except Exception as e:
            logger.warning(f"Could not check for page access token: {e}")
            
        self._effective_token = token
        return token

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

            # Step 1: Initialize Upload Session
            r1 = requests.post(
                f"{_GRAPH}/{page}/video_reels",
                data={
                    "upload_phase": "start",
                    "access_token": token,
                },
                timeout=30,
            )
            if r1.status_code != 200:
                logger.error(f"Facebook video init failed: {r1.text}")
                return PlatformResult("facebook", "failed", error=f"FB Init Error: {r1.text}")
            
            video_id = r1.json().get("video_id")
            if not video_id:
                return PlatformResult("facebook", "failed", error="No video_id returned from init")

            # Step 2: Upload Video via rupload
            r2 = requests.post(
                f"https://rupload.facebook.com/video-upload/{video_id}",
                headers={
                    "Authorization": f"OAuth {token}",
                    "file_url": video_url,
                },
                timeout=120,
            )
            if r2.status_code != 200:
                logger.error(f"Facebook rupload failed: {r2.text}")
                return PlatformResult("facebook", "failed", error=f"FB Upload Error: {r2.text}")

            # Step 3: Publish Reel
            r3 = requests.post(
                f"{_GRAPH}/{page}/video_reels",
                data={
                    "video_id": video_id,
                    "upload_phase": "finish",
                    "video_state": "PUBLISHED",
                    "description": caption,
                    "access_token": token,
                },
                timeout=30,
            )
            if r3.status_code != 200:
                logger.error(f"Facebook video finish failed: {r3.text}")
                return PlatformResult("facebook", "failed", error=f"FB Finish Error: {r3.text}")

            url = f"https://www.facebook.com/{page}/videos/{video_id}"
            return PlatformResult("facebook", "posted", post_id=video_id, url=url)

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
