"""Instagram Graph API platform."""
from __future__ import annotations

import logging
import os
import time

import requests

from wisdom.platforms.base import BasePlatform
from wisdom.schemas import PlatformResult, PostMeta
from wisdom.storage.uploader import GitHubUploader

logger = logging.getLogger(__name__)

_GRAPH = "https://graph.facebook.com/v19.0"


class InstagramPlatform(BasePlatform):
    name = "instagram"

    def available(self) -> bool:
        return bool(
            os.environ.get("INSTAGRAM_ACCESS_TOKEN")
            and os.environ.get("INSTAGRAM_BUSINESS_ID")
        )

    def _token(self) -> str:
        return os.environ["INSTAGRAM_ACCESS_TOKEN"]

    def _biz_id(self) -> str:
        return os.environ["INSTAGRAM_BUSINESS_ID"]

    def _caption(self, meta: PostMeta) -> str:
        tags = " ".join(meta.hashtags)
        return f"{meta.caption}\n\n{tags}" if tags else meta.caption

    def post_video(self, video: bytes, thumbnail: bytes, meta: PostMeta) -> PlatformResult:
        uploader = GitHubUploader()
        try:
            video_url = uploader.upload(video, filename="reel.mp4")
            thumb_url = uploader.upload(thumbnail, filename="thumb.jpg")
            if not video_url:
                return PlatformResult("instagram", "failed", error="Upload failed")

            caption = self._caption(meta)
            biz = self._biz_id()
            token = self._token()

            # Step 1: create container
            r = requests.post(
                f"{_GRAPH}/{biz}/media",
                data={"media_type": "REELS", "video_url": video_url,
                      "caption": caption, "thumb_offset": 1000,
                      "cover_url": thumb_url, "access_token": token},
                timeout=30,
            )
            r.raise_for_status()
            container_id = r.json()["id"]

            # Step 2: wait for processing
            for _ in range(15):
                time.sleep(4)
                s = requests.get(
                    f"{_GRAPH}/{container_id}",
                    params={"fields": "status_code", "access_token": token},
                    timeout=15,
                ).json()
                status = s.get("status_code", "")
                if status == "FINISHED":
                    break
                if status == "ERROR":
                    return PlatformResult("instagram", "failed",
                                         error=f"Container error: {s}")

            # Step 3: publish
            pub = requests.post(
                f"{_GRAPH}/{biz}/media_publish",
                data={"creation_id": container_id, "access_token": token},
                timeout=30,
            )
            pub.raise_for_status()
            post_id = pub.json()["id"]
            return PlatformResult("instagram", "posted", post_id=post_id)

        except Exception as exc:
            return PlatformResult("instagram", "failed", error=str(exc))
        finally:
            uploader.cleanup()

    def post_image(self, image: bytes, meta: PostMeta) -> PlatformResult:
        uploader = GitHubUploader()
        try:
            image_url = uploader.upload(image, filename="cover.jpg")
            if not image_url:
                return PlatformResult("instagram", "failed", error="Upload failed")

            caption = self._caption(meta)
            biz = self._biz_id()
            token = self._token()

            r = requests.post(
                f"{_GRAPH}/{biz}/media",
                data={"image_url": image_url, "caption": caption,
                      "access_token": token},
                timeout=30,
            )
            r.raise_for_status()
            container_id = r.json()["id"]

            time.sleep(3)
            pub = requests.post(
                f"{_GRAPH}/{biz}/media_publish",
                data={"creation_id": container_id, "access_token": token},
                timeout=30,
            )
            pub.raise_for_status()
            post_id = pub.json()["id"]
            return PlatformResult("instagram", "posted", post_id=post_id)

        except Exception as exc:
            return PlatformResult("instagram", "failed", error=str(exc))
        finally:
            uploader.cleanup()
