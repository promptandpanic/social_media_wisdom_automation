"""YouTube Data API v3 platform — uploads Shorts (≤60s vertical video)."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from wisdom.platforms.base import BasePlatform
from wisdom.schemas import PlatformResult, PostMeta, ThemeConfig

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _build_title(meta: PostMeta, theme: ThemeConfig | None) -> str:
    """Build YouTube title from theme template or fallback to first line of caption."""
    if theme and theme.youtube and theme.youtube.title_template:
        tmpl = theme.youtube.title_template
        if "{quote_snippet}" not in tmpl:
            return tmpl
        max_len = 97 - len(tmpl.replace("{quote_snippet}", ""))
        raw = meta.caption.split("\n")[0]
        snippet = raw if len(raw) <= max_len else raw[:max_len].rsplit(" ", 1)[0] + "…"
        return tmpl.format(quote_snippet=snippet)
    return meta.title or meta.caption[:100]


class YouTubePlatform(BasePlatform):
    name = "youtube"

    def available(self) -> bool:
        return bool(
            os.environ.get("YOUTUBE_CLIENT_ID")
            and os.environ.get("YOUTUBE_CLIENT_SECRET")
            and os.environ.get("YOUTUBE_REFRESH_TOKEN")
        )

    def _credentials(self):
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request

        creds = Credentials(
            token=None,
            refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
            client_id=os.environ["YOUTUBE_CLIENT_ID"],
            client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=_SCOPES,
        )
        creds.refresh(Request())
        return creds

    def _service(self):
        from googleapiclient.discovery import build
        return build("youtube", "v3", credentials=self._credentials(), cache_discovery=False)

    def _upload(self, video: bytes, meta: PostMeta, theme: ThemeConfig | None,
                thumbnail: bytes | None = None) -> PlatformResult:
        from googleapiclient.http import MediaIoBaseUpload
        import io

        yt_cfg = theme.youtube if theme else None
        title = _build_title(meta, theme)
        hashtag_line = " ".join(meta.hashtags)
        description = f"{meta.caption}\n\n{hashtag_line}" if hashtag_line else meta.caption
        tags = (yt_cfg.tags if yt_cfg else []) + meta.tags
        category_id = yt_cfg.category_id if yt_cfg else "22"
        privacy = yt_cfg.privacy if yt_cfg else "public"

        service = self._service()
        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags[:500],
                "categoryId": category_id,
            },
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
        }

        media = MediaIoBaseUpload(
            io.BytesIO(video), mimetype="video/mp4", chunksize=-1, resumable=True
        )
        req = service.videos().insert(
            part=",".join(body.keys()), body=body, media_body=media
        )

        response = None
        while response is None:
            _, response = req.next_chunk()

        video_id = response["id"]

        # Set thumbnail if provided
        if thumbnail:
            try:
                import io as _io
                thumb_media = MediaIoBaseUpload(
                    _io.BytesIO(thumbnail), mimetype="image/jpeg", resumable=False
                )
                service.thumbnails().set(videoId=video_id, media_body=thumb_media).execute()
            except Exception as exc:
                logger.warning(f"YouTube thumbnail failed (non-fatal): {exc}")

        url = f"https://youtube.com/shorts/{video_id}"
        logger.info(f"YouTube: posted {url}")
        return PlatformResult("youtube", "posted", post_id=video_id, url=url)

    def post_video(self, video: bytes, thumbnail: bytes, meta: PostMeta,
                   theme: ThemeConfig | None = None) -> PlatformResult:
        try:
            return self._upload(video, meta, theme, thumbnail)
        except Exception as exc:
            logger.error(f"YouTube post_video failed: {exc}")
            return PlatformResult("youtube", "failed", error=str(exc))

    def post_image(self, image: bytes, meta: PostMeta,
                   theme: ThemeConfig | None = None) -> PlatformResult:
        # YouTube doesn't support image posts — skip gracefully
        return PlatformResult("youtube", "skipped",
                              error="YouTube does not support image-only posts")


# ---------------------------------------------------------------------------
# One-time OAuth2 setup helper (called by: task youtube-auth)
# ---------------------------------------------------------------------------

def run_oauth_flow() -> str:
    """Interactive OAuth2 flow — prints the refresh token to store as a secret."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_config = {
        "installed": {
            "client_id": os.environ["YOUTUBE_CLIENT_ID"],
            "client_secret": os.environ["YOUTUBE_CLIENT_SECRET"],
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, scopes=_SCOPES)
    creds = flow.run_local_server(port=0)
    token = creds.refresh_token
    print(f"\n✓ YOUTUBE_REFRESH_TOKEN={token}\n")
    print("Add this to your .env and GitHub Secrets.")
    return token
