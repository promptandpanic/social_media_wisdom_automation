"""
Image generation provider registry.

Fallback chain is declared in config/image.yml.
Adding a new provider = implement BaseImageProvider, add entry to image.yml.
"""

from __future__ import annotations

import importlib
import io
import logging
import os
from abc import ABC, abstractmethod
from urllib.parse import quote as url_encode

import requests
from PIL import Image, ImageDraw

import wisdom.config as cfg

logger = logging.getLogger(__name__)

_W = 1080
_H = 1920

_SAFETY_SUFFIX = (
    " ABSOLUTELY NO text, signs, words, letters, numbers, logos, brand marks, or watermarks anywhere in the scene. "
    "Focus on high-aesthetic lifestyle photography. "
    "9:16 portrait format."
)

_NEGATIVE_PROMPT = (
    "text, letters, words, watermark, logo, brand mark, clothing brand, jersey logo, "
    "signature, handwritten name, artist signature, corner signature, monogram, initials, copyright, © symbol, "
    "studio mark, caption, label, typography, graffiti, banner, hashtag, # symbol, "
    "code, url, numbers, equation, symbol, glyph, written character, "
    "distorted face, extra fingers, anatomical anomalies, low quality, blurry"
)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseImageProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, native_text: bool = False) -> bytes: ...

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class HuggingFaceProvider(BaseImageProvider):
    name = "huggingface"

    def __init__(
        self, model: str = "black-forest-labs/FLUX.1-schnell", timeout: int = 60, **_
    ):
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get("HF_API_KEY"))

    def generate(self, prompt: str, native_text: bool = False) -> bytes:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {os.environ['HF_API_KEY']}"},
            json={
                "inputs": prompt if native_text else prompt + _SAFETY_SUFFIX,
                "parameters": {"width": _W, "height": _H},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _resize(resp.content)


class LeonardoFluxProProvider(BaseImageProvider):
    """Leonardo FLUX.2 Pro via v2 API — generates 810x1440 (9:16), resized to 1080x1920."""

    name = "leonardo_flux_pro"

    def __init__(self, timeout: int = 120, **_):
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get("LEONARDO_API_KEY"))

    def generate(self, prompt: str, native_text: bool = False) -> bytes:
        import time

        key = os.environ["LEONARDO_API_KEY"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        r = requests.post(
            "https://cloud.leonardo.ai/api/rest/v2/generations",
            headers=headers,
            json={
                "model": "flux-pro-2.0",
                "public": False,
                "parameters": {
                    "prompt": prompt if native_text else prompt + _SAFETY_SUFFIX,
                    "negativePrompt": "" if native_text else _NEGATIVE_PROMPT,
                    "width": 810,
                    "height": 1440,
                    "quantity": 1,
                },
            },
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        if isinstance(body, list):
            errors = ", ".join(err.get("message", "Unknown error") for err in body)
            raise ValueError(f"Leonardo API error: {errors}")
        if "errors" in body:
            errors = ", ".join(
                err.get("message", "Unknown error") for err in body["errors"]
            )
            raise ValueError(f"Leonardo API error: {errors}")

        gen = body.get("generate")
        if not gen or "generationId" not in gen:
            raise ValueError(f"Leonardo API returned invalid response: {body}")

        gen_id = gen["generationId"]

        for _ in range(30):
            time.sleep(4)
            poll = requests.get(
                f"https://cloud.leonardo.ai/api/rest/v1/generations/{gen_id}",
                headers=headers,
                timeout=15,
            )
            poll.raise_for_status()
            poll_body = poll.json()
            if isinstance(poll_body, list):
                errors = ", ".join(
                    err.get("message", "Unknown error") for err in poll_body
                )
                raise ValueError(f"Leonardo polling error: {errors}")
            if "errors" in poll_body:
                errors = ", ".join(
                    err.get("message", "Unknown error") for err in poll_body["errors"]
                )
                raise ValueError(f"Leonardo polling error: {errors}")

            imgs = poll_body.get("generations_by_pk", {}).get("generated_images", [])
            if imgs:
                return _resize(requests.get(imgs[0]["url"], timeout=30).content)

        raise TimeoutError("Leonardo FLUX.2 Pro generation timed out")


class GeminiImagenProvider(BaseImageProvider):
    """Imagen 3 via Google Gen AI SDK."""

    name = "gemini_imagen"

    def __init__(self, model: str = "imagen-4.0-generate-001", **_):
        self.model = model

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def generate(self, prompt: str, native_text: bool = False) -> bytes:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_images(
            model=self.model,
            prompt=prompt if native_text else prompt + _SAFETY_SUFFIX,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                safety_filter_level="BLOCK_LOW_AND_ABOVE",
            ),
        )
        return _resize(resp.generated_images[0].image.image_bytes)


class GeminiFlashProvider(BaseImageProvider):
    """Gemini native image generation (response modalities)."""

    name = "gemini_flash"

    def __init__(self, model: str = "gemini-3.5-flash-image", **_):
        self.model = model

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def generate(self, prompt: str, native_text: bool = False) -> bytes:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt if native_text else prompt + _SAFETY_SUFFIX,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        if not resp.candidates or not resp.candidates[0].content or not resp.candidates[0].content.parts:
            raise ValueError(f"Invalid or blocked response from Gemini Flash: {resp}")
        for part in resp.candidates[0].content.parts:
            if part.inline_data:
                return _resize(part.inline_data.data)
        raise ValueError("No image in Gemini Flash response")




class GradientFallback(BaseImageProvider):
    name = "gradient"

    def __init__(self, **_):
        pass

    def generate(self, prompt: str, native_text: bool = False) -> bytes:
        import os
        fallback_path = "assets/fallback_bg.jpg"
        if os.path.exists(fallback_path):
            with open(fallback_path, "rb") as f:
                return f.read()
                
        # Pure black fallback if file is missing
        img = Image.new("RGB", (_W, _H), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resize(data: bytes) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    scale = max(_W / img.width, _H / img.height)
    nw, nh = int(img.width * scale), int(img.height * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - _W) // 2, (nh - _H) // 2
    img = img.crop((left, top, left + _W, top + _H))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _import_class(dotted: str) -> type:
    module, cls = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


_PROVIDER_CACHE: dict[str, BaseImageProvider] = {}


def _get(name: str) -> BaseImageProvider | None:
    if name in _PROVIDER_CACHE:
        return _PROVIDER_CACHE[name]
    providers = cfg.image_providers()
    if name not in providers:
        return None
    pcfg = providers[name]
    kwargs: dict = {}
    if pcfg.model:
        kwargs["model"] = pcfg.model
    if pcfg.timeout:
        kwargs["timeout"] = pcfg.timeout
    cls = _import_class(pcfg.cls)
    p = cls(**kwargs)
    _PROVIDER_CACHE[name] = p
    return p


def generate(prompt: str, exclude: list[str] | None = None, native_text: bool = False) -> tuple[bytes, str]:
    """Run prompt through the image fallback chain. Returns (bytes, provider_name)."""
    chain = cfg.image_fallback_chain()
    exclude = exclude or []
    last_error: Exception | None = None

    for name in chain:
        if name in exclude:
            logger.debug(f"Image: skipping {name} (blacklisted for this run)")
            continue
        provider = _get(name)
        if provider is None:
            logger.debug(f"Image provider {name!r} not found — skipping")
            continue
        if not provider.available():
            logger.debug(f"Image provider {name!r}: key missing — skipping")
            continue
        try:
            logger.info(f"Image: trying {name}")
            result = provider.generate(prompt, native_text=native_text)
            logger.info(f"Image: ✓ {name} ({len(result) // 1024} KB)")
            return result, name
        except Exception as exc:
            logger.warning(f"Image: {name} failed: {exc}")
            last_error = exc

    logger.error("Image: all providers failed — returning gradient fallback")
    return GradientFallback().generate(prompt, native_text=native_text), "gradient"
