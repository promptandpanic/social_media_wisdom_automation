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
    " No text, signs, words, letters, numbers, logos, or watermarks anywhere in the image."
    " 9:16 portrait format."
)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseImageProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str) -> bytes:
        ...

    def available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------

class HuggingFaceProvider(BaseImageProvider):
    name = "huggingface"

    def __init__(self, model: str = "black-forest-labs/FLUX.1-schnell", timeout: int = 60, **_):
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get("HF_API_KEY"))

    def generate(self, prompt: str) -> bytes:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {os.environ['HF_API_KEY']}"},
            json={"inputs": prompt + _SAFETY_SUFFIX,
                  "parameters": {"width": _W, "height": _H}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _resize(resp.content)


class LeonardoProvider(BaseImageProvider):
    name = "leonardo"

    def __init__(self, timeout: int = 90, **_):
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get("LEONARDO_API_KEY"))

    def generate(self, prompt: str) -> bytes:
        import time
        key = os.environ["LEONARDO_API_KEY"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        r = requests.post(
            "https://cloud.leonardo.ai/api/rest/v1/generations",
            headers=headers,
            json={"prompt": prompt + _SAFETY_SUFFIX, "width": _W, "height": _H,
                  "num_images": 1, "modelId": os.environ.get("LEONARDO_MODEL_ID", "")},
            timeout=30,
        )
        r.raise_for_status()
        gen_id = r.json()["sdGenerationJob"]["generationId"]

        for _ in range(20):
            time.sleep(4)
            poll = requests.get(
                f"https://cloud.leonardo.ai/api/rest/v1/generations/{gen_id}",
                headers=headers, timeout=15,
            )
            poll.raise_for_status()
            imgs = poll.json().get("generations_by_pk", {}).get("generated_images", [])
            if imgs:
                return _resize(requests.get(imgs[0]["url"], timeout=30).content)

        raise TimeoutError("Leonardo generation timed out")


class LeonardoFluxProProvider(BaseImageProvider):
    """Leonardo FLUX.2 Pro via v2 API — generates 810x1440 (9:16), resized to 1080x1920."""
    name = "leonardo_flux_pro"

    def __init__(self, timeout: int = 120, **_):
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get("LEONARDO_API_KEY"))

    def generate(self, prompt: str) -> bytes:
        import time
        key = os.environ["LEONARDO_API_KEY"]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        r = requests.post(
            "https://cloud.leonardo.ai/api/rest/v2/generations",
            headers=headers,
            json={"model": "flux-pro-2.0", "prompt": prompt + _SAFETY_SUFFIX,
                  "width": 810, "height": 1440, "quantity": 1},
            timeout=30,
        )
        r.raise_for_status()
        gen_id = r.json()["sdGenerationJob"]["generationId"]

        for _ in range(30):
            time.sleep(4)
            poll = requests.get(
                f"https://cloud.leonardo.ai/api/rest/v1/generations/{gen_id}",
                headers=headers, timeout=15,
            )
            poll.raise_for_status()
            imgs = poll.json().get("generations_by_pk", {}).get("generated_images", [])
            if imgs:
                return _resize(requests.get(imgs[0]["url"], timeout=30).content)

        raise TimeoutError("Leonardo FLUX.2 Pro generation timed out")


class GeminiImagenProvider(BaseImageProvider):
    """Imagen 3 via Google Gen AI SDK."""
    name = "gemini_imagen"

    def __init__(self, model: str = "imagen-3.0-generate-002", **_):
        self.model = model

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> bytes:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_images(
            model=self.model,
            prompt=prompt + _SAFETY_SUFFIX,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="9:16",
                safety_filter_level="BLOCK_ONLY_HIGH",
            ),
        )
        return _resize(resp.generated_images[0].image.image_bytes)


class GeminiFlashProvider(BaseImageProvider):
    """Gemini native image generation (response modalities)."""
    name = "gemini_flash"

    def __init__(self, model: str = "gemini-2.5-flash-image", **_):
        self.model = model

    def available(self) -> bool:
        return bool(os.environ.get("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> bytes:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt + _SAFETY_SUFFIX,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
            ),
        )
        for part in resp.candidates[0].content.parts:
            if part.inline_data:
                return _resize(part.inline_data.data)
        raise ValueError("No image in Gemini Flash response")


class PollinationsProvider(BaseImageProvider):
    name = "pollinations"

    def __init__(self, **_):
        pass

    def generate(self, prompt: str) -> bytes:
        encoded = url_encode(prompt[:500])
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={_W}&height={_H}&nologo=true"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return _resize(resp.content)


class GradientFallback(BaseImageProvider):
    name = "gradient"

    def __init__(self, **_):
        pass

    def generate(self, prompt: str) -> bytes:
        img = Image.new("RGB", (_W, _H))
        d = ImageDraw.Draw(img)
        top = (15, 15, 20)  # Charcoal
        bot = (40, 40, 45)  # Dark Gray
        for y in range(_H):
            t = y / _H
            r = int(top[0] + (bot[0] - top[0]) * t)
            g = int(top[1] + (bot[1] - top[1]) * t)
            b = int(top[2] + (bot[2] - top[2]) * t)
            d.line([(0, y), (_W, y)], fill=(r, g, b))
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


def generate(prompt: str) -> bytes:
    """Run prompt through the image fallback chain. Always returns bytes."""
    chain = cfg.image_fallback_chain()
    last_error: Exception | None = None

    for name in chain:
        provider = _get(name)
        if provider is None:
            logger.debug(f"Image provider {name!r} not found — skipping")
            continue
        if not provider.available():
            logger.debug(f"Image provider {name!r}: key missing — skipping")
            continue
        try:
            logger.info(f"Image: trying {name}")
            result = provider.generate(prompt)
            logger.info(f"Image: ✓ {name} ({len(result)//1024} KB)")
            return result
        except Exception as exc:
            logger.warning(f"Image: {name} failed: {exc}")
            last_error = exc

    logger.error("Image: all providers failed — returning gradient fallback")
    return GradientFallback().generate(prompt)
