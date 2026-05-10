"""
Loads and validates all YAML config files.
Returns typed dataclass instances — no raw dicts escape this module.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from wisdom.schemas import (
    LLMRoleConfig, ProviderConfig, ThemeConfig, YouTubeConfig,
)

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load(filename: str) -> dict:
    path = _CONFIG_DIR / filename
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# App config
# ---------------------------------------------------------------------------

@lru_cache
def app() -> dict:
    return _load("app.yml")["app"]


@lru_cache
def image_cfg() -> dict:
    return _load("app.yml")["image"]


@lru_cache
def reel_cfg() -> dict:
    return _load("app.yml")["reel"]


# ---------------------------------------------------------------------------
# Theme config
# ---------------------------------------------------------------------------

@lru_cache
def themes() -> dict[str, ThemeConfig]:
    raw = _load("themes.yml")["themes"]
    out: dict[str, ThemeConfig] = {}
    for key, data in raw.items():
        yt_raw = data.get("youtube")
        yt = YouTubeConfig(**yt_raw) if yt_raw else None
        out[key] = ThemeConfig(
            key=key,
            name=data["name"],
            format=data.get("format", "reel"),
            max_words=data.get("max_words", 24),
            platforms=data.get("platforms", ["instagram"]),
            hashtags=data.get("hashtags", []),
            enabled=data.get("enabled", True),
            styles=data.get("styles"),
            youtube=yt,
        )
    return out


def theme(key: str) -> ThemeConfig:
    t = themes().get(key)
    if not t:
        raise ValueError(f"Unknown theme: {key!r}. Available: {list(themes())}")
    return t


def enabled_themes() -> dict[str, ThemeConfig]:
    return {k: v for k, v in themes().items() if v.enabled}


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

@lru_cache
def _llm_raw() -> dict:
    return _load("llm.yml")


def llm_providers() -> dict[str, ProviderConfig]:
    raw = _llm_raw()["providers"]
    return {
        name: ProviderConfig(
            model=data.get("model"),
            key_env=data.get("key_env"),
            timeout=data.get("timeout", 60),
        )
        for name, data in raw.items()
    }


def llm_role(role: str) -> LLMRoleConfig:
    raw = _llm_raw()["roles"].get(role)
    if not raw:
        raise ValueError(f"Unknown LLM role: {role!r}")
        
    providers = raw["providers"]
    env_override = os.environ.get("LLM_PROVIDER_ORDER")
    if env_override:
        providers = [p.strip() for p in env_override.split(",")]
        
    return LLMRoleConfig(
        providers=providers,
        temperature=raw.get("temperature", 0.85),
        max_tokens=raw.get("max_tokens", 1024),
        disable_thinking=raw.get("disable_thinking", False),
    )


# ---------------------------------------------------------------------------
# Image config
# ---------------------------------------------------------------------------

@lru_cache
def _image_raw() -> dict:
    return _load("image.yml")


def image_providers() -> dict[str, ProviderConfig]:
    raw = _image_raw()["providers"]
    return {
        name: ProviderConfig(
            cls=data["class"],
            key_env=data.get("key_env"),
            model=data.get("model"),
            timeout=data.get("timeout", 60),
        )
        for name, data in raw.items()
    }


def image_fallback_chain() -> list[str]:
    env_override = os.environ.get("IMAGE_PROVIDER_ORDER")
    if env_override:
        return [p.strip() for p in env_override.split(",")]
    return _image_raw()["fallback_chain"]


# ---------------------------------------------------------------------------
# Topics / Styles (passed through as raw dicts — agents consume them directly)
# ---------------------------------------------------------------------------

@lru_cache
def topics() -> dict:
    return _load("topics.yml")["categories"]


@lru_cache
def styles() -> dict:
    return _load("styles.yml")["styles"]


@lru_cache
def curated_quotes() -> dict[str, list[dict]]:
    data = _load("curated_quotes.yml") or {}
    return data.get("quotes", {})
