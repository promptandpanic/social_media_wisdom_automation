"""
Central type definitions for the entire pipeline.
All LangGraph state and config models live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


# ---------------------------------------------------------------------------
# Config models (loaded from YAML)
# ---------------------------------------------------------------------------

@dataclass
class YouTubeConfig:
    title_template: str
    tags: list[str]
    category_id: str = "22"
    privacy: str = "public"


@dataclass
class ThemeConfig:
    key: str
    name: str
    format: Literal["reel", "image"]
    max_words: int
    platforms: list[str]
    hashtags: list[str]
    enabled: bool = True
    styles: list[str] | None = None      # None = all applicable styles
    youtube: YouTubeConfig | None = None


@dataclass
class LLMRoleConfig:
    providers: list[str]
    temperature: float = 0.85
    max_tokens: int = 1024
    disable_thinking: bool = False


@dataclass
class ProviderConfig:
    model: str | None = None             # LiteLLM model string for LLM providers
    cls: str | None = None               # dotted import path for custom image providers
    key_env: str | None = None
    timeout: int = 60
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline data models
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    text: str
    author: str
    highlight: str
    source: Literal["real_author", "internet_found", "fallback"]
    image_hint: str = ""
    score: int = 0


@dataclass
class Overlay:
    type: Literal["gradient_bottom", "gradient_top", "gradient_center", "solid", "vignette", "none"]
    opacity: int = 150
    color: str = "#000000"


@dataclass
class DesignBrief:
    image_prompt: str
    style: str
    font: str
    text_color: str
    highlight_color: str
    author_color: str
    overlay: Overlay
    text_zone: Literal["top", "center", "bottom"]
    layout: Literal["big_center", "sentence_reveal", "full_card"]
    decoration: Literal["rule", "quote_mark", "none"]
    highlight: str
    highlight_style: Literal["color", "italic", "underline", "caps", "caps_italic", "script"]
    font_size: int
    animation: Literal["fade", "reveal", "none"]
    skip_kenburns: bool = False
    mood_note: str = ""
    voice_gender: Literal["male", "female"] = "male"


@dataclass
class PostMeta:
    caption: str
    title: str                            # YouTube title / ignored by Instagram
    hashtags: list[str]
    tags: list[str]                       # YouTube tags / ignored by Instagram
    theme: str


@dataclass
class PlatformResult:
    platform: str
    status: Literal["posted", "failed", "skipped"]
    post_id: str = ""
    url: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# LangGraph pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    # Input
    theme_key: str
    theme: ThemeConfig
    dry_run: bool
    generate_only: bool
    offline: bool

    # Quote generation
    quote: Quote | None
    recent_quotes: list[str]
    recent_styles: list[str]

    # Quote agent internal state
    _quote_mode: str
    _quote_attempt: int
    _valid_quote: Quote | None
    _quote_fallback: dict[str, Any] | None
    _image_hint: str

    # Design
    brief: DesignBrief | None
    _chosen_style: str

    # Media
    image_bytes: bytes | None
    composed_image: bytes | None
    video_bytes: bytes | None
    thumbnail_bytes: bytes | None
    design_attempt: int
    best_score: int
    best_state: dict[str, Any] | None
    _accepted: bool
    _hard_gate: bool

    # Post metadata
    meta: PostMeta | None
    llm_caption: str

    # Storage
    pending_id: str | None

    # Results
    platform_results: list[PlatformResult]
    errors: list[str]
