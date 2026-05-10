"""
Media agent — image generation and composition.

Graph:
  generate_image → compose → END
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from wisdom import providers
from wisdom.schemas import PipelineState

logger = logging.getLogger(__name__)


def generate_image(state: PipelineState) -> PipelineState:
    brief = state.get("brief")
    theme_key = state["theme_key"]
    logger.info("Generating image…")

    if state.get("offline"):
        from wisdom.providers.image import GradientFallback
        logger.info("  Offline mode: using gradient fallback image")
        image_bytes = GradientFallback().generate("offline fallback")
    else:
        prompt = brief.image_prompt if brief else f"Beautiful inspirational {theme_key} image, 9:16."
        image_bytes = providers.image.generate(prompt)
    return {**state, "image_bytes": image_bytes}


def compose(state: PipelineState) -> PipelineState:
    from wisdom.composers.card import compose_image
    image_bytes = state.get("image_bytes", b"")
    quote = state.get("quote")
    brief = state.get("brief")
    composed = compose_image(image_bytes, quote, brief)
    logger.info(f"  Composed ({len(composed)//1024} KB)")
    return {**state, "composed_image": composed}


def build() -> any:
    g = StateGraph(PipelineState)
    g.add_node("generate_image", generate_image)
    g.add_node("compose", compose)
    g.set_entry_point("generate_image")
    g.add_edge("generate_image", "compose")
    g.add_edge("compose", END)
    return g.compile()
