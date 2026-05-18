"""Builds the topic block injected into quote prompts. All direction lives in topics.yml."""

from __future__ import annotations

import random

import wisdom.config as cfg


def build_topic_block(theme_key: str) -> tuple[str, str]:
    """Returns (topic_block, image_hint)."""
    topics = cfg.topics()
    cat = topics.get(theme_key, {})
    if not cat:
        return f"Find an inspiring quote for the '{theme_key}' theme.", ""

    brief = cat.get("brief", "").strip()

    # Feature author spotlight ~30% of the time for ANY theme that has them
    featured = cat.get("featured_authors", [])
    if featured and random.random() < 0.30:
        pick = random.choice(featured)
        block = (
            f"TODAY: Find a quote by {pick['name'].upper()}.\n"
            f"Context: {pick['note']}\n"
            f"Choose a lesser-known gem — not their most-circulated line."
        )
        # We removed the hardcoded dark academia marble statue hint
        # so it uses the dynamic, vibrant image style prompt instead!
        return block, ""

    # Latenight: pick a weighted topic group for variety
    if "topic_groups" in cat:
        groups = cat["topic_groups"]
        weights = [g.get("weight", 10) for g in groups]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0.0
        for g, w in zip(groups, weights):
            cumulative += w
            if r <= cumulative:
                return g.get("brief", brief), ""
        return groups[-1].get("brief", brief), ""

    return brief, ""
