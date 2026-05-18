"""Quote generation prompts. Two modes: real_author and internet_found."""

from __future__ import annotations

_AUDIENCE = (
    "global inspirational quotes account with a large Indian following, ages 18–35"
)

_BANNED = """\
Hard bans — reject any quote with:
- Weak hedges: "some of us", "sometimes", "maybe", "perhaps"
- Abstract-noun clichés: "healing", "energy", "vibes", "journey", "warrior", "storm", "bloom"
- Hollow verbs: "thrive", "manifest", "align", "elevate", "glow up"
- Empty openers: "Life is...", "We all...", "At the end of the day..."
- Brand-name spiritual: "the universe", "divine timing", "higher self"
- Generic advice: "be yourself", "chase your dreams", "trust the process"\
"""

_CLICHES: dict[str, str] = {
    "morning": '"believe in yourself", "rise and shine", "hustle hard", "warrior"',
    "wisdom": '"everything happens for a reason", "be the change", "your journey"',
    "love": '"soulmates", "love conquers all", "you complete me", "red flags"',
    "mindfulness": '"be present", "let it go", "inner peace", "heal yourself"',
    "goodnight": '"count your blessings", "tomorrow is a new day", "sweet dreams"',
    "latenight": '"time heals", "let go", "you deserve better", "healing is not linear"',
    "womenpower": '"boss babe", "girl boss", "she believed she could", "know your worth"',
}


def _avoid_quotes(recent_quotes: list[str]) -> str:
    if not recent_quotes:
        return ""
    lines = "\n".join(f'- "{q}"' for q in recent_quotes)
    return f"\nDO NOT use any of these (already posted recently):\n{lines}\n"


def build_quote_prompt(
    theme_key: str,
    mode: str,
    topic_block: str,
    max_words: int,
    recent_quotes: list[str],
) -> str:
    cliches = _CLICHES.get(theme_key, "")
    avoid = _avoid_quotes(recent_quotes)
    no_cliches = f"Avoid: {cliches}" if cliches else ""

    if mode == "real_author":
        return f"""\
Find ONE real quote from a real named person for a {_AUDIENCE}.

{topic_block}

{_BANNED}
{no_cliches}

Rules:
- REAL quote — do not invent, paraphrase, or composite
- BREVITY: Favor shorter quotes (10-15 words) that hit hard immediately.
- Maximum {max_words} words
- Named author — not "Unknown"
- Choose lesser-known gems over widely-circulated lines
- Specific and concrete — not vague philosophy
- Must be a visceral, raw truth about modern life, ambition, or relationships
- It should make the reader instantly think "This is exactly how I feel"

Uniqueness score (1–10): how fresh and non-clichéd is this exact phrasing?
  10 = almost nobody has seen this line  |  1 = posted on every motivational page
{avoid}
Return ONLY a JSON array with 1 item:
[{{"quote":"exact text","author":"Full Name","uniqueness":7}}]
Replace 7 with your actual score."""

    else:  # internet_found
        return f"""\
Find ONE quote from the internet — Reddit, Pinterest, Tumblr, Instagram captions, Twitter/X,
or a traditional proverb / folk saying. Author may be known or unknown.

{topic_block}

{_BANNED}
{no_cliches}

Rules:
- DO NOT write or invent — find something that genuinely exists
- BREVITY: Favor shorter quotes (8-15 words) that hit hard immediately.
- Maximum {max_words} words
- Must feel instantly shareable — the kind people screenshot and send
- Must be a visceral, raw truth about modern life, ambition, or relationships
- It should make the reader instantly think "This is exactly how I feel"
- Author: real name if known, "Unknown" otherwise

Uniqueness score (1–10): how fresh and non-clichéd is this exact phrasing?
  10 = almost nobody has seen this line  |  1 = posted on every motivational page
{avoid}
Return ONLY a JSON array with 1 item:
[{{"quote":"exact text","author":"Name or Unknown","uniqueness":7}}]
Replace 7 with your actual score."""
