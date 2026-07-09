"""Quote generation prompts. Two modes: real_author and internet_found."""

from __future__ import annotations

_AUDIENCE = (
    "global inspirational quotes account, ages 18–35"
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
Generate 5 DISTINCT real quotes from real named persons for a {_AUDIENCE}.

{topic_block}

{_BANNED}
{no_cliches}

Rules:
- REAL quote — do not invent, paraphrase, or composite.
- BREVITY: Focus on extreme brevity. Target 6-15 words so it can be read in under 5 seconds.
- Maximum {max_words} words per quote.
- Named author — not "Unknown" or "Anonymous".
- Choose lesser-known gems over widely-circulated lines.
- Specific and concrete — must be a visceral, raw truth about modern life, human nature, power, strategy, or relationships.
- Must feel instantly shareable and have high psychological impact (makes the reader pause and reflect).

Uniqueness / Viral Potential score (1–10): how fresh, relatable, and shareable is this exact phrasing?
  10 = mind-blowing truth almost nobody has seen  |  1 = posted on every motivational page
{avoid}
Return ONLY a JSON array with exactly 5 items:
[
  {{"quote":"exact text","author":"Full Name","uniqueness":9}},
  {{"quote":"exact text","author":"Full Name","uniqueness":7}},
  ...
]
Replace uniqueness with your actual score."""

    else:  # internet_found
        return f"""\
Find 5 DISTINCT quotes from the internet — Reddit, Pinterest, Tumblr, Instagram captions, Twitter/X,
or a traditional proverb / folk saying. Author may be known or unknown.

{topic_block}

{_BANNED}
{no_cliches}

Rules:
- DO NOT write or invent — find things that genuinely exist.
- BREVITY: Focus on extreme brevity. Target 6-15 words so it can be read in under 5 seconds.
- CRITICAL: Focus entirely on viral, bitter-truth, street-smart realizations, and sharp psychological observations (e.g., subtle gaslighting dynamics, boundaries, human nature, power, strategy, and street-smart wisdom).
- Example vibes/quotes of the exact type desired:
  * "Horses that go to war don't dance at weddings."
  * "The same people who mock you when you are nothing will pretend they knew you when you are something."
  * "The people you would take a bullet for are often the ones holding the gun."
  * "You don't lose friends. You just learn who the real ones are."
- Maximum {max_words} words per quote.
- Must feel instantly shareable — the kind people screenshot and send.
- Must be a visceral, raw truth about modern life, ambition, or relationships.
- It should make the reader instantly think "This is exactly how I feel" or "This is a cheat code to life".
- Author: real name if known, "Unknown" otherwise.

Uniqueness / Viral Potential score (1–10): how fresh, relatable, and shareable is this exact phrasing?
  10 = mind-blowing truth almost nobody has seen  |  1 = posted on every motivational page
{avoid}
Return ONLY a JSON array with exactly 5 items:
[
  {{"quote":"exact text","author":"Name or Unknown","uniqueness":9}},
  {{"quote":"exact text","author":"Name or Unknown","uniqueness":7}},
  ...
]
Replace uniqueness with your actual score."""
