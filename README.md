# Social Media Wisdom Automation

An end-to-end AI pipeline that generates, composes, and publishes daily wisdom content to Instagram and YouTube Shorts — fully automated via GitHub Actions.

---

## What It Does

Every day, at scheduled times, the system:

1. Generates a unique, non-repetitive quote using Kimi (Moonshot) with Gemini as fallback
2. Creates a cinematic AI image via Leonardo FLUX.2 Pro with multi-provider fallback
3. Composes the image with per-style typography, color-coded text, gradient overlays
4. Renders a 23-second Reel with Ken Burns zoom, static text, and background music
5. Posts to Instagram and YouTube Shorts simultaneously
6. Sends a styled email report with live links

Zero manual work after setup.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                           │
│   7 AM IST · 10 AM IST · 2 PM IST · 11 PM IST (4×/day)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline (pipeline.py)                     │
│                                                                 │
│   Quote Agent  →  Design Agent  →  Media Agent  →  Video       │
│       │                │               │             │          │
│   LangGraph        LangGraph       LangGraph      FFmpeg        │
│   state machine    state machine   state machine               │
└──────────┬──────────────┬───────────────┬────────────┬─────────┘
           │              │               │            │
           ▼              ▼               ▼            ▼
      Kimi / Gemini  Kimi / Gemini   Leonardo      Ken Burns
      (LiteLLM)      (LiteLLM)       FLUX.2 Pro    Zoom +
                                     + fallbacks   Text Overlay
                                                   + Music Fade
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Storage                                │
│                                                                 │
│   SQLite DB (local)  ←→  GitHub Releases (db tag, CI sync)     │
│   Dedup · Pending · Posted history                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
┌──────────────────┐       ┌───────────────────────┐
│    Instagram     │       │    YouTube Shorts      │
│  Graph API v19   │       │   Data API v3          │
│                  │       │                        │
│  GitHub Releases │       │  Direct upload         │
│  (media-pool)    │       │  (resumable)           │
│  as CDN          │       │                        │
└──────────────────┘       └───────────────────────┘
```

---

## Pipeline Stages

### 1 — Quote Generation (`wisdom/agents/quote.py`)

A LangGraph state machine that generates a unique, high-quality quote.

- Alternates between **real author** and **internet-found** quote modes
- Runs up to 2 attempts, each scored on uniqueness (1–10)
- Deduplicates against recently posted quotes from the DB
- Falls back to a curated YAML quote pool if LLM attempts fail
- Strips Gemini markdown fences and validates JSON before accepting

```
start → pick_mode → attempt_1 → score ≥ threshold? → done
                  → attempt_2 → below threshold?   → curated fallback
```

### 2 — Design Brief (`wisdom/agents/design.py`)

Picks a visual style and generates a full creative brief for the image.

- Style picker selects from theme-locked or all applicable styles (weighted)
- Creative brief LLM call generates: image prompt, color palette, font, layout, gradient type, highlight phrase, Ken Burns flag
- Falls back to a sensible default brief if the LLM call fails
- Brief is a typed `DesignBrief` dataclass — no raw dicts escape this stage

### 3 — Media: Image + Compose (`wisdom/agents/media.py`)

Generates an image and composites it with text and overlays.

```
generate image → compose → done
```

Image generation uses the provider fallback chain (see Provider Architecture). In offline mode, a PIL gradient fallback is used instead.

### 4 — Video Composition (`wisdom/composers/reel.py`)

FFmpeg renders a 23-second Reel in three independent layers:

```
Layer 0  bg.jpg      Raw photo       → Ken Burns zoom (0→1.12× over full duration)
Layer 1  overlay.png Gradient RGBA   → Static (no zoom)
Layer 2  text.png    Text RGBA       → Static + fade out 2s before end

[bg] → zoompan → [bg_zoomed]
[bg_zoomed][overlay] → overlay → [bg_ov]
[bg_ov][text_fade]   → overlay → [vout]
```

Separating the layers ensures text never zooms with the background — only the photo moves.

Audio: background music (theme-matched MP3) looped, volume-ducked, faded out in the last 1.5s.

### 5 — Image Composition (`wisdom/composers/card.py`)

PIL-based renderer that produces three outputs per run:

| Output | Contents | Used by |
|---|---|---|
| `compose_base()` | Raw photo, scaled/cropped to 1080×1920 | FFmpeg zoom layer |
| `compose_overlay_layer()` | Transparent RGBA gradient PNG | FFmpeg static overlay |
| `compose_text_layer()` | Transparent RGBA text PNG | FFmpeg static text |
| `compose_image()` | All three merged (JPEG) | Thumbnail + Instagram image fallback |

Supports 6 gradient types: `gradient_bottom`, `gradient_top`, `gradient_center`, `solid`, `vignette`, `none`.

Text rendering: auto word-wrap, font size scaling, highlight phrase in accent color. Each style has its own `text_color`, `highlight_color`, and `author_color` — no style uses plain white text. The `cinematic_female_protagonist` style uses the **Architects Daughter** handwritten font.

---

## Provider Architecture

### LLM (`wisdom/providers/llm.py`)

Powered by **LiteLLM** — swap any provider by changing a single model string in `config/llm.yml`.

Primary provider is **Kimi** (`moonshot/kimi-latest-128k`), with **Gemini 2.5 Flash** as fallback. If `MOONSHOT_API_KEY` is missing or exhausted, Gemini takes over automatically.

| Role | Primary | Fallback | Purpose |
|---|---|---|---|
| `quote_generation` | kimi | gemini | Quote + uniqueness score |
| `style_picker` | kimi | gemini | Style selection |
| `creative_brief` | kimi | gemini | Image prompt + design brief |

### Image (`wisdom/providers/image.py`)

Six-provider fallback chain — tried in order until one succeeds:

```
leonardo_flux_pro  →  gemini_flash  →  gemini_imagen  →  leonardo  →  pollinations  →  gradient
(FLUX.2 Pro,          (2.5-flash-      (imagen-4.0-      (API)        (free API)       (PIL local
 810×1440 → resize)    image)           fast-001)                                        fallback)
```

`GradientFallback` always succeeds — guarantees the pipeline never crashes on image generation.

---

## Storage

### SQLite Database (`wisdom/storage/db.py`)

Two tables:

- **`posted`** — quote text, author, theme, style, timestamp. Used for deduplication.
- **`pending`** — generated but not yet posted content. Decouples generation from publishing.

**CI sync**: In GitHub Actions, the DB is downloaded from a GitHub Release asset (`db` tag) at the start of each run, and re-uploaded at the end. This makes SQLite stateful across ephemeral runners without any external database service.

### GitHub Releases CDN (`wisdom/storage/uploader.py`)

Instagram's Graph API requires a public URL to ingest video — it cannot accept direct file uploads.

The uploader pushes the MP4 and thumbnail JPEG to a `media-pool` GitHub Release, gets the public `browser_download_url`, passes it to Instagram, then deletes the asset after posting.

---

## Configuration

Everything is driven by YAML — adding a new theme or style requires zero Python changes.

```
config/
├── app.yml           Image dimensions, reel duration, output paths
├── themes.yml        Theme definitions (platforms, hashtags, YouTube metadata, schedule)
├── styles.yml        Visual styles (rendering params, fonts, gradient types, Ken Burns flag)
├── llm.yml           LLM provider config (model, temperature, max_tokens per role)
├── image.yml         Image provider config (models, API keys, fallback chain)
├── topics.yml        Topic categories used in quote prompts
└── curated_quotes.yml  Fallback quotes when LLM is unavailable
```

### Adding a new theme

```yaml
# config/themes.yml
mytheme:
  name: My Theme
  format: reel
  max_words: 20
  platforms: [instagram, youtube]
  hashtags: ["#Tag1", "#Tag2"]
  youtube:
    title_template: "My Theme"
    tags: ["tag1", "tag2"]
    category_id: "22"
    privacy: public
  enabled: true
```

That's it. Add the cron line to the workflow and it runs.

---

## Automation Schedule

| Theme | Time (IST) | UTC Cron | Frequency |
|---|---|---|---|
| Morning Motivation | 7:00 AM | `30 1 * * *` | Daily |
| She Feels | 10:00 AM | `30 4 * * *` | Daily |
| Rotating¹ | 2:00 PM | `30 8 * * *` | Daily |
| Late Night Feels | 1:15 AM | `45 19 * * *` | Daily |

¹ Rotates through: Wisdom → Love → Mindfulness → Dark Academia (cycles by day-of-year mod 4)

Manual trigger available via `workflow_dispatch` with theme selection and offline mode toggle.

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── post.yml          GitHub Actions schedule + manual trigger
├── assets/
│   ├── audio/                Theme-matched background music (MP3)
│   ├── fonts/                40+ Google Fonts (TTF, pre-downloaded)
│   └── static/               Static fallback images per theme
├── config/                   All YAML configuration (no secrets)
├── wisdom/
│   ├── agents/
│   │   ├── pipeline.py       Top-level orchestrator
│   │   ├── quote.py          Quote generation LangGraph agent
│   │   ├── design.py         Design brief LangGraph agent
│   │   ├── media.py          Image generation + compose LangGraph agent
│   │   ├── _prompt_builder.py  Quote prompt construction
│   │   └── _topic_builder.py   Topic selection helpers
│   ├── composers/
│   │   ├── card.py           PIL image + text + overlay compositor
│   │   └── reel.py           FFmpeg three-layer video renderer
│   ├── platforms/
│   │   ├── instagram.py      Instagram Graph API (Reels + images)
│   │   └── youtube.py        YouTube Data API v3 (Shorts + OAuth)
│   ├── providers/
│   │   ├── llm.py            LiteLLM provider registry (Kimi → Gemini fallback)
│   │   └── image.py          Multi-provider image generation + fallback
│   ├── storage/
│   │   ├── db.py             SQLite + GitHub Releases sync
│   │   └── uploader.py       GitHub Releases CDN for Instagram media
│   ├── cli.py                Click CLI (run / dry-run / generate / post)
│   ├── config.py             YAML loader → typed dataclasses
│   └── schemas.py            All dataclasses and TypedDicts
├── .env.example              Required environment variables
└── requirements.txt
```

---

## Local Development

```bash
# Install
pip install -r requirements.txt
sudo apt-get install ffmpeg   # or: brew install ffmpeg

# Configure
cp .env.example .env
# fill in API keys

# One-time font download
python3 -c "from wisdom.composers.card import _ensure_fonts; _ensure_fonts()"

# Dry run (generates content, saves to output/, never posts)
python -m wisdom.cli dry-run latenight

# Full run + post
python -m wisdom.cli run morning

# YouTube OAuth (one-time)
python -m wisdom.cli youtube-auth
```

---

## Required Secrets (GitHub Actions)

| Secret | Description |
|---|---|
| `MOONSHOT_API_KEY` | Moonshot Kimi — primary LLM provider |
| `GEMINI_API_KEY` | Gemini 2.5 Flash — LLM fallback + image generation |
| `LEONARDO_API_KEY` | Leonardo.ai — image fallback |
| `HF_API_KEY` | HuggingFace — image fallback |
| `INSTAGRAM_ACCESS_TOKEN` | Instagram Graph API token |
| `INSTAGRAM_BUSINESS_ID` | Instagram Business Account ID |
| `YOUTUBE_CLIENT_ID` | Google OAuth client ID |
| `YOUTUBE_CLIENT_SECRET` | Google OAuth client secret |
| `YOUTUBE_REFRESH_TOKEN` | Long-lived YouTube refresh token |
| `GITPROVIDER_TOKEN` | GitHub PAT (Contents + Actions write) |
| `SMTP_USER` | Gmail address for email reports |
| `SMTP_PASS` | Gmail App Password |
| `EMAIL_RECIPIENT` | Report destination email |

`GITHUB_TOKEN` and `GITHUB_REPOSITORY` are provided automatically by Actions.
