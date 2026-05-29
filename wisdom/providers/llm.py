"""
LLM registry via LiteLLM.

All provider-specific code is gone — LiteLLM handles Gemini, Claude, OpenAI, etc.
To add or swap a provider: update config/llm.yml with any LiteLLM model string.
See https://docs.litellm.ai/docs/providers for all supported models.

Text generation:
  generate(prompt, role)  → runs the role's fallback chain, raises RuntimeError if all fail
"""

from __future__ import annotations

import base64
import logging
import os

import litellm

import wisdom.config as cfg

litellm.suppress_debug_info = True
logger = logging.getLogger(__name__)


def generate(prompt: str, role: str) -> tuple[str, str]:
    """Run prompt through the role's fallback chain. Raises RuntimeError if all fail. Returns (content, provider_info)"""
    role_cfg = cfg.llm_role(role)
    providers_cfg = cfg.llm_providers()
    last_err: Exception | None = None

    for p_name in role_cfg.providers:
        p = providers_cfg.get(p_name)
        if not p or not p.model:
            continue
        if p.key_env and not os.environ.get(p.key_env):
            logger.debug(f"[{role}] {p_name}: key missing — skipping")
            continue
        try:
            extra = {}
            if role_cfg.disable_thinking and p.model.startswith("gemini/"):
                extra["extra_body"] = {
                    "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}
                }
            resp = litellm.completion(
                model=p.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=role_cfg.temperature,
                max_tokens=role_cfg.max_tokens,
                num_retries=1,
                api_key=os.environ.get(p.key_env),
                **extra,
            )
            content = resp.choices[0].message.content or ""
            if not content:
                raise ValueError("Empty response content")
            logger.info(f"[{role}] ✓ {p_name} ({p.model})")
            return content, f"{p_name} ({p.model})"
        except Exception as exc:
            logger.warning(f"[{role}] {p_name} failed: {exc}")
            last_err = exc

    raise RuntimeError(
        f"All LLM providers exhausted for role '{role}'. "
        f"Chain: {role_cfg.providers}. Last error: {last_err}"
    )


def judge_image(image_bytes: bytes, prompt: str, role: str) -> tuple[str, str]:
    """Evaluate an image using a vision-capable LLM (Gemini Flash). Returns (content, provider_info)"""
    role_cfg = cfg.llm_role(role)
    providers_cfg = cfg.llm_providers()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    last_err: Exception | None = None

    for p_name in role_cfg.providers:
        p = providers_cfg.get(p_name)
        if not p or not p.model:
            continue
        if p.key_env and not os.environ.get(p.key_env):
            continue
        try:
            resp = litellm.completion(
                model=p.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=role_cfg.temperature,
                max_tokens=role_cfg.max_tokens,
                api_key=os.environ.get(p.key_env),
            )
            content = resp.choices[0].message.content or ""
            if not content:
                raise ValueError("Empty response content")
            logger.info(f"[{role}] ✓ {p_name} ({p.model})")
            return content, f"{p_name} ({p.model})"
        except Exception as exc:
            logger.warning(f"[{role}] {p_name} failed: {exc}")
            last_err = exc

    raise RuntimeError(
        f"All LLM providers exhausted for role '{role}'. Last error: {last_err}"
    )
