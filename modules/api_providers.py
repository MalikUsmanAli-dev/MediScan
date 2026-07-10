"""
api_providers.py
-------------------
Registry of cloud vision-AI providers. Every provider here exposes an
OpenAI-compatible `/chat/completions` endpoint that accepts image inputs,
so a single generic HTTP client (see cloud_ai_engine.py) can talk to any
of them — the user just picks a provider (or "Custom") and pastes their
own API key, instead of the app being hardcoded to one vendor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProviderInfo:
    label: str
    base_url: str
    default_model: str
    suggested_models: list
    key_help: str


PROVIDERS = {
    "groq": ProviderInfo(
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="meta-llama/llama-4-scout-17b-16e-instruct",
        suggested_models=[
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
        ],
        key_help="Free key from https://console.groq.com/keys",
    ),
    "openai": ProviderInfo(
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        suggested_models=["gpt-4o-mini", "gpt-4o"],
        key_help="Key from https://platform.openai.com/api-keys",
    ),
    "openrouter": ProviderInfo(
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.2-11b-vision-instruct",
        suggested_models=[
            "meta-llama/llama-3.2-11b-vision-instruct",
            "qwen/qwen-2.5-vl-7b-instruct",
        ],
        key_help="Key from https://openrouter.ai/keys — routes to many vision models with one key",
    ),
    "together": ProviderInfo(
        label="Together AI",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-Vision-Free",
        suggested_models=["meta-llama/Llama-Vision-Free"],
        key_help="Key from https://api.together.ai/settings/api-keys",
    ),
    "custom": ProviderInfo(
        label="Custom (any OpenAI-compatible endpoint)",
        base_url="",
        default_model="",
        suggested_models=[],
        key_help="Point this at any self-hosted or third-party OpenAI-compatible /chat/completions API.",
    ),
}

DEFAULT_PROVIDER = "groq"


@dataclass
class ApiConfig:
    provider: str = DEFAULT_PROVIDER
    base_url: str = PROVIDERS[DEFAULT_PROVIDER].base_url
    api_key: str = ""
    model: str = PROVIDERS[DEFAULT_PROVIDER].default_model


def api_is_configured(cfg: Optional[ApiConfig]) -> bool:
    return bool(cfg and cfg.api_key and cfg.base_url and cfg.model)
