"""OpenAI-compatible clients for the Meta Model API and Baseten."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from .config import BASETEN_BASE_URL, META_BASE_URL, require_env


def meta_client() -> OpenAI:
    return OpenAI(base_url=META_BASE_URL, api_key=require_env("MODEL_API_KEY"))


def meta_async_client() -> AsyncOpenAI:
    # SAM 3.1 video requests stream for over 10 min on 152 frames (floor: 731 s, 31 Oct).
    return AsyncOpenAI(base_url=META_BASE_URL, api_key=require_env("MODEL_API_KEY"), timeout=1800.0)


def baseten_client() -> OpenAI:
    return OpenAI(base_url=BASETEN_BASE_URL, api_key=require_env("BASETEN_API_KEY"))


def baseten_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASETEN_BASE_URL, api_key=require_env("BASETEN_API_KEY"))
