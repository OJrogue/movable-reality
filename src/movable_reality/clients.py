"""OpenAI-compatible clients for the Meta Model API and Baseten."""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from .config import BASETEN_BASE_URL, META_BASE_URL, require_env


def meta_client() -> OpenAI:
    return OpenAI(base_url=META_BASE_URL, api_key=require_env("MODEL_API_KEY"))


def meta_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=META_BASE_URL, api_key=require_env("MODEL_API_KEY"))


def baseten_client() -> OpenAI:
    return OpenAI(base_url=BASETEN_BASE_URL, api_key=require_env("BASETEN_API_KEY"))
