"""Visual verifier: Muse Spark Open on Baseten looks at the viewer before and after an edit and
answers whether the edit did what the user asked (ARCHITECTURE stage 7)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..clients import baseten_async_client
from ..config import Models
from .llm import cached_chat

PROMPT = (
    "You check edits to a 3D room viewer. The first image is the view before the edit, the "
    'second after it. The user asked: "{instruction}". The edit tool reported: {summary}. '
    "Does the second image show that request carried out (objects moved, turned, removed or "
    "added as asked) with no obvious damage such as a missing or floating object? Answer "
    "YES or NO on the first line, then one short sentence why."
)


@dataclass
class Verdict:
    ok: bool | None  # None: no usable answer, so the edit stays unverified
    reason: str


def parse_verdict(text: str) -> Verdict:
    first = text.strip().splitlines()[0] if text.strip() else ""
    if re.match(r"^\W*yes\b", first, re.IGNORECASE):
        ok: bool | None = True
    elif re.match(r"^\W*no\b", first, re.IGNORECASE):
        ok = False
    else:
        ok = None
    rest = text.strip()[len(first) :].strip() or first
    return Verdict(ok, rest[:300])


async def verify_edit(
    instruction: str, summary: str, before_jpeg: str, after_jpeg: str, client=None
) -> Verdict:
    """before/after are data URLs (data:image/jpeg;base64,...) from the viewer's mrCapture()."""
    client = client or baseten_async_client()
    resp = await cached_chat(
        client,
        "baseten",
        model=Models.VERIFIER_OPEN,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT.format(instruction=instruction, summary=summary),
                    },
                    {"type": "image_url", "image_url": {"url": before_jpeg}},
                    {"type": "image_url", "image_url": {"url": after_jpeg}},
                ],
            }
        ],
        max_tokens=2000,  # a reasoning model: its thinking counts against this
    )
    msg = resp["choices"][0]["message"]
    verdict = parse_verdict(msg.get("content") or "")
    if verdict.ok is None and not verdict.reason:
        verdict.reason = f"no answer ({resp['choices'][0].get('finish_reason')})"
    return verdict
