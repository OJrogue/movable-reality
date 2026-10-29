"""M0 smoke test for Muse Spark Open on Baseten (the edit verifier).

    uv run python scripts/smoke_baseten.py
    uv run python scripts/smoke_baseten.py --image data/smoke/sam_chair.png

Baseten accepts images by URL or base64; this sends a base64 data URL. If that is rejected,
check https://docs.baseten.co/reference/inference-api/chat-completions for the format.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

from movable_reality.clients import baseten_client
from movable_reality.config import Models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, help="optional image for a vision check")
    args = ap.parse_args()

    try:
        client = baseten_client()
        resp = client.chat.completions.create(
            model=Models.VERIFIER_OPEN,
            messages=[{"role": "user", "content": "Reply with exactly the word OK."}],
        )
        print(f"[text] reply: {resp.choices[0].message.content!r}")
        if args.image:
            mime = mimetypes.guess_type(args.image.name)[0] or "image/png"
            data = base64.b64encode(args.image.read_bytes()).decode()
            resp = client.chat.completions.create(
                model=Models.VERIFIER_OPEN,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "List the objects you see, comma-separated."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{data}"},
                            },
                        ],
                    }
                ],
            )
            print(f"[vision] reply: {resp.choices[0].message.content!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
