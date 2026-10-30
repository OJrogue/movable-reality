"""Stage 3: Muse Wonder watches the capture and lists the movable objects as SAM nouns.

    uv run python scripts/inventory.py --scene room01            # writes inventory.json
    uv run python scripts/inventory.py --scene room01 --force

Uploads sam_input.mp4 once (file id cached by content hash), asks Muse Wonder for short
concrete nouns SAM 3.1 can match, and always adds "floor", "wall" and "ceiling" for
alignment and baking. The raw answer is cached under data/cache/inventory/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys

from movable_reality.clients import meta_async_client
from movable_reality.config import CACHE_DIR, Models, scene_dir
from movable_reality.sam import sha256_file, upload_video

STRUCTURE = ["floor", "wall", "ceiling"]
PROMPT = """This video is a slow walk through one room. List every piece of furniture and
every movable object that is clearly visible, as JSON only, no prose:
{"objects": [{"noun": "armchair", "count": 1, "on": null},
             {"noun": "vase", "count": 1, "on": "coffee table"}]}
Rules: "noun" is a short concrete name SAM can segment ("sofa", "floor lamp", "potted plant",
"coffee table", "books"), lower case, singular, at most two words, one entry per kind of object
with its count. "on" names the furniture it stands on, or null for the floor. Leave out walls,
floor, ceiling, windows, doors, built-in fixtures, light switches, rugs and wall art."""


def parse_objects(text: str) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in the answer: {text[:200]}")
    data = json.loads(m.group(0))
    out, seen = [], set()
    for o in data.get("objects", []):
        noun = str(o.get("noun", "")).strip().lower()
        if not noun or noun in seen or noun in STRUCTURE:
            continue
        seen.add(noun)
        out.append({"noun": noun, "count": int(o.get("count") or 1), "on": o.get("on")})
    return out


async def ask(video) -> str:
    sha = sha256_file(video)
    key = hashlib.sha256(f"{Models.AGENT}|{sha}|{PROMPT}".encode()).hexdigest()[:24]
    cache = CACHE_DIR / "inventory" / f"{key}.txt"
    if cache.exists():
        return cache.read_text()
    file_id = await upload_video(video)
    resp = await meta_async_client().responses.create(
        model=Models.AGENT,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {"type": "input_file", "file_id": file_id},
                ],
            }
        ],
    )
    text = resp.output_text
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = scene_dir(args.scene)
    out = root / "inventory.json"
    if out.exists() and not args.force:
        print(f"{out} exists (--force to redo)")
        return 0
    video = root / "sam_input.mp4"
    if not video.exists():
        print(f"{video} missing; run scripts/extract_frames.py first", file=sys.stderr)
        return 1
    objects = parse_objects(asyncio.run(ask(video)))
    nouns = [o["noun"] for o in objects] + STRUCTURE
    out.write_text(
        json.dumps({"model": Models.AGENT, "objects": objects, "nouns": nouns}, indent=2)
    )
    for o in objects:
        print(f"  {o['count']} x {o['noun']}" + (f" (on {o['on']})" if o["on"] else ""))
    print(f"wrote {out}: {len(nouns)} nouns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
