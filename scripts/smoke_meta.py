"""M0 smoke test for the Meta Model API.

    uv run python scripts/smoke_meta.py                                  # models + Muse Wonder
    uv run python scripts/smoke_meta.py --image room.jpg --noun chair    # + SAM 3.1 on an image
    uv run python scripts/smoke_meta.py --video data/scenes/room01/sam_input.mp4 --noun chair

Costs a few cents. SAM results are cached under data/cache/sam/.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from movable_reality.clients import meta_client
from movable_reality.config import DATA_DIR, Models
from movable_reality.sam import paste_mask, segment_image, segment_video

EXPECTED = [Models.AGENT, Models.FAST, Models.IMAGE, Models.TRANSCRIBE, Models.SAM]


def check_models() -> None:
    ids = sorted(m.id for m in meta_client().models.list())
    print(f"[models] {len(ids)} available: {', '.join(ids)}")
    missing = [m for m in EXPECTED if m not in ids]
    if missing:
        print(f"[models] WARNING, not listed for this key: {missing}")


def check_wonder() -> None:
    resp = meta_client().chat.completions.create(
        model=Models.AGENT,
        messages=[{"role": "user", "content": "Reply with exactly the word OK."}],
    )
    print(f"[wonder] reply: {resp.choices[0].message.content!r}")
    if resp.usage:
        print(f"[wonder] tokens in/out: {resp.usage.prompt_tokens}/{resp.usage.completion_tokens}")


def check_sam_image(path: Path, noun: str) -> None:
    instances = asyncio.run(segment_image(path, noun))
    print(f"[sam image] '{noun}': {len(instances)} instance(s)")
    img = Image.open(path).convert("RGB")
    overlay = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(0)
    for inst in instances:
        print(f"  id={inst.object_id} box={inst.box} conf={inst.confidence}")
        full = paste_mask(inst, img.width, img.height)
        color = rng.integers(64, 255, size=3)
        overlay[full] = 0.5 * overlay[full] + 0.5 * color
    out = DATA_DIR / "smoke" / f"sam_{noun.replace(' ', '_')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay.astype(np.uint8)).save(out)
    print(f"[sam image] overlay saved to {out}")


def check_sam_video(path: Path, noun: str) -> None:
    instances = asyncio.run(segment_video(path, noun))
    frames = Counter(i.object_id for i in instances)
    print(f"[sam video] '{noun}': {len(frames)} object(s) across {len(instances)} masks")
    for oid, n in sorted(frames.items()):
        print(f"  id={oid} seen in {n} frame(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--noun", default="chair")
    ap.add_argument("--skip-wonder", action="store_true")
    args = ap.parse_args()

    failures = 0
    steps = [("models", check_models)]
    if not args.skip_wonder:
        steps.append(("wonder", check_wonder))
    if args.image:
        steps.append(("sam image", lambda: check_sam_image(args.image, args.noun)))
    if args.video:
        steps.append(("sam video", lambda: check_sam_video(args.video, args.noun)))
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - smoke test reports everything
            failures += 1
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
    print("OK" if not failures else f"{failures} step(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
