"""Build the premade mesh library: data/assets/<id>/ -> data/library/<id>.glb + catalog.json.

    uv run python scripts/blender_assets.py     # once: the CC0 models
    uv run python scripts/build_library.py      # packs what is missing, then renders thumbnails
    uv run python scripts/build_library.py --force --no-thumbs

Thumbnails come from Blender (scripts/library_thumbs.py; BLENDER or `blender` on PATH, else the
default Windows install). The viewer shows labels only for items without one.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from movable_reality.config import DATA_DIR, LIBRARY_DIR, REPO_ROOT
from movable_reality.library import CATALOG, CURATED, LibraryItem, pack_model, read_glb_json

ASSETS = DATA_DIR / "assets"
WINDOWS_BLENDER = Path("C:/Program Files/Blender Foundation/Blender 4.5/blender.exe")


def find_blender() -> str | None:
    found = os.environ.get("BLENDER") or shutil.which("blender")
    return found or (str(WINDOWS_BLENDER) if WINDOWS_BLENDER.exists() else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--force", action="store_true", help="pack and render again")
    ap.add_argument("--no-thumbs", action="store_true")
    args = ap.parse_args()

    credits = {a["id"]: a for a in json.loads((ASSETS / "assets.json").read_text(encoding="utf-8"))}
    old = {}
    if (LIBRARY_DIR / CATALOG).exists() and not args.force:
        old = {d["id"]: d for d in json.loads((LIBRARY_DIR / CATALOG).read_text(encoding="utf-8"))}
    items = []
    for c in CURATED:
        src, out = ASSETS / c.asset / f"{c.asset}.gltf", LIBRARY_DIR / f"{c.asset}.glb"
        if not src.exists():
            print(f"{c.asset:<28} missing in data/assets, skipped (blender_assets.py)")
            continue
        if out.exists() and c.asset in old:
            size, tris = tuple(old[c.asset]["size"]), old[c.asset]["triangles"]
        else:
            size, tris = pack_model(src, out)
        credit = credits.get(c.asset, {})
        if not credit.get("license"):
            raise SystemExit(f"{c.asset}: no licence in assets.json (run blender_assets.py)")
        items.append(
            LibraryItem(
                id=c.asset, label=c.label, category=c.category, on_surface=c.on_surface,
                size=tuple(round(v, 3) for v in size), triangles=tris, glb=out.name,
                author=", ".join(credit.get("authors", [])), source_url=credit.get("url", ""),
                license=credit.get("license", ""),
            )  # fmt: skip
        )
        w, h, d = items[-1].size
        print(f"{c.asset:<28} {w:5.2f} x {h:4.2f} x {d:4.2f} m  {tris:>7} tris  "
              f"{out.stat().st_size / 1e6:5.1f} MB")  # fmt: skip
        read_glb_json(out)  # a broken file fails the build here, not in the viewer

    thumbs = LIBRARY_DIR / "thumbs"
    if not args.no_thumbs:
        blender = find_blender()
        todo = [i.id for i in items if args.force or not (thumbs / f"{i.id}.png").exists()]
        if blender and todo:
            print(f"rendering {len(todo)} thumbnails with {blender}")
            subprocess.run(
                [blender, "-b", "-P", str(REPO_ROOT / "scripts" / "library_thumbs.py"), "--",
                 "--dir", str(LIBRARY_DIR), "--ids", *todo],
                check=True, stdout=subprocess.DEVNULL,
            )  # fmt: skip
        elif not blender:
            print("no Blender found: no thumbnails")
    for item in items:
        if (thumbs / f"{item.id}.png").exists():
            item.thumb = f"thumbs/{item.id}.png"

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    (LIBRARY_DIR / CATALOG).write_text(
        json.dumps([i.model_dump(mode="json") for i in items], indent=2), encoding="utf-8"
    )
    total = sum(p.stat().st_size for p in LIBRARY_DIR.rglob("*") if p.is_file())
    print(f"{len(items)} items, {total / 1e6:.0f} MB in {LIBRARY_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
