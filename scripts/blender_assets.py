"""Download the CC0 Poly Haven assets for the synthetic Blender rooms (blender01, blender02).

    uv run python scripts/blender_assets.py            # -> data/assets/<asset>/ + assets.json

Models come as glTF (textures alongside), floor and wall textures as JPG maps. Every
Poly Haven asset is CC0; the script records name, URL, author and licence per asset in
data/assets/assets.json, which docs/ASSETS.md mirrors. Skips files that already exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from movable_reality.config import DATA_DIR

API = "https://api.polyhaven.com"
UA = {"User-Agent": "movable-reality-hackathon (github.com/OJrogue)"}

# asset id -> resolution. Furniture at 2k, small items at 1k.
MODELS = {
    "sofa_02": "2k",
    "ArmChair_01": "2k",
    "modern_coffee_table_01": "2k",
    "side_table_01": "1k",
    "painted_wooden_cabinet": "2k",
    "potted_plant_01": "2k",
    "book_encyclopedia_set_01": "1k",
    "ceramic_vase_01": "1k",
    "potted_plant_04": "1k",
    "wooden_bowl_01": "1k",
    # blender02 (L-shaped bedroom-study)
    "vintage_day_bed": "2k",
    "painted_wooden_nightstand": "1k",
    "wooden_table_02": "2k",
    "dining_chair_02": "2k",
    "wooden_bookshelf_worn": "2k",
    "modern_arm_chair_01": "2k",
    "potted_plant_02": "2k",
    "ceramic_vase_03": "1k",
    "horse_statue_01": "1k",
}
TEXTURES = {"laminate_floor_02": "2k", "beige_wall_001": "2k"}
TEXTURE_MAPS = ("Diffuse", "nor_gl", "Rough")


def get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        tmp.write_bytes(r.read())
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "assets")
    args = ap.parse_args()

    records = []
    for asset, res in MODELS.items():
        files = get_json(f"{API}/files/{asset}")["gltf"][res]["gltf"]
        root = args.out / asset
        fetch(files["url"], root / f"{asset}.gltf")
        for rel, inc in files["include"].items():
            fetch(inc["url"], root / rel)
        records.append((asset, "model", res, root / f"{asset}.gltf"))
        print(f"model {asset} ({res})")
    for asset, res in TEXTURES.items():
        files = get_json(f"{API}/files/{asset}")
        root = args.out / asset
        for m in TEXTURE_MAPS:
            url = files[m][res]["jpg"]["url"]
            fetch(url, root / url.rsplit("/", 1)[1])
        records.append((asset, "texture", res, root))
        print(f"texture {asset} ({res})")

    manifest = []
    for asset, kind, res, path in records:
        info = get_json(f"{API}/info/{asset}")
        manifest.append(
            {
                "id": asset,
                "name": info["name"],
                "kind": kind,
                "resolution": res,
                "authors": sorted(info.get("authors", {})),
                "url": f"https://polyhaven.com/a/{asset}",
                "license": "CC0-1.0",
                "path": path.relative_to(args.out).as_posix(),
            }
        )
    (args.out / "assets.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} assets -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
