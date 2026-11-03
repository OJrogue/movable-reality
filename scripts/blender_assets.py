"""Download the CC0 Poly Haven assets for the synthetic Blender rooms (blender01 to blender07).

    uv run python scripts/blender_assets.py            # -> data/assets/<asset>/ + assets.json

Models come as glTF (textures alongside), floor, wall and fabric textures as JPG maps, the
sky outside the blender06 window as an HDR. Every Poly Haven asset is CC0. The
blender06 posters and blender07 paintings are public-domain images from Wikimedia Commons
(POSTERS). The script
records name, URL, author and licence per asset in data/assets/assets.json, which
docs/ASSETS.md mirrors. Skips files that already exist.
"""

from __future__ import annotations

import argparse
import json
import re
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
    # blender06 (the realistic bedroom, scripts/blender_bedroom.py)
    "painted_wooden_chair_01": "2k",
    "desk_lamp_arm_01": "2k",
    "alarm_clock_01": "1k",
    "carved_wooden_elephant": "1k",
    "Ukulele_01": "2k",
    "wicker_basket_02": "2k",
    "modern_ceiling_lamp_01": "1k",
    "wall_clock": "1k",
    "standing_picture_frame_01": "1k",
    # blender07 (the realistic living room, scripts/blender_living.py)
    "modern_wooden_cabinet": "2k",
    "coffee_table_round_01": "2k",
    "gaming_console": "1k",
    "ceramic_vase_02": "1k",
    "ceramic_vase_04": "1k",
    "wooden_bowl_02": "1k",
    # only in the mesh library (src/movable_reality/library.py)
    "sofa_03": "2k",
    "Ottoman_01": "2k",
    "throw_pillows_01": "2k",
    "brass_vase_01": "1k",
    "mantel_clock_01": "1k",
}
TEXTURES = {
    "laminate_floor_02": "2k",
    "beige_wall_001": "2k",
    # blender06
    "wood_floor": "2k",
    "white_stucco": "2k",
    "white_oak_veneer": "2k",
    "poly_wool_herringbone": "2k",
    "rough_linen": "2k",
    "waffle_pique_cotton": "2k",
    "cotton_jersey": "2k",
}
HDRIS = {"eilenriede_park": "4k"}  # blender06: the view out of the window
# blender06 posters: Wikimedia Commons file -> local name. Both are public domain.
COMMONS = "https://commons.wikimedia.org/w/api.php"
POSTERS = {
    "Tsunami_by_hokusai_19th_century.jpg": "poster_great_wave",
    "NASA-Apollo8-Dec24-Earthrise.jpg": "poster_earthrise",
    # blender07 paintings (modernist, all public domain)
    "Kandinsky_-_Jaune_Rouge_Bleu.jpg": "painting_yellow_red_blue",
    "Piet_Mondriaan,_1930_-_Mondrian_Composition_II_in_Red,_Blue,_and_Yellow.jpg": (
        "painting_mondrian"
    ),
    "Vassily_Kandinsky,_1913_-_Color_Study,_Squares_with_Concentric_Circles.jpg": (
        "painting_kandinsky"
    ),
    "Hilma_af_Klint_-_The_Ten_Largest_No._3_-_Youth_-_1907.jpg": "painting_af_klint",
    "Ad_Parnassum,_Paul_Klee_(1932,_274).jpg": "painting_klee",
}
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


def fetch_posters(out: Path) -> list[dict]:
    """Download the poster images (1920 px wide) and check that Commons lists each one as
    public domain."""
    records = []
    for title, name in POSTERS.items():
        q = (
            f"{COMMONS}?action=query&format=json&prop=imageinfo&iiprop=url|extmetadata"
            f"&iiurlwidth=1920&titles=File:{title}"
        )
        info = next(iter(get_json(q)["query"]["pages"].values()))["imageinfo"][0]
        meta = info["extmetadata"]
        licence = meta["LicenseShortName"]["value"]
        if licence.lower() != "public domain":
            raise SystemExit(f"{title}: licence {licence!r}, expected public domain")
        fetch(info["thumburl"], out / "posters" / f"{name}.jpg")
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
        records.append(
            {
                "id": name,
                "name": title,
                "kind": "poster",
                "resolution": "1920px",
                "authors": [artist],
                "url": info["descriptionurl"],
                "license": "Public domain",
                "path": f"posters/{name}.jpg",
            }
        )
        print(f"poster {name} ({licence})")
    return records


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
    for asset, res in HDRIS.items():
        url = get_json(f"{API}/files/{asset}")["hdri"][res]["hdr"]["url"]
        root = args.out / asset
        fetch(url, root / url.rsplit("/", 1)[1])
        records.append((asset, "hdri", res, root))
        print(f"hdri {asset} ({res})")

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
    manifest += fetch_posters(args.out)
    (args.out / "assets.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} assets -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
