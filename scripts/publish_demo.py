"""Upload the demo's scenes to the hosted demo's Modal volume (modal_jobs/demo.py).

    uv run python scripts/publish_demo.py                      # the scenes on the landing page
    uv run python scripts/publish_demo.py blender07 --dry-run  # list files and sizes only
    uv run python scripts/publish_demo.py blender07 --keep-edits

Uploads the mesh library (data/library), then each scene.json and the files it names. Edits
made while testing are undone unless --keep-edits is passed. Files already on the volume with
the same size and a newer time are skipped, unless --force.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import modal

from movable_reality.config import LIBRARY_DIR, scene_dir
from movable_reality.publish import missing_lod, referenced_files, without_edits
from movable_reality.scene import Scene

VOLUME_NAME = "movable-reality-demo"  # modal_jobs/demo.py
DEMO_SCENES = ["blender07", "blender06"]  # web/src/landing/scenes.ts, those with a scene


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("scenes", nargs="*", default=DEMO_SCENES)
    ap.add_argument("--keep-edits", action="store_true", help="publish scene.json as it is")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-no-lod", action="store_true", help="publish a scene without LOD trees")
    ap.add_argument("--force", action="store_true", help="upload files that look unchanged")
    args = ap.parse_args()

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    library = [p for p in LIBRARY_DIR.rglob("*") if p.is_file()] if LIBRARY_DIR.is_dir() else []
    print(f"library: {len(library)} files, {sum(p.stat().st_size for p in library) / 1e6:.0f} MB")
    if library and not args.dry_run:
        with volume.batch_upload(force=True) as batch:
            for p in library:
                batch.put_file(p, f"library/{p.relative_to(LIBRARY_DIR).as_posix()}")
    for sid in args.scenes:
        folder: Path = scene_dir(sid)
        scene = Scene.load(folder / "scene.json")
        if not args.keep_edits:
            undone = len(scene.history)
            scene = without_edits(scene)
            print(f"{sid}: undid {undone} logged edits")
        missing = missing_lod(scene)
        if missing and not args.allow_no_lod:
            raise SystemExit(
                f"{sid}: no LOD for {len(missing)} meshes (first: {missing[0]}); the viewer would "
                f"download every PLY. Run scripts/build_lod.py {sid}, or pass --allow-no-lod."
            )
        names = referenced_files(scene.model_dump(mode="json"), folder)
        total = sum((folder / n).stat().st_size for n in names)
        print(f"{sid}: {len(names)} files, {total / 1e6:.0f} MB")
        if args.dry_run:
            for n in names:
                print(f"  {(folder / n).stat().st_size / 1e6:8.1f} MB  {n}")
            continue

        remote = {}
        try:
            remote = {e.path: e for e in volume.listdir(f"scenes/{sid}", recursive=True)}
        except modal.exception.NotFoundError:
            pass
        with volume.batch_upload(force=True) as batch:
            sent = 0
            for n in names:
                st = (folder / n).stat()
                e = remote.get(f"scenes/{sid}/{n}")
                if not args.force and e and e.size == st.st_size and e.mtime >= st.st_mtime:
                    continue
                batch.put_file(folder / n, f"scenes/{sid}/{n}")
                sent += 1
            blob = scene.model_dump_json(indent=2).encode()
            batch.put_file(io.BytesIO(blob), f"scenes/{sid}/scene.json")
        print(f"{sid}: uploaded {sent} files and scene.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
