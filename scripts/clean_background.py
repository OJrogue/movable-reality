"""Repaint the background behind and under objects so moving one leaves no mark; splats left on
an object join it. Needs room/shell.json. Then run build_lod.py. See movable_reality/bgclean.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from movable_reality.bgclean import run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", type=Path, help="scene folder (default data/scenes/<scene>)")
    a = ap.parse_args()
    root = a.root or Path(__file__).resolve().parents[1] / "data" / "scenes" / a.scene
    print(run(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
