"""PSNR of a viewer capture (mrEvalRun, see blender_viewer_eval.py) against the GT test walk, at the
capture's own size: GT is centre-cropped to the capture's aspect, then resized. Prints the summary
and writes eval/viewer/<run>/score.json. --base <run> compares it view by view with another run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def score_view(capture: Path, gt: Path) -> float:
    """PSNR in dB of one capture against its GT frame."""
    shot = Image.open(capture).convert("RGB")
    truth = Image.open(gt).convert("RGB")
    w, h = truth.size
    keep = min(1.0, (shot.width / shot.height) / (w / h))
    tall = min(1.0, (w / h) / (shot.width / shot.height))
    cw, ch = round(w * keep), round(h * tall)
    x, y = (w - cw) // 2, (h - ch) // 2
    truth = truth.crop((x, y, x + cw, y + ch)).resize(shot.size, Image.Resampling.BICUBIC)
    err = (np.asarray(shot, float) - np.asarray(truth, float)) ** 2
    return float(10 * np.log10(255**2 / max(err.mean(), 1e-9)))


def score_run(viewer: Path, run: str, gt: Path) -> dict[str, float]:
    return {p.stem: score_view(p, gt / p.name) for p in sorted((viewer / run).glob("*.png"))}


def summary(scores: dict[str, float]) -> dict[str, float]:
    a = np.array(list(scores.values()))
    return {
        "views": len(a),
        "mean": round(float(a.mean()), 2),
        "median": round(float(np.median(a)), 2),
        "min": round(float(a.min()), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", action="append", default=[], help="other capture(s) to compare with")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "data"
    viewer = root / "scenes" / a.scene / "eval" / "viewer"
    gt = root / "raw" / a.scene / "test" / "rgb"
    scores = score_run(viewer, a.run, gt)
    out = {"run": a.run, **summary(scores), "per_view": scores, "bases": {}}
    print(a.run, summary(scores))
    for base in a.base:
        other = {k: v for k, v in score_run(viewer, base, gt).items() if k in scores}
        better = sum(scores[k] > v for k, v in other.items())
        out["bases"][base] = {**summary(other), "run_better_in": better}
        print(base, summary(other), f"{a.run} better in {better} of {len(other)}")
    (viewer / a.run / "score.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
