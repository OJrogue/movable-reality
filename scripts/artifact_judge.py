"""Model judge for room-surface artifacts: a Meta vision model looks at each viewer frame and
lists the rendering artifacts it sees on walls, floor and ceiling. Needs no ground truth, so
it works on real rooms too (the pixel score in blender_viewer_eval.py needs GT renders).

    uv run python scripts/artifact_judge.py --scene blender01 --run baseline
    uv run python scripts/artifact_judge.py --scene room01 --run walk1 --model wonder

Reads data/scenes/<scene>/eval/viewer/<run>/*.png (the viewer frames, mrEvalRun), writes
judge_<model>.json next to them: per frame the artifacts (kind, surface, severity 1-3, box in
0..1 image coordinates) and a summary (frames with any artifact, severity sum). Every answer is
cached under data/cache/chat/, so a rerun on the same frames is free. Evaluation only: the
viewer and the pipeline never call it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re

from PIL import Image

from movable_reality.agent.llm import cached_chat
from movable_reality.clients import baseten_async_client, meta_async_client
from movable_reality.config import Models, scene_dir

PROMPT = (
    "This is a frame from a 3D Gaussian-splat reconstruction of a real room, shown in a viewer. "
    "Judge ONLY the room surfaces: walls, floor and ceiling (ignore furniture and objects "
    "themselves, but do report surface damage next to or behind them). List every rendering "
    "artifact you see on those surfaces, such as: blur or smear (a patch much blurrier than its "
    "surroundings), streak or needle (long thin coloured strokes), floater or haze (fuzzy blobs "
    "or fog in front of a surface), hole (black, missing or see-through area), seam or step (a "
    "visible edge where two surface pieces do not match), ghost (a faint duplicate of an "
    "object or of furniture that is not there), blotch (wrong-coloured patch), texture stretch. "
    "Do not report soft lighting, real shadows, or real texture such as wood grain. "
    "Answer with JSON only, no prose: "
    '{"artifacts": [{"kind": "...", "surface": "wall|floor|ceiling", "severity": 1-3, '
    '"box": [x0, y0, x1, y1], "note": "few words"}]} with box in 0..1 image coordinates. '
    "Severity 1 = only visible on close inspection, 3 = obvious at a glance. An empty list "
    "means clean surfaces."
)

MODELS = {
    "spark": ("meta", Models.FAST),
    "wonder": ("meta", Models.AGENT),
    "open": ("baseten", Models.VERIFIER_OPEN),
}


def data_url(path, width: int = 960) -> str:
    img = Image.open(path).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse(text: str) -> list[dict] | None:
    """The artifact list from a model answer (JSON, maybe in a code fence), None if unusable."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        arts = json.loads(m.group(0)).get("artifacts")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(arts, list):
        return None
    out = []
    for a in arts:
        if not isinstance(a, dict):
            continue
        sev = a.get("severity", 1)
        out.append({"kind": str(a.get("kind", "?")), "surface": str(a.get("surface", "?")),
                    "severity": int(sev) if str(sev).isdigit() else 1,
                    "box": a.get("box"), "note": str(a.get("note", ""))[:80]})  # fmt: skip
    return out


async def judge(paths, model: str, concurrency: int = 6) -> dict[str, dict]:
    provider, model_id = MODELS[model]
    client = meta_async_client() if provider == "meta" else baseten_async_client()
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            resp = await cached_chat(
                client, provider, model=model_id, max_tokens=4000,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url(p)}}]}],
            )  # fmt: skip
        text = resp["choices"][0]["message"].get("content") or ""
        arts = parse(text)
        return p.stem, {"artifacts": arts, "raw": None if arts is not None else text[:400]}

    return dict(await asyncio.gather(*(one(p) for p in paths)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="spark", choices=sorted(MODELS))
    args = ap.parse_args()
    run = scene_dir(args.scene) / "eval" / "viewer" / args.run
    # view PNGs only (not the worst_ pairs or sheet.png)
    paths = sorted(p for p in run.glob("*.png") if p.stem.isdigit())
    if not paths:
        print(f"no frames in {run}")
        return 1
    res = asyncio.run(judge(paths, args.model))
    arts = [a for r in res.values() for a in (r["artifacts"] or [])]
    summary = {
        "frames": len(res),
        "unparsed": sum(r["artifacts"] is None for r in res.values()),
        "frames_with_artifacts": sum(bool(r["artifacts"]) for r in res.values()),
        "artifacts": len(arts),
        "severity_sum": sum(a["severity"] for a in arts),
        "by_surface": {
            s: sum(a["surface"] == s for a in arts) for s in ("wall", "floor", "ceiling")
        },
        "by_kind": {
            k: sum(a["kind"] == k for a in arts) for k in sorted({a["kind"] for a in arts})
        },
    }
    out = {"scene": args.scene, "run": args.run, "model": MODELS[args.model][1],
           "summary": summary, "frames": res}  # fmt: skip
    (run / f"judge_{args.model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
