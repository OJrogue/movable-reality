"""Whole pipeline in one command: phone video -> movable scene in the viewer.

    uv run python scripts/run_pipeline.py data/raw/room01.mp4 --scene room01 --focal35 26
    uv run python scripts/run_pipeline.py --scene room01                  # resume / refresh
    uv run python scripts/run_pipeline.py --scene room01 --from lift --ceiling-height 2.5
    uv run python scripts/run_pipeline.py --scene room01 --dry-run        # show the plan only

Stages, in order (SAM runs next to the GPU reconstruction, since it needs only the clip):

    frames      extract_frames.py         sharpest frames, SAM clip, intrinsics (--focal35)
    inventory   inventory.py              Muse Wonder lists the objects
    sam         lift_ids.py --sam-only    SAM 3.1 per noun (cached)        | in parallel
    reconstruct modal reconstruct.py      MapAnything, SfM, gsplat (L40S)  |
    lift        lift_ids.py               objects, floor, metric scale (object heights)
    objects     modal reconstruct.py --objects   every object trained alone (no ghosts)
    complete    complete_objects.py --mesh       SAM 3D Objects: closed backs and meshes
    shell       room_shell.py             room as textured floor, wall and ceiling quads
    lod         build_lod.py              Spark LOD tree per object
    bake        bake_planes.py --labels floor    floor as an opaque textured quad
    coverage    score_coverage.py         gap map and next-best views

A stage is skipped when its outputs exist and are newer than its inputs, and rerun (with
--force) when an input changed, so a rerun after a fix only redoes what depends on it.
Each stage logs to data/scenes/<id>/logs/<stage>.log, and data/scenes/<id>/pipeline.json
records the state of every stage (the viewer's /scenes/ page shows it).

Paid parts: Modal GPU time (about $1 to $2 for a 150-frame room), Meta Model API calls
(cached under data/cache/, so reruns are free).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from movable_reality.config import scene_dir

ROOT = Path(__file__).resolve().parents[1]
PY = ["uv", "run", "python"]
MODAL = ["uv", "run", "modal"]


@dataclass
class Stage:
    name: str
    what: str
    cmd: list[str]
    inputs: list[str]  # relative to the scene folder; the newest must be older than outputs
    outputs: list[str]
    force_flag: bool = True  # the script takes --force
    optional: bool = False  # a failure warns and the run goes on
    state: dict = field(default_factory=dict)


def stages(a: argparse.Namespace) -> list[Stage]:
    s = a.scene
    metric = []
    if a.ceiling_height:
        metric = ["--ceiling-height", str(a.ceiling_height)]
    elif a.scale:
        metric = ["--scale", str(a.scale)]
    skip = ["--skip", a.skip] if a.skip else []
    frames = [*PY, "scripts/extract_frames.py", str(a.video or "<video>"), "--scene", s]
    if a.focal35:
        frames += ["--focal35", str(a.focal35)]
    recon = [*MODAL, "run", "modal_jobs/reconstruct.py", "--scene", s, "--preset", a.preset]
    return [
        Stage("frames", "sharpest frames and SAM clip", frames, [],
              ["frames.json", "sam_input.mp4"]),
        Stage("inventory", "Muse Wonder object list", [*PY, "scripts/inventory.py", "--scene", s],
              ["sam_input.mp4"], ["inventory.json"]),
        Stage("sam", "SAM 3.1 masks per noun", [*PY, "scripts/lift_ids.py", "--scene", s,
              "--sam-only", *skip], ["inventory.json"], [], force_flag=False),
        Stage("reconstruct", "MapAnything + SfM + gsplat on Modal", recon, ["frames.json"],
              ["splat/scene.ply"], force_flag=False),
        Stage("lift", "objects, floor and metric scale",
              [*PY, "scripts/lift_ids.py", "--scene", s, *skip, *metric],
              ["splat/scene.ply", "inventory.json"], ["ids/gaussian_ids.npy", "scene.json"]),
        Stage("objects", "per-object training on Modal",
              [*MODAL, "run", "modal_jobs/reconstruct.py", "--scene", s, "--objects"],
              ["ids/gaussian_ids.npy"], ["objects_trained/report.json"], force_flag=False),
        Stage("complete", "SAM 3D Objects completions and meshes",
              [*PY, "scripts/complete_objects.py", "--scene", s, "--mesh"],
              ["objects_trained/report.json"], ["completion.json"], optional=True),
        Stage("shell", "room as textured quads", [*PY, "scripts/room_shell.py", "--scene", s],
              ["ids/gaussian_ids.npy"], ["room/shell.json"], optional=True),
        Stage("lod", "LOD trees", [*PY, "scripts/build_lod.py", s],
              ["objects_trained/report.json", "completion.json"], ["lod"], optional=True),
        Stage("bake", "floor quad", [*PY, "scripts/bake_planes.py", "--scene", s, "--labels",
              "floor"], ["objects_trained/report.json"], ["planes"], optional=True),
        Stage("coverage", "gap map", [*PY, "scripts/score_coverage.py", "--scene", s],
              ["objects_trained/report.json"], ["coverage"], optional=True),
    ]  # fmt: skip


NAMES = [st.name for st in stages(argparse.Namespace(
    scene="x", video=None, focal35=None, preset="quality", ceiling_height=None, scale=None,
    skip="",
))]  # fmt: skip


def mtime(p: Path) -> float | None:
    if not p.exists():
        return None
    if p.is_dir():
        times = [f.stat().st_mtime for f in p.rglob("*") if f.is_file()]
        return max(times) if times else None
    return p.stat().st_mtime


def freshness(root: Path, st: Stage) -> str:
    """'done', 'stale' (outputs older than an input) or 'todo' (an output is missing)."""
    if st.name == "sam":  # no outputs of its own; the lift's output means SAM ran
        return "done" if (root / "ids" / "gaussian_ids.npy").exists() else "todo"
    outs = [mtime(root / o) for o in st.outputs]
    if any(t is None for t in outs):
        return "todo"
    ins = [t for t in (mtime(root / i) for i in st.inputs) if t is not None]
    return "stale" if ins and max(ins) > min(outs) + 1 else "done"


class Report:
    def __init__(self, root: Path, plan: list[Stage]):
        self.path = root / "pipeline.json"
        self.lock = threading.Lock()
        old = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data = {"scene": root.name, "stages": old.get("stages", {}), "order": NAMES}
        for st in plan:
            self.data["stages"].setdefault(st.name, {"state": "todo"})

    def set(self, name: str, **kw) -> None:
        with self.lock:
            self.data["stages"].setdefault(name, {}).update(kw)
            self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.path.write_text(json.dumps(self.data, indent=2))


def run_stage(root: Path, st: Stage, force: bool, report: Report) -> bool:
    cmd = list(st.cmd) + (["--force"] if force and st.force_flag else [])
    log = root / "logs" / f"{st.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    report.set(st.name, state="running", started=time.strftime("%H:%M:%S"),
               log=log.relative_to(root).as_posix(), cmd=" ".join(cmd))  # fmt: skip
    print(f"[{st.name}] {st.what}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with log.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, encoding="utf-8", errors="replace")  # fmt: skip
        last = ""
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if line.strip():
                last = line.strip()
        code = proc.wait()
    dt = round(time.time() - t0, 1)
    ok = code == 0
    report.set(st.name, state="done" if ok else "failed", seconds=dt, last_line=last[:300])
    tail = "" if ok else f"; last line: {last}\n         full log: {log}"
    print(f"[{st.name}] {'ok' if ok else f'FAILED (exit {code})'} in {dt:.0f} s{tail}", flush=True)
    return ok


def preflight(a: argparse.Namespace, root: Path) -> list[str]:
    problems = []
    if not (root / "frames.json").exists() and not a.video:
        problems.append(f"{root} has no frames yet: pass the video path")
    if a.video and not Path(a.video).exists():
        problems.append(f"video {a.video} not found")
    if not shutil.which("ffmpeg"):
        problems.append("ffmpeg is not on PATH")
    env_file = ROOT / ".env"
    keys = env_file.read_text() if env_file.exists() else ""
    for k in ("MODEL_API_KEY", "HF_TOKEN"):
        if k not in os.environ and f"{k}=" not in keys:
            problems.append(f"{k} missing from the environment and .env")
    return problems


def sam3d_deployed() -> bool:
    out = subprocess.run([*MODAL, "app", "list", "--json"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")  # fmt: skip
    try:
        apps = json.loads(out.stdout)
    except json.JSONDecodeError:
        return True  # cannot tell; complete_objects.py reports a missing app itself
    return any(
        a.get("description") == "movable-reality-sam3d" and a.get("state") == "deployed"
        for a in apps
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video", nargs="?", help="phone video (needed only for the first run)")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--focal35", type=float, help="35 mm-equivalent focal length (phone EXIF)")
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--ceiling-height", type=float, help="tape-measured ceiling height, m")
    m.add_argument("--scale", type=float, help="metric correction factor")
    ap.add_argument(
        "--preset",
        default="quality",
        choices=["quality", "fast"],
        help="gsplat preset: quality (30k steps) or fast (7k, a quick look)",
    )
    ap.add_argument("--skip", default="wall", help="SAM nouns to leave out (default: wall)")
    ap.add_argument("--from", dest="start", choices=NAMES, help="rerun from this stage on")
    ap.add_argument("--to", dest="stop", choices=NAMES, help="stop after this stage")
    ap.add_argument("--only", choices=NAMES, help="run one stage (forced)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = ap.parse_args()

    root = scene_dir(a.scene)
    plan = stages(a)
    lo = NAMES.index(a.only or a.start) if (a.only or a.start) else 0
    hi = NAMES.index(a.only or a.stop) if (a.only or a.stop) else len(NAMES) - 1
    forced = set(NAMES[lo : hi + 1]) if (a.only or a.start) else set()
    if a.video and not (root / "frames.json").exists():
        forced.add("frames")

    print(f"scene {a.scene} ({root})")
    todo: list[Stage] = []
    for i, st in enumerate(plan):
        fresh = freshness(root, st)
        if i < lo or i > hi:
            verdict = "out of range"
        elif st.name in forced:
            verdict = "run (forced)"
        elif fresh == "done":
            verdict = "skip (up to date)"
        else:
            verdict = f"run ({fresh})"
        if verdict.startswith("run"):
            todo.append(st)
            if st.name != "sam":  # SAM only fills a cache; the lift reads it
                forced.update(NAMES[i + 1 : hi + 1])  # every later stage depends on this one
        print(f"  {st.name:<12} {verdict:<20} {st.what}")
    if a.dry_run:
        return 0
    problems = preflight(a, root)
    if problems:
        print("cannot start:\n  " + "\n  ".join(problems))
        return 2
    if any(st.name == "complete" for st in todo) and not sam3d_deployed():
        print("SAM 3D Objects is not deployed on Modal; deploying (first time about 9 min)")
        subprocess.run([*MODAL, "deploy", "modal_jobs/sam3d.py"], cwd=ROOT, check=False)

    root.mkdir(parents=True, exist_ok=True)
    report = Report(root, plan)
    for st in plan:
        report.set(st.name, state="queued" if st in todo else freshness(root, st))
    t_all = time.time()
    names = [st.name for st in todo]
    i = 0
    failed: list[str] = []
    while i < len(todo):
        st = todo[i]
        # frames --force wipes the old frames: only when a video is given for a scene that has
        # them already.
        has = freshness(root, st) != "todo"
        force = has and (st.name in forced or freshness(root, st) == "stale")
        if st.name == "frames":
            force = has and bool(a.video)
        if st.name == "sam" and "reconstruct" in names:
            # SAM needs only the clip: run it beside the GPU job.
            rec = todo[names.index("reconstruct")]
            results: dict[str, bool] = {}

            def sam(stage: Stage = st, out: dict[str, bool] = results) -> None:
                out["sam"] = run_stage(root, stage, False, report)

            th = threading.Thread(target=sam)
            th.start()
            results["reconstruct"] = run_stage(root, rec, False, report)
            th.join()
            if not all(results.values()):
                failed += [k for k, v in results.items() if not v]
                break
            i = names.index("reconstruct") + 1
            continue
        if not run_stage(root, st, force, report):
            failed.append(st.name)
            if not st.optional:
                break
        i += 1

    mins = (time.time() - t_all) / 60
    print(f"\n{'FAILED at ' + ', '.join(failed) if failed else 'done'} after {mins:.1f} min")
    print(f"status: {report.path}")
    if (root / "scene.json").exists():
        print(f"viewer: http://localhost:5173/?scene={a.scene}   (cd web && npm run dev)")
        print("all scenes: http://localhost:5173/scenes/")
    return 1 if any(not s.optional for s in plan if s.name in failed) else 0


if __name__ == "__main__":
    sys.exit(main())
