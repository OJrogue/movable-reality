"""Turn frame-time logs into a results table (lane renderer).

    uv run python scripts/bench_report.py               # benchmarks/raw/ -> benchmarks/RESULTS.md
    uv run python scripts/bench_report.py --headless    # benchmarks/headless/ -> .../RESULTS.md
    uv run python scripts/bench_report.py --logs a.json b.json --out /tmp/r.md

Logs come from the viewer's /__bench endpoint (web/src/perf/frameLog.ts). Per log:
p50 / p99 frame interval, share of frames over budget, FPS over the first and the last
minute and the drop between them, splats drawn, and pass/fail against the bar in
AGENTS.md (p99 <= target, < 1% over budget, FPS drop <= 5%).

"Over budget" means an interval longer than 1.2x the target (a missed vsync): raw
intervals jitter around the display period (13.8 to 14.1 ms at 72 Hz), so a strict
comparison would count jitter. The viewer's HUD and budget controller use the same factor.
For runs shorter than two minutes, "first" and "last minute" become the first and last
half of the run. Percentiles use numpy's default (linear) definition, like the viewer.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"
MISS_FACTOR = 1.2
MAX_OVER_SHARE = 0.01
MAX_FPS_DROP = 0.05


@dataclass
class Row:
    file: str
    created: str
    device: str
    gpu: str
    scene: str
    total_splats: int | None
    mode: str
    budget: bool
    path: str
    xr: bool
    headless: bool
    target_ms: float
    target_source: str
    duration_s: float
    frames: int
    p50: float
    p99: float
    over_share: float
    fps: float
    fps_first: float
    fps_last: float
    fps_drop: float
    drawn_median: float
    cpu_p99: float
    window_s: float

    @property
    def passes(self) -> bool:
        return (
            self.p99 <= self.target_ms
            and self.over_share < MAX_OVER_SHARE
            and self.fps_drop <= MAX_FPS_DROP
        )


def fps_of(dt: np.ndarray) -> float:
    total = float(dt.sum())
    return len(dt) * 1000.0 / total if total > 0 else 0.0


def analyse(log: dict, name: str) -> Row:
    frames = log["frames"]
    t = np.asarray(frames["t"], dtype=float)
    dt = np.asarray(frames["dt"], dtype=float)
    if len(dt) == 0:
        raise ValueError(f"{name}: no frames")
    target = float(log["target_ms"])
    end = t[-1]
    duration = float(dt.sum()) / 1000.0
    window = min(60_000.0, (end - t[0]) / 2) if len(t) > 1 else 0.0
    first = dt[t < t[0] + window] if window > 0 else dt
    last = dt[t >= end - window] if window > 0 else dt
    fps_first, fps_last = fps_of(first), fps_of(last)
    drop = (fps_first - fps_last) / fps_first if fps_first > 0 else 0.0
    drawn = np.asarray(frames.get("drawn") or [0], dtype=float)
    cpu = np.asarray(frames.get("cpu_ms") or [0], dtype=float)
    device = log.get("device", {})
    return Row(
        file=name,
        created=str(log.get("created", "")),
        device=str(device.get("label", "unknown")),
        gpu=str(device.get("gpu_renderer") or ""),
        scene=str(log.get("scene", "?")),
        total_splats=log.get("total_splats"),
        mode=str(log.get("mode", "?")),
        budget=bool(log.get("budget", False)),
        path=str(log.get("path", "?")),
        xr=bool(log.get("xr", False)),
        headless=bool(log.get("headless", False)),
        target_ms=target,
        target_source=str(log.get("target_source", "?")),
        duration_s=duration,
        frames=len(dt),
        p50=float(np.percentile(dt, 50)),
        p99=float(np.percentile(dt, 99)),
        over_share=float(np.mean(dt > target * MISS_FACTOR)),
        fps=fps_of(dt),
        fps_first=fps_first,
        fps_last=fps_last,
        fps_drop=drop,
        drawn_median=float(np.median(drawn)),
        cpu_p99=float(np.percentile(cpu, 99)),
        window_s=window / 1000.0,
    )


def load_rows(paths: list[Path]) -> tuple[list[Row], list[str]]:
    rows, skipped = [], []
    for p in paths:
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
            if log.get("kind") != "mr-frame-log":
                skipped.append(f"{p.name}: not a frame log")
                continue
            rows.append(analyse(log, p.name))
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            skipped.append(f"{p.name}: {e}")
    return rows, skipped


def fmt_count(n: float | int | None) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "?"
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def sort_key(r: Row):
    return (r.device, r.total_splats or 0, r.scene, r.mode, r.budget, r.created)


def render(rows: list[Row], skipped: list[str], source: str, headless: bool) -> str:
    out = ["# Renderer benchmark results", ""]
    if headless:
        out += [
            "**Headless Chromium (SwiftShader, software rendering). These are not device",
            "numbers**: they only show that bench mode, the log endpoint and this report work.",
            "",
        ]
    out += [
        f"Generated by `scripts/bench_report.py` from `{source}` ({len(rows)} logs).",
        "Bar (AGENTS.md): p99 <= target, < 1% of frames over budget, FPS drop <= 5% between",
        "the first and the last minute. Over budget = interval > 1.2x target (a missed vsync).",
        "Runs shorter than 2 min compare their first and last halves (window column).",
        "",
        "| device | scene | splats | mode | budget | xr | path | s | frames | target ms "
        "| p50 ms | p99 ms | over | fps | fps first | fps last | drop | drawn | cpu p99 ms "
        "| window s | pass | log |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=sort_key):
        out.append(
            f"| {r.device} | {r.scene} | {fmt_count(r.total_splats)} | {r.mode} "
            f"| {'on' if r.budget else 'off'} | {'yes' if r.xr else 'no'} | {r.path} "
            f"| {r.duration_s:.0f} | {r.frames} | {r.target_ms:.1f} | {r.p50:.1f} | {r.p99:.1f} "
            f"| {r.over_share * 100:.1f}% | {r.fps:.1f} | {r.fps_first:.1f} | {r.fps_last:.1f} "
            f"| {r.fps_drop * 100:+.1f}% | {fmt_count(r.drawn_median)} | {r.cpu_p99:.1f} "
            f"| {r.window_s:.0f} | {'PASS' if r.passes else 'fail'} | `{r.file}` |"
        )
    gpus = sorted({(r.device, r.gpu) for r in rows})
    if gpus:
        out += ["", "GPUs reported by WebGL:", ""]
        out += [f"- {d}: {g or 'not exposed'}" for d, g in gpus]
    if skipped:
        out += ["", "Skipped:", ""] + [f"- {s}" for s in skipped]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="report benchmarks/headless instead")
    ap.add_argument("--logs", type=Path, nargs="+", help="explicit log files")
    ap.add_argument("--out", type=Path, help="output Markdown file")
    args = ap.parse_args()

    if args.logs:
        paths = args.logs
        source = "the given files"
    else:
        src = BENCH_DIR / ("headless" if args.headless else "raw")
        paths = sorted(src.glob("*.json"))
        source = str(src.relative_to(REPO_ROOT))
    out = args.out or (BENCH_DIR / ("headless" if args.headless else "") / "RESULTS.md")
    rows, skipped = load_rows(paths)
    headless = args.headless or (bool(rows) and all(r.headless for r in rows))
    if not args.headless and any(r.headless for r in rows):
        skipped += [f"{r.file}: headless log among device logs" for r in rows if r.headless]
        rows = [r for r in rows if not r.headless]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, skipped, source, headless), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} logs, {len(skipped)} skipped)")
    for r in sorted(rows, key=sort_key):
        print(
            f"  {r.device:12} {r.scene:16} {r.mode:6} p50 {r.p50:6.1f}  p99 {r.p99:7.1f}  "
            f"over {r.over_share * 100:5.1f}%  fps {r.fps:5.1f}  {'PASS' if r.passes else 'fail'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
