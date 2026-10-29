import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


report = load_script("bench_report")


def make_log(dts, *, headless=False, target=13.9, device="pico", mode="lod", scene="bench_300000"):
    dts = [float(d) for d in dts]
    t = np.concatenate([[0.0], np.cumsum(dts[1:])]).round(2).tolist()
    n = len(dts)
    return {
        "schema": 1,
        "kind": "mr-frame-log",
        "created": "2026-10-29T21:00:00.000Z",
        "scene": scene,
        "mode": mode,
        "budget": False,
        "path": "walk",
        "xr": True,
        "headless": headless,
        "device": {"label": device, "gpu_renderer": "Adreno (TM) 650"},
        "target_ms": target,
        "target_source": "xr",
        "total_splats": 300_000,
        "events": [],
        "frames": {
            "t": t,
            "dt": dts,
            "cpu_ms": [2.0] * n,
            "drawn": [250_000] * n,
            "cut": [250_000] * n,
            "lod_splat_count": [500_000] * n,
            "lod_render_scale": [1.0] * n,
            "lod": [1] * n,
        },
    }


def test_steady_72hz_run_passes():
    rng = np.random.default_rng(0)
    dts = 13.9 + rng.normal(0, 0.05, 72 * 120)
    r = report.analyse(make_log(dts), "a.json")
    assert r.p50 == pytest.approx(13.9, abs=0.05)
    assert r.p99 < 13.9 + 0.2
    assert r.over_share == 0
    assert r.fps == pytest.approx(72, rel=0.01)
    assert r.window_s == pytest.approx(60, abs=0.1)
    # p99 can sit a hair over 13.9 from jitter alone; the bar is on missed frames too.
    assert r.fps_drop == pytest.approx(0, abs=0.01)


def test_missed_frames_and_late_slowdown_fail():
    # First minute at 72 fps, second minute every 10th frame missed.
    first = [13.9] * (72 * 60)
    last = ([13.9] * 9 + [27.8]) * (72 * 6)
    r = report.analyse(make_log(first + last), "b.json")
    assert r.over_share == pytest.approx(0.05, abs=0.005)
    assert r.p99 == pytest.approx(27.8)
    assert r.fps_last < r.fps_first
    assert r.fps_drop > 0.05
    assert not r.passes


def test_percentiles_match_numpy_and_viewer():
    dts = [10, 1, 4, 2, 3]
    r = report.analyse(make_log(dts), "c.json")
    assert r.p50 == 3
    assert r.p99 == pytest.approx(9.76)


def test_short_runs_compare_halves():
    r = report.analyse(make_log([13.9] * 720), "d.json")  # 10 s
    assert r.window_s == pytest.approx(5, abs=0.1)


def test_report_writes_table_and_keeps_headless_out(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dev.json").write_text(json.dumps(make_log([13.9] * 1000)))
    (raw / "hl.json").write_text(json.dumps(make_log([500.0] * 20, headless=True)))
    (raw / "junk.json").write_text("{}")
    rows, skipped = report.load_rows(sorted(raw.glob("*.json")))
    assert len(rows) == 2 and len(skipped) == 1
    md = report.render([r for r in rows if not r.headless], skipped, "raw", headless=False)
    assert "| pico | bench_300000 | 300k | lod |" in md
    assert "PASS" in md
    assert "not device" not in md
    md_h = report.render([r for r in rows if r.headless], [], "headless", headless=True)
    assert "not device" in md_h


def test_empty_log_is_skipped(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps(make_log([])))
    rows, skipped = report.load_rows([p])
    assert rows == [] and "no frames" in skipped[0]
