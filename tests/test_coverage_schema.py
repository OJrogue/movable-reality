import json

from movable_reality.scene import Coverage, Gap, NextBestView, Scene


def test_coverage_with_next_best_views_roundtrip(tmp_path):
    nbv = NextBestView(eye=(1.0, 1.6, 0.5), target=(0.0, 0.3, -1.8), visible=120, gain=0.4)
    gap = Gap(id="gap_000", center=(0.0, 0.3, -1.8), radius=0.5, score=0.9, next_best_views=[nbv])
    s = Scene(scene_id="t", coverage=Coverage(gaps=[gap], gap_share=0.12, generated_splats=7))
    path = tmp_path / "scene.json"
    s.save(path)
    loaded = Scene.load(path)
    assert loaded == s
    assert loaded.coverage is not None
    assert loaded.coverage.gaps[0].next_best_views[0].eye == (1.0, 1.6, 0.5)


def test_gap_without_next_best_views_still_loads(tmp_path):
    data = {
        "scene_id": "t",
        "coverage": {
            "gap_map": None,
            "gaps": [{"id": "g", "center": [0, 0, 0], "radius": 0.2, "score": 1.0}],
            "generated_splats": 0,
        },
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    cov = Scene.load(path).coverage
    assert cov is not None
    assert cov.gaps[0].next_best_views == [] and cov.gap_share is None
