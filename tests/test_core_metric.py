import numpy as np
import pytest

from movable_reality.metric import HeightPrior, object_height, parse_height, scale_from_heights

PRIORS = {
    "sofa": HeightPrior(0.85, 0.7, 1.0, True),
    "armchair": HeightPrior(0.9, 0.75, 1.1, True),
    "coffee table": HeightPrior(0.45, 0.35, 0.55, True),
    "potted plant": HeightPrior(0.6, 0.2, 2.0, True),
    "vase": HeightPrior(0.3, 0.15, 0.6, False),
}


def test_parse_height_tolerates_prose_and_widens_range():
    reply = '{"height_m": 0.9, "min_m": 0.95, "max_m": 1.1, "on_floor": true}'
    p = parse_height(f"Sure:\n```json\n{reply}\n```")
    assert (p.height_m, p.min_m, p.max_m, p.on_floor) == (0.9, 0.9, 1.1, True)
    with pytest.raises(ValueError):
        parse_height('{"height_m": 9, "min_m": 8, "max_m": 10}')


def test_recovers_true_scale_from_furniture():
    s_true = 1.8  # recon is 1.8x too small
    true_h = {"sofa#0": ("sofa", 0.82), "armchair#0": ("armchair", 0.95),
              "coffee table#0": ("coffee table", 0.42)}  # fmt: skip
    measured = {k: (n, 0.0, h / s_true) for k, (n, h) in true_h.items()}
    s, votes = scale_from_heights(measured, PRIORS)
    assert s == pytest.approx(s_true, rel=0.08)
    assert all(v.used for v in votes)


def test_items_off_the_floor_and_outliers_do_not_vote():
    s_true = 2.0
    measured = {
        "sofa#0": ("sofa", 0.0, 0.85 / s_true),
        "armchair#0": ("armchair", 0.0, 0.9 / s_true),
        "coffee table#0": ("coffee table", 0.0, 0.45 / s_true),
        "vase#0": ("vase", 0.2, 0.5),  # not a floor object
        "armchair#1": ("armchair", 0.3, 0.4),  # floating base
        # SAM label covering a plant and the shelf behind it: implies 3 m at s = 2
        "potted plant#0": ("potted plant", 0.0, 1.5),
    }
    s, votes = scale_from_heights(measured, PRIORS)
    assert s == pytest.approx(s_true, rel=0.02)
    by = {v.object_id: v for v in votes}
    assert not by["vase#0"].used and not by["armchair#1"].used
    assert not by["potted plant#0"].used
    # a narrow prior outweighs a wide one
    assert by["coffee table#0"].weight > 10 * by["potted plant#0"].weight


def test_needs_two_votes():
    s, _ = scale_from_heights({"sofa#0": ("sofa", 0.0, 0.4)}, PRIORS)
    assert s is None


def test_object_height_ignores_floaters():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.uniform(0, 0.8, 5000), [3.0] * 20])
    lo, hi = object_height(y)
    assert lo < 0.05 and 0.75 < hi < 0.82
