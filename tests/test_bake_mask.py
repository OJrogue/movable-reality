"""Alpha mask (B1): texels outside a plane's outline are transparent, holes stay opaque."""

import io

import numpy as np
import pytest
from PIL import Image

from movable_reality.bake import glb, mask, synthetic
from movable_reality.bake.pipeline import BakeOptions, bake_scene
from movable_reality.scene import Scene

TEXEL = 0.01  # 1 cm texels, 5 cm mask cells


def l_shaped_alpha() -> np.ndarray:
    """4 x 3 m quad (400 x 300 texels) covered as an L: the 1.5 x 1.2 m corner at the
    bottom right is missing. Inside lie a 0.8 m hole (a sofa) and a 4 cm crack."""
    a = np.ones((300, 400))
    a[180:, 250:] = 0.0  # missing corner, touches two borders
    a[40:120, 40:120] = 0.0  # enclosed hole
    a[150:300, 100:104] = 0.0  # thin crack reaching the border
    return a


def test_l_shape_corner_transparent_holes_and_cracks_opaque():
    keep = mask.plane_mask(l_shaped_alpha(), TEXEL)
    assert keep.shape == (300, 400)
    assert not keep[200:, 270:].any()  # the missing corner, away from its edge
    assert keep[:170, :].all()  # everything above the corner, including the hole
    assert keep[40:120, 40:120].all()  # hole under furniture is filled, not cut
    assert keep[150:300, 100:104].all()  # a crack narrower than 2 * close_m is bridged
    # The cut follows the outline to within one 5 cm cell.
    assert keep[175:, :245].all()
    assert keep[:, 250:][180:].sum() <= 5 * 150 + 5 * 120


def test_object_footprint_keeps_a_border_hole_opaque():
    a = np.ones((300, 400))
    a[0:80, 150:250] = 0.0  # sofa against the wall, no floor splats, touches the border
    assert not mask.plane_mask(a, TEXEL)[:60, 160:240].any()
    origin, u, v = np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    sofa = (np.array([1.45, 0.0, -0.05]), np.array([2.55, 0.9, 0.85]))  # world x, z cover it
    keep = mask.plane_mask(a, TEXEL, origin=origin, u=u, v=v, supports=[sofa])
    assert keep.all()


def test_full_coverage_gives_an_all_opaque_mask():
    assert mask.plane_mask(np.ones((50, 70)), TEXEL).all()


@pytest.fixture(scope="module")
def l_room(tmp_path_factory):
    d = tmp_path_factory.mktemp("bakedemo_l")
    synthetic.make_room(d, floor_spacing=0.04, ceiling_spacing=0.06, wall_spacing=0.1, shape="l")
    report = bake_scene(d, BakeOptions(texel=0.02), log=lambda _: None)
    return d, report


def _rgba_and_rect(d, p):
    doc, blob = glb.read_glb((d / p.mesh).read_bytes())
    img = np.asarray(Image.open(io.BytesIO(glb.image_bytes(doc, blob))))
    c = np.asarray(p.corners)
    return doc, img, c[0], c[1] - c[0], c[3] - c[0]


def test_l_shaped_room_bakes_transparent_corner(l_room):
    d, report = l_room
    assert report.status == "baked"
    scene = Scene.load(d / "scene.json")
    cut_share = (np.ptp(synthetic.L_CUT_X) * np.ptp(synthetic.L_CUT_Z)) / 20.0  # of 4 x 5 m
    for p in scene.planes:
        doc, img, o, du, dv = _rgba_and_rect(d, p)
        assert doc["materials"][0]["alphaMode"] == "MASK"
        assert doc["materials"][0]["alphaCutoff"] == 0.5
        assert doc["images"][0]["mimeType"] == "image/png"  # JPEG was asked, but has no alpha
        assert img.shape[2] == 4
        h, w = img.shape[:2]
        jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        world = o + (jj / w)[..., None] * du + (ii / h)[..., None] * dv
        x, z = world[..., 0], world[..., 2]
        alpha = img[..., 3]
        # Transparent 10 cm inside the cut, opaque 10 cm inside the room.
        deep_cut = (x > synthetic.L_CUT_X[0] + 0.1) & (z > synthetic.L_CUT_Z[0] + 0.1)
        in_room = (np.abs(x) < 1.9) & (np.abs(z) < 2.4) & ~synthetic.in_l_cut(x + 0.1, z + 0.1)
        in_room &= ~synthetic.in_l_cut(x - 0.1, z - 0.1)
        assert deep_cut.sum() > 1000
        assert (alpha[deep_cut] == 0).all(), p.label
        assert (alpha[in_room] == 255).all(), p.label
        assert p.transparent_fraction == pytest.approx(cut_share, abs=0.015), p.label


def test_floor_hole_under_the_sofa_stays_opaque(l_room):
    d, _ = l_room
    floor = next(p for p in Scene.load(d / "scene.json").planes if p.label == "floor")
    _, img, o, du, dv = _rgba_and_rect(d, floor)
    h, w = img.shape[:2]
    sofa = synthetic.BOXES[0]
    centre = np.array([(sofa.lo[0] + sofa.hi[0]) / 2, 0.0, (sofa.lo[2] + sofa.hi[2]) / 2])
    rel = centre - o
    col = int(rel @ du / (du @ du) * w)
    row = int(rel @ dv / (dv @ dv) * h)
    assert img[row - 5 : row + 5, col - 5 : col + 5, 3].min() == 255
    assert floor.filled_fraction > 0.03
