import numpy as np
import pytest

from movable_reality.bake import gaussians as G


def random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    return G.quat_to_rotmat(q)


def test_quat_roundtrip_including_half_turns():
    rng = np.random.default_rng(0)
    r = random_rotations(rng, 500)
    # Rotations by 180 degrees about each axis and about a diagonal: w = 0.
    r = np.concatenate([r, np.diag([1, -1, -1])[None], np.diag([-1, 1, -1])[None],
                        np.diag([-1, -1, 1])[None]])  # fmt: skip
    q = G.rotmat_to_quat(r)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    assert np.all(q[:, 0] >= 0)
    assert np.allclose(G.quat_to_rotmat(q), r, atol=1e-9)


def test_covariance_is_r_s2_rt():
    q = np.array([[np.cos(0.3), 0.0, np.sin(0.3), 0.0]])  # rotation about y by 0.6 rad
    g = G.Gaussians(
        means=np.zeros((1, 3)),
        quats=q,
        log_scales=np.log([[0.1, 0.02, 0.3]]),
        opacity_logits=np.zeros(1),
        f_dc=np.zeros((1, 3)),
    )
    r = G.quat_to_rotmat(q)[0]
    expected = r @ np.diag([0.01, 0.0004, 0.09]) @ r.T
    assert np.allclose(g.covariances()[0], expected)
    assert g.opacities[0] == pytest.approx(0.5)
    assert np.allclose(g.rgb, 0.5)


def test_ply_roundtrip_and_subset_keeps_extra_properties(tmp_path):
    rng = np.random.default_rng(1)
    n = 50
    g = G.Gaussians(
        means=rng.normal(size=(n, 3)),
        quats=G.rotmat_to_quat(random_rotations(rng, n)),
        log_scales=rng.normal(-4, 0.5, (n, 3)),
        opacity_logits=rng.normal(size=n),
        f_dc=rng.normal(size=(n, 3)),
    )
    G.write_gaussians(tmp_path / "a.ply", g)
    ply = G.read_ply(tmp_path / "a.ply")
    back = G.gaussians_from_vertices(G.vertices(ply))
    assert np.allclose(back.means, g.means, atol=1e-6)
    assert np.allclose(back.f_dc, g.f_dc, atol=1e-6)

    # A PLY with an extra per-splat property (coverage lane's `generated`).
    from plyfile import PlyData, PlyElement

    v = G.vertices(ply)
    dtype = v.dtype.descr + [("generated", "u1")]
    w = np.zeros(n, dtype=dtype)
    for name in v.dtype.names:
        w[name] = v[name]
    w["generated"] = np.arange(n) % 2
    PlyData([PlyElement.describe(w, "vertex")]).write(str(tmp_path / "b.ply"))
    keep = np.arange(n) >= 10
    assert G.write_subset(tmp_path / "c.ply", G.read_ply(tmp_path / "b.ply"), keep) == 40
    c = G.vertices(G.read_ply(tmp_path / "c.ply"))
    assert list(c["generated"]) == list(w["generated"][10:])
    assert np.array_equal(c["x"], w["x"][10:])


def test_match_positions_is_exact():
    v = np.zeros(5, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")])
    v["x"] = [0.0, 1.0, 2.0, 3.0, 1.0]
    v["y"] = [0.0, 0.0, 0.0, 0.0, 1e-7]  # differs from vertex 1 only in y
    t = v[[1, 3]].copy()
    t["opacity"] = 9.0  # other properties do not matter
    assert G.match_positions(v, t).tolist() == [False, True, False, True, False]
    assert not G.match_positions(v, t[:0]).any()
