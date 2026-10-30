import numpy as np
import pytest
from plyfile import PlyData

from movable_reality.coverage.ply import (
    Splats,
    append_generated,
    count_generated,
    read_generated,
    read_splats,
    write_generated,
    write_splats,
)


def small_ply(path, n=10, generated=None):
    rng = np.random.default_rng(0)
    write_splats(
        path,
        rng.uniform(-1, 1, (n, 3)),
        rng.uniform(0, 1, (n, 3)),
        scales=0.02,
        opacity=0.9,
        generated=generated,
    )
    return path


def test_write_read_roundtrip(tmp_path):
    p = small_ply(tmp_path / "a.ply")
    s = read_splats(p)
    assert len(s) == 10
    assert s.opacity == pytest.approx(np.full(10, 0.9), abs=1e-5)
    assert s.scales == pytest.approx(np.full((10, 3), 0.02), rel=1e-5)
    assert (s.generated == 0).all()  # no property -> all scanned


def test_generated_property_is_added_and_other_properties_kept(tmp_path):
    p = small_ply(tmp_path / "a.ply")
    before = PlyData.read(str(p), mmap=False)["vertex"].data
    flags = np.zeros(10, np.uint8)
    flags[[2, 5]] = 1
    write_generated(p, flags)
    after = PlyData.read(str(p), mmap=False)["vertex"].data
    assert after.dtype.names == before.dtype.names + ("generated",)
    assert after.dtype["generated"] == np.uint8
    for name in before.dtype.names:
        assert np.array_equal(after[name], before[name])
    assert read_generated(p).tolist() == flags.tolist()
    assert not (tmp_path / "a.ply.tmp").exists()
    # Overwriting an existing property keeps a single column.
    write_generated(p, np.ones(10, np.uint8))
    assert PlyData.read(str(p), mmap=False)["vertex"].data.dtype.names.count("generated") == 1
    assert read_generated(p).sum() == 10


def test_write_generated_validates(tmp_path):
    p = small_ply(tmp_path / "a.ply")
    with pytest.raises(ValueError, match="expected 10"):
        write_generated(p, np.zeros(3))
    with pytest.raises(ValueError, match="0 or 1"):
        write_generated(p, np.full(10, 2))


def test_count_and_append_generated(tmp_path):
    a = small_ply(tmp_path / "a.ply", generated=np.array([0, 1] * 5, np.uint8))
    b = small_ply(tmp_path / "b.ply")
    assert count_generated([a, b, tmp_path / "missing.ply"]) == 5
    new = Splats(
        means=np.zeros((3, 3), np.float32),
        opacity=np.full(3, 0.5, np.float32),
        scales=np.full((3, 3), 0.01, np.float32),
        rgb=np.full((3, 3), 0.5, np.float32),
        generated=np.zeros(3, np.uint8),
    )
    append_generated(b, new)
    assert read_generated(b).tolist() == [0] * 10 + [1] * 3
    assert count_generated([a, b]) == 8
