import os

import pytest

from movable_reality import fsutil


def test_replace_retries_while_the_target_is_held(tmp_path, monkeypatch):
    tmp, path = tmp_path / "a.tmp", tmp_path / "a.ply"
    tmp.write_text("new")
    path.write_text("old")
    calls = {"n": 0}
    real = os.replace

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("held")
        real(a, b)

    monkeypatch.setattr(fsutil.os, "replace", flaky)
    fsutil.replace_file(tmp, path, wait_s=0)
    assert path.read_text() == "new" and calls["n"] == 3


def test_replace_gives_a_hint_when_it_stays_held(tmp_path, monkeypatch):
    def held(a, b):
        raise PermissionError("held")

    monkeypatch.setattr(fsutil.os, "replace", held)
    with pytest.raises(PermissionError, match="dev server"):
        fsutil.replace_file(tmp_path / "a.tmp", tmp_path / "a.ply", tries=2, wait_s=0)
