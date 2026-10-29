import json
import struct

import pytest

from movable_reality.lod.rad import parse_rad_header, read_rad_header


def rad_bytes(meta: dict, payload: bytes = b"") -> bytes:
    body = json.dumps(meta).encode()
    pad = b"\0" * ((8 - len(body) % 8) % 8)
    return b"RAD0" + struct.pack("<I", len(body)) + body + pad + payload


def test_parse_header_fields_and_build_lod_stats():
    stats = {"input_splat_count": 4000, "method": "BhattLod { lod_base: 1.75 }"}
    meta = {
        "version": 1,
        "type": "gsplat",
        "count": 7486,
        "maxSh": 0,
        "lodTree": True,
        "chunks": [{"offset": 0, "bytes": 10}],
        "comment": json.dumps(stats),
    }
    h = parse_rad_header(rad_bytes(meta, b"x" * 10))
    assert (h.count, h.lod_tree, h.max_sh, h.num_chunks) == (7486, True, 0, 1)
    assert h.stats["input_splat_count"] == 4000


def test_header_without_lod_tree_or_comment():
    h = parse_rad_header(rad_bytes({"count": 3, "chunks": []}))
    assert not h.lod_tree and h.stats == {}


def test_rejects_other_files_and_truncated_headers():
    with pytest.raises(ValueError, match="RAD0"):
        parse_rad_header(b"ply\nformat binary_little_endian 1.0\n")
    data = rad_bytes({"count": 1})
    with pytest.raises(ValueError, match="truncated"):
        parse_rad_header(data[:12])


def test_read_header_from_file_reads_only_the_header(tmp_path):
    path = tmp_path / "a.rad"
    path.write_bytes(rad_bytes({"count": 9, "lodTree": True}, b"\xff" * 4096))
    assert read_rad_header(path).count == 9
