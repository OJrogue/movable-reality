"""Read the header of a Spark .RAD file.

Layout (Spark 2.4.0, rust/spark-lib/src/rad.rs, `decode_rad_header`): 4 bytes magic
"RAD0" (u32 LE 0x30444152), 4 bytes u32 LE JSON length, the JSON metadata, padding to a
multiple of 8, then the chunks. The JSON has `count` (number of splats, LOD interior
nodes included), `lodTree` (true when the file holds a LOD tree), `maxSh`, `chunks`
(one entry per 65,536 splats) and a free-form `comment` that `build-lod` fills with JSON
statistics such as `input_splat_count`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAD_MAGIC = b"RAD0"


@dataclass(frozen=True)
class RadHeader:
    count: int  # splats in the file, LOD interior nodes included
    lod_tree: bool
    max_sh: int | None
    num_chunks: int
    meta: dict[str, Any]

    @property
    def stats(self) -> dict[str, Any]:
        """Statistics `build-lod` writes into the comment (empty if absent or not JSON)."""
        comment = self.meta.get("comment")
        if not isinstance(comment, str):
            return {}
        try:
            parsed = json.loads(comment)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def parse_rad_header(data: bytes) -> RadHeader:
    """Parse the header from the first bytes of a .RAD file."""
    if len(data) < 8 or data[:4] != RAD_MAGIC:
        raise ValueError("not a RAD file (missing RAD0 magic)")
    (length,) = struct.unpack_from("<I", data, 4)
    if len(data) < 8 + length:
        raise ValueError(f"RAD header truncated: need {8 + length} bytes, have {len(data)}")
    meta = json.loads(data[8 : 8 + length])
    return RadHeader(
        count=int(meta["count"]),
        lod_tree=bool(meta.get("lodTree", False)),
        max_sh=meta.get("maxSh"),
        num_chunks=len(meta.get("chunks", [])),
        meta=meta,
    )


def read_rad_header(path: Path) -> RadHeader:
    with Path(path).open("rb") as f:
        head = f.read(8)
        if len(head) < 8 or head[:4] != RAD_MAGIC:
            raise ValueError(f"{path} is not a RAD file (missing RAD0 magic)")
        (length,) = struct.unpack_from("<I", head, 4)
        return parse_rad_header(head + f.read(length))
