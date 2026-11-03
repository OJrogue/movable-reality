"""What the hosted demo needs from a scene folder. Pure functions; scripts/publish_demo.py uploads.

Only files the manifest names go up, so frames, the input video and training output stay private
and the volume stays small.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .scene import Scene

# Pipeline bookkeeping named by the manifest that neither the viewer nor the server reads.
SKIP_KEYS = {"ids", "ids_ply"}
# Files the viewer fetches by convention, without the manifest naming them.
UNNAMED_FILES = ("room/shell.json",)


def without_edits(scene: Scene) -> Scene:
    """The scene as the pipeline wrote it: every logged edit undone, the history empty."""
    clean = scene.model_copy(deep=True)
    while clean.undo():
        pass
    clean.history = []
    return clean


def missing_lod(scene: Scene) -> list[str]:
    """Splat meshes without a .RAD tree. Over the network the viewer falls back to whole PLYs."""
    missing = [] if scene.background_lod else ["background"]
    return missing + [o.id for o in scene.objects if o.ply and not o.lod]


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)
    elif isinstance(node, dict):
        for key, value in node.items():
            if key not in SKIP_KEYS:
                yield from _strings(value)


def referenced_files(manifest: dict[str, Any], folder: Path) -> list[str]:
    """Every string in the manifest that names a file inside the scene folder, sorted."""
    root = folder.resolve()
    found = set()
    for text in _strings(manifest):
        if "/" not in text or len(text) > 512:
            continue
        path = (folder / text).resolve()
        if path.is_relative_to(root) and path.is_file():
            found.add(path.relative_to(root).as_posix())
    return sorted(found | {name for name in UNNAMED_FILES if (folder / name).is_file()})
