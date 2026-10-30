"""Build Spark LOD trees (.RAD) for a scene with Spark's `build-lod` tool.

`build-lod` is a Rust program in Spark's repository (rust/build-lod, MIT). The npm package
ships no binary, so `ensure_build_lod` clones Spark at the tag that matches web/package.json
into a cache folder outside the repo and runs `cargo build --release`. The CPU-only build
(`--no-default-features`) is used because the default `gpu` feature only speeds up
optional SH clustering and pulls in wgpu. Nothing from Spark is copied into this repo.

`build-lod` writes `<stem>-lod.rad` next to its input. We build in a temporary folder so
input folders stay clean, then move the result to `lod/<stem>.rad` in the scene folder.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from movable_reality.config import REPO_ROOT
from movable_reality.lod.rad import read_rad_header
from movable_reality.scene import Scene

SPARK_REPO = "https://github.com/sparkjsdev/spark"
DEFAULT_TOOLS_DIR = Path.home() / ".cache" / "movable-reality" / "tools"
Method = Literal["quality", "quick"]


def spark_version(package_json: Path = REPO_ROOT / "web" / "package.json") -> str:
    """Spark version pinned in the viewer, so the LOD tool matches the runtime."""
    deps = json.loads(package_json.read_text(encoding="utf-8"))["dependencies"]
    return str(deps["@sparkjsdev/spark"]).lstrip("^~=")


def find_build_lod(version: str | None = None, tools_dir: Path | None = None) -> Path | None:
    """Path of a usable build-lod binary: $SPARK_BUILD_LOD, then the cache, then PATH."""
    env = os.environ.get("SPARK_BUILD_LOD")
    if env:
        return Path(env) if Path(env).is_file() else None
    version = version or spark_version()
    cached = _cached_binary(version, tools_dir or DEFAULT_TOOLS_DIR)
    if cached.is_file():
        return cached
    on_path = shutil.which("build-lod")
    return Path(on_path) if on_path else None


def _cached_binary(version: str, tools_dir: Path) -> Path:
    exe = "build-lod.exe" if os.name == "nt" else "build-lod"
    return tools_dir / f"spark-v{version}" / "rust" / "target" / "release" / exe


def ensure_build_lod(
    version: str | None = None, tools_dir: Path | None = None, log: Callable[[str], None] = print
) -> Path:
    """Find build-lod, or clone Spark at tag v<version> and build it (needs git and cargo)."""
    version = version or spark_version()
    tools_dir = tools_dir or DEFAULT_TOOLS_DIR
    found = find_build_lod(version, tools_dir)
    if found:
        return found
    for tool in ("git", "cargo"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"build-lod not found and `{tool}` is missing. Install Rust (https://rustup.rs) "
                "and git, or set SPARK_BUILD_LOD to a build-lod binary."
            )
    src = tools_dir / f"spark-v{version}"
    if not (src / "rust" / "Cargo.toml").is_file():
        tools_dir.mkdir(parents=True, exist_ok=True)
        log(f"cloning Spark v{version} into {src}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", f"v{version}", SPARK_REPO, str(src)],
            check=True,
        )
    log("building build-lod (cargo, release, CPU only; about 2 min the first time)")
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "-p", "build-lod", "--no-default-features"],
        cwd=src / "rust",
        check=True,
    )
    binary = _cached_binary(version, tools_dir)
    if not binary.is_file():
        raise RuntimeError(f"cargo finished but {binary} is missing")
    return binary


def build_rad(
    binary: Path,
    ply: Path,
    out: Path,
    method: Method = "quality",
    max_sh: int | None = None,
    log_file: Path | None = None,
) -> None:
    """Run build-lod on one PLY and write the tree to `out` (a .rad path)."""
    args = [f"--{method}"]
    if max_sh is not None:
        args.append(f"--max-sh={max_sh}")
    with tempfile.TemporaryDirectory(prefix="build-lod-") as tmp:
        tmp_ply = Path(tmp) / "input.ply"
        try:
            os.link(ply, tmp_ply)
        except OSError:
            shutil.copyfile(ply, tmp_ply)
        proc = subprocess.run(
            [str(binary), *args, str(tmp_ply)], capture_output=True, text=True, check=False
        )
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        produced = Path(tmp) / "input-lod.rad"
        # build-lod exits 0 without writing a file when decoding or validation fails.
        if proc.returncode != 0 or not produced.is_file():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            raise RuntimeError(f"build-lod failed on {ply}: " + " | ".join(tail))
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), out)


def rad_name(ply_rel: str) -> str:
    """lod/<stem>.rad for a PLY path relative to the scene folder."""
    return f"lod/{Path(ply_rel).stem}.rad"


@dataclass
class BuildResult:
    ply: str
    rad: str
    status: Literal["built", "skipped", "empty", "missing"]
    input_splats: int | None = None
    tree_splats: int | None = None


def _ply_vertex_count(path: Path) -> int:
    """Vertex count from the PLY header without reading the body."""
    with path.open("rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"{path}: no vertex element in PLY header")


def build_scene_lod(
    scene_path: Path,
    binary: Path,
    method: Method = "quality",
    max_sh: int | None = None,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> list[BuildResult]:
    """Build one .RAD per object PLY and one for the background; record paths in scene.json.

    Idempotent: an existing .RAD newer than its PLY is reused unless `force`; one older than
    its PLY (object re-split or completed) is rebuilt. The manifest is reloaded right
    before saving, so edits made while the trees were building (agent moves) are kept.
    """
    root = scene_path.parent
    scene = Scene.load(scene_path)
    jobs: list[tuple[str | None, str]] = [(None, scene.background_ply)]
    jobs += [(o.id, o.ply) for o in scene.objects]
    results: list[BuildResult] = []
    paths: dict[str | None, str | None] = {}
    for object_id, ply_rel in jobs:
        rad_rel = rad_name(ply_rel)
        ply, rad = root / ply_rel, root / rad_rel
        name = object_id or "background"
        if not ply.is_file():
            log(f"{name}: {ply_rel} missing, no LOD")
            results.append(BuildResult(ply_rel, rad_rel, "missing"))
            paths[object_id] = None
            continue
        n = _ply_vertex_count(ply)
        if n == 0:  # build-lod panics on empty input
            log(f"{name}: {ply_rel} has no splats, no LOD")
            results.append(BuildResult(ply_rel, rad_rel, "empty", 0))
            paths[object_id] = None
            continue
        fresh = rad.is_file() and rad.stat().st_mtime >= ply.stat().st_mtime
        if fresh and not force:
            status: Literal["built", "skipped"] = "skipped"
        else:
            build_rad(
                binary, ply, rad, method, max_sh, log_file=root / "lod" / "logs" / f"{rad.stem}.log"
            )
            status = "built"
        header = read_rad_header(rad)
        if not header.lod_tree:
            raise RuntimeError(f"{rad} has no LOD tree")
        log(f"{name}: {status} {rad_rel} ({n} splats -> {header.count} in tree)")
        results.append(BuildResult(ply_rel, rad_rel, status, n, header.count))
        paths[object_id] = rad_rel

    scene = Scene.load(scene_path)
    scene.background_lod = paths.get(None)
    for obj in scene.objects:
        if obj.id in paths:
            obj.lod = paths[obj.id]
    scene.save(scene_path)
    return results
