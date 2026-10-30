"""Reconstruction on Modal: frames -> MapAnything (metric COLMAP) -> gsplat -> splat/scene.ply.

    uv run modal run modal_jobs/reconstruct.py --scene room01
    uv run modal run modal_jobs/reconstruct.py --scene room01 --max-steps 7000   # quick look

Prerequisite: `uv run python scripts/extract_frames.py <video> --scene room01`.
Uploads data/scenes/<scene>/frames to the Modal volume, runs on one L40S, downloads
splat/scene.ply and the gsplat stats back into data/scenes/<scene>/.

Trains on the full-resolution frames: recon_full/ holds MapAnything's cameras rescaled from
model input resolution (src/movable_reality/recon.py) with images/ linked to frames/.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "movable-reality-recon"
GPU = "L40S"
VOLUME_ROOT = "/data"  # mount point of the scenes volume in the container
# Only the GPU the job runs on: each extra architecture multiplies the gsplat compile time
# (four of them took over 2 h on the CPU builder, 30 Oct). Change both together.
CUDA_ARCHS = "8.9"  # L40S
# Pinned upstream commits (main on 29 Oct 2026), so a rebuild of the image gets the same code.
MAPANYTHING_SHA = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
GSPLAT_SHA = "512d366b67073d77ca099ede742683c165dfc23b"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.9.1", "torchvision==0.24.1", index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install("setuptools", "wheel", "ninja", "numpy")
    .run_commands(
        "git clone https://github.com/facebookresearch/map-anything /opt/map-anything",
        f"git -C /opt/map-anything checkout {MAPANYTHING_SHA}",
        "pip install -e '/opt/map-anything[colmap]'",
    )
    .run_commands(
        # gsplat's CUDA build needs the glm submodule. --no-build-isolation builds against
        # the torch installed above; with isolation pip fetches the newest torch from PyPI
        # (pyproject asks for torch>=2.7) and the extension then fails to import.
        "git clone https://github.com/nerfstudio-project/gsplat /opt/gsplat",
        f"git -C /opt/gsplat checkout {GSPLAT_SHA}",
        "git -C /opt/gsplat submodule update --init --recursive",
        "nproc && python -c 'import sysconfig; print(sysconfig.get_config_var(\"LDCXXSHARED\"))'",
        # -v streams nvcc progress into the build log (a quiet pip looks like a hang).
        "cd /opt/gsplat && pip install -v --no-build-isolation .",  # CUDA kernels for CUDA_ARCHS
        "python -c 'from gsplat.cuda._backend import _C; assert _C is not None'",
        "pip install --no-build-isolation -r /opt/gsplat/examples/requirements.txt",
        # add_python's interpreter was built with clang, so setuptools would link with clang++,
        # which the CUDA image lacks (the link failed after all 50 objects, 30 Oct). Use g++.
        env={
            "TORCH_CUDA_ARCH_LIST": CUDA_ARCHS,
            "MAX_JOBS": "8",
            "CC": "gcc",
            "CXX": "g++",
            "LDSHARED": "gcc -shared",
            "LDCXXSHARED": "g++ -shared",
        },
        # The GPU builder still had 1 CPU (nproc, 30 Oct), so gpu= does not speed the compile.
        # It stays only because removing it rebuilds this cached layer (41 min). When the layer
        # is rebuilt anyway, drop gpu= and check `import gsplat.csrc` instead of _C, since
        # gsplat's _backend can fall back to a JIT compile and hide a broken wheel.
        gpu=GPU,
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/cache/hf"})
    # Our camera helpers (numpy only), for the full-resolution COLMAP model. Added last so
    # code changes never rebuild the gsplat layer.
    .add_local_python_source("movable_reality", copy=False)
)

app = modal.App(APP_NAME, image=image)
scenes_vol = modal.Volume.from_name("movable-reality-scenes", create_if_missing=True)
cache_vol = modal.Volume.from_name("movable-reality-cache", create_if_missing=True)


def _frames_hash(frames: Path) -> str:
    """Hash of the frame names and sizes: a re-extracted video gives a new hash."""
    h = hashlib.sha256()
    for f in sorted(frames.glob("*.jpg")):
        h.update(f"{f.name}:{f.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def _full_res_model(recon: Path, frames: Path, full: Path) -> float:
    from PIL import Image  # noqa: PLC0415 - container only

    from movable_reality.coverage.colmap import find_sparse_dir, read_model, write_model
    from movable_reality.recon import full_res_camera

    t0 = time.time()
    shutil.rmtree(full, ignore_errors=True)
    sparse = full / "sparse" / "0"
    cams = read_model(recon)
    w, h = Image.open(frames / cams[0].name).size
    write_model(sparse, [full_res_camera(c, w, h) for c in cams])
    shutil.copyfile(find_sparse_dir(recon) / "points3D.bin", sparse / "points3D.bin")
    (full / "images").symlink_to(frames)
    print(f"full-res model: {len(cams)} cameras at {w}x{h}")
    return round(time.time() - t0, 1)


def _run(cmd: list[str], cwd: str | None = None) -> float:
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=cwd)
    return round(time.time() - t0, 1)


@app.function(
    gpu=GPU,
    volumes={VOLUME_ROOT: scenes_vol, "/cache": cache_vol},
    timeout=60 * 60,
)
def reconstruct(scene_id: str, max_steps: int = 15_000, force: bool = False) -> dict:
    scenes_vol.reload()
    root = Path(VOLUME_ROOT) / "scenes" / scene_id
    frames, recon, result = root / "frames", root / "recon", root / "gsplat"
    if not frames.is_dir():
        raise FileNotFoundError(f"{frames} missing; run extract_frames.py and upload first")
    timings: dict[str, float] = {}
    fhash = _frames_hash(frames)
    stamp = recon / "frames_hash.txt"
    if not force and stamp.exists() and stamp.read_text().strip() != fhash:
        print("frames changed since the last reconstruction; redoing every stage")
        force = True

    # 1. MapAnything: metric cameras + points, exported as a COLMAP model (Apache weights).
    if force or not (recon / "sparse").exists():
        shutil.rmtree(recon, ignore_errors=True)
        timings["mapanything_s"] = _run(
            [
                "python",
                "/opt/map-anything/scripts/demo_colmap.py",
                f"--images_dir={frames}",
                f"--output_dir={recon}",
                "--apache",
            ]
        )
        stamp.write_text(fhash)

    # 1b. MapAnything's model stores ~518 px images and intrinsics. Train on the full-res
    # frames instead: recon_full/ = frames as images/ plus the cameras rescaled to them.
    full = root / "recon_full"
    if force or not (full / "sparse" / "0" / "cameras.bin").exists():
        timings["full_res_s"] = _full_res_model(recon, frames, full)

    # 2. gsplat: keep the metric frame (no world-space normalisation), export PLY.
    if force or not (result / "ply").exists():
        shutil.rmtree(result, ignore_errors=True)
        timings["gsplat_s"] = _run(
            [
                "python",
                "simple_trainer.py",
                "default",
                "--data_dir",
                str(full),
                "--data_factor",
                "1",
                "--result_dir",
                str(result),
                "--max_steps",
                str(max_steps),
                "--save_ply",
                "--no-normalize-world-space",
                "--disable_viewer",
            ],
            cwd="/opt/gsplat/examples",
        )

    plys = sorted(
        (result / "ply").glob("point_cloud_*.ply"), key=lambda p: int(p.stem.split("_")[-1])
    )
    if not plys:
        raise RuntimeError(f"no PLY written under {result / 'ply'}")
    (root / "splat").mkdir(exist_ok=True)
    shutil.copyfile(plys[-1], root / "splat" / "scene.ply")
    (root / "splat" / "timings.json").write_text(json.dumps(timings, indent=2))
    scenes_vol.commit()
    return {"ply": plys[-1].name, **timings}


@app.local_entrypoint()
def main(scene: str, max_steps: int = 15_000, force: bool = False, skip_upload: bool = False):
    local = Path(__file__).resolve().parents[1] / "data" / "scenes" / scene
    if not skip_upload:
        if not (local / "frames").is_dir():
            raise SystemExit(f"{local / 'frames'} missing; run scripts/extract_frames.py first")
        try:  # drop frames from an earlier, longer extraction
            scenes_vol.remove_file(f"/scenes/{scene}/frames", recursive=True)
        except Exception:  # noqa: BLE001 - nothing uploaded yet
            pass
        with scenes_vol.batch_upload(force=True) as batch:
            batch.put_directory(local / "frames", f"/scenes/{scene}/frames")
        print(f"uploaded {local / 'frames'}")

    out = reconstruct.remote(scene, max_steps=max_steps, force=force)
    print("remote result:", out)

    (local / "splat").mkdir(parents=True, exist_ok=True)
    for name in ("scene.ply", "timings.json"):
        with open(local / "splat" / name, "wb") as f:
            for chunk in scenes_vol.read_file(f"/scenes/{scene}/splat/{name}"):
                f.write(chunk)
    # Cameras: MapAnything's model (recon/) and the full-res copy, for lift_ids and coverage.
    for model in ("recon", "recon_full"):
        for entry in scenes_vol.listdir(f"/scenes/{scene}/{model}", recursive=True):
            rel = entry.path.split(f"/scenes/{scene}/", 1)[-1]
            if not rel.endswith((".bin", ".txt", ".ply")) or "/images/" in rel:
                continue
            dest = local / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in scenes_vol.read_file(entry.path):
                    f.write(chunk)
    stats_dir = local / "gsplat_stats"  # PSNR/SSIM/LPIPS, written at gsplat eval steps
    stats_dir.mkdir(exist_ok=True)
    try:
        entries = scenes_vol.listdir(f"/scenes/{scene}/gsplat/stats")
    except Exception as exc:  # noqa: BLE001 - stats are optional (no eval step reached)
        print(f"no gsplat stats to download ({exc})")
        entries = []
    for entry in entries:
        with open(stats_dir / Path(entry.path).name, "wb") as f:
            for chunk in scenes_vol.read_file(entry.path):
                f.write(chunk)
    print(f"downloaded to {local / 'splat'}")
