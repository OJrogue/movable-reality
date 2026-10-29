"""Reconstruction on Modal: frames -> MapAnything (metric COLMAP) -> gsplat -> splat/scene.ply.

    uv run modal run modal_jobs/reconstruct.py --scene room01
    uv run modal run modal_jobs/reconstruct.py --scene room01 --max-steps 7000   # quick look

Prerequisite: `uv run python scripts/extract_frames.py <video> --scene room01`.
Uploads data/scenes/<scene>/frames to the Modal volume, runs on one L40S, downloads
splat/scene.ply and the gsplat stats back into data/scenes/<scene>/.

STATUS: first version, not yet run. The image build is the most likely thing to need fixes
(torch / CUDA / gsplat example requirements). Fix it here, then log the fix in BUILD_LOG.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "movable-reality-recon"
GPU = "L40S"
VOLUME_ROOT = "/data"  # mount point of the scenes volume in the container
CUDA_ARCHS = "8.0;8.6;8.9;9.0"  # A100, A10/3090, L40S/4090, H100

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.9.1", "torchvision==0.24.1", index_url="https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/map-anything /opt/map-anything",
        "pip install -e '/opt/map-anything[colmap]'",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/nerfstudio-project/gsplat /opt/gsplat",
        "cd /opt/gsplat && pip install .",  # compiles CUDA kernels for CUDA_ARCHS
        "pip install -r /opt/gsplat/examples/requirements.txt",
        env={"TORCH_CUDA_ARCH_LIST": CUDA_ARCHS, "MAX_JOBS": "8"},
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": "/cache/hf"})
)

app = modal.App(APP_NAME, image=image)
scenes_vol = modal.Volume.from_name("movable-reality-scenes", create_if_missing=True)
cache_vol = modal.Volume.from_name("movable-reality-cache", create_if_missing=True)


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
        # TODO(day 2): replace recon/images (model input resolution, ~518 px) with full-res
        # frames and rescale intrinsics in sparse/cameras.bin before training.

    # 2. gsplat: keep the metric frame (no world-space normalisation), export PLY.
    if force or not (result / "ply").exists():
        shutil.rmtree(result, ignore_errors=True)
        timings["gsplat_s"] = _run(
            [
                "python",
                "simple_trainer.py",
                "default",
                "--data_dir",
                str(recon),
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
