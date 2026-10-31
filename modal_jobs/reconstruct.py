"""Reconstruction on Modal: frames -> MapAnything (metric) -> COLMAP SfM refinement -> gsplat
-> splat/scene.ply.

    uv run modal run modal_jobs/reconstruct.py --scene room01                  # quality preset
    uv run modal run modal_jobs/reconstruct.py --scene room01 --preset fast    # quick look

Prerequisite: `uv run python scripts/extract_frames.py <video> --scene room01`.
Uploads data/scenes/<scene>/frames to the Modal volume, runs on one L40S, downloads
splat/scene.ply and the gsplat stats back into data/scenes/<scene>/.

MapAnything writes recon_ma/. src/movable_reality/sfm.py refines its cameras with COLMAP and
writes recon/ in the same convention (--no-sfm copies MapAnything's model through).
Trains on the full-resolution frames: recon_full/ holds recon/'s cameras rescaled from
model input resolution (src/movable_reality/recon.py) with images/ linked to frames/.

Presets (31 Oct): "quality" = gsplat's default densification with absolute gradients, 30k
steps, antialiased rasterisation; "mcmc" = MCMC densification up to --cap-max Gaussians (on
blender01 it relocated ~99% of 3M Gaussians every 100 steps and ran out of memory at step
22k); "fast" = gsplat's default strategy for 7k steps. The 7k run on blender01 with
MapAnything's poses reached 25.5 dB PSNR on held-out frames.
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
    # MapAnything's COLMAP export voxel-downsamples with open3d (missing on the first run,
    # 30 Oct). Its own layer, so the gsplat layer above stays cached.
    .pip_install("open3d==0.19.0")
    # Our camera helpers (numpy only), for the full-resolution COLMAP model. Added last so
    # code changes never rebuild the gsplat layer.
    .add_local_python_source("movable_reality", copy=False)
)

# COLMAP SfM (src/movable_reality/sfm.py) needs pycolmap 4; the gsplat image above has 3.13.
sfm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pycolmap==4.2.1", "numpy", "pillow", "plyfile")
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


def _train_args(
    preset: str, max_steps: int | None, cap_max: int, grow_grad2d: float = 0.0003
) -> list[str]:
    """simple_trainer.py arguments for a preset (data and result dirs are added by the caller)."""
    if preset == "quality":  # absolute-gradient densification; 0.0003 gave 763k on blender01
        steps = max_steps or 30_000
        head = ["default", "--strategy.absgrad", "--strategy.grow-grad2d", str(grow_grad2d),
                "--antialiased"]  # fmt: skip
    elif preset == "mcmc":  # on blender01 (31 Oct) 99% of 3M Gaussians died every refine step
        steps = max_steps or 30_000
        head = ["mcmc", "--strategy.cap-max", str(cap_max), "--antialiased"]
    elif preset == "fast":
        steps = max_steps or 7_000
        head = ["default"]
    else:
        raise ValueError(f"unknown preset {preset!r} (quality, mcmc, fast)")
    return [
        *head, "--data_factor", "1", "--max_steps", str(steps), "--eval_steps", str(steps),
        "--ply_steps", str(steps), "--save_steps", str(steps), "--save_ply",
        "--no-normalize-world-space", "--disable_viewer",
    ]  # fmt: skip


def _read(path: Path) -> str | None:
    return path.read_text() if path.exists() else None


def _run(cmd: list[str], cwd: str | None = None) -> float:
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=cwd)
    return round(time.time() - t0, 1)


@app.function(
    gpu=GPU,
    volumes={VOLUME_ROOT: scenes_vol, "/cache": cache_vol},
    timeout=3 * 60 * 60,
)
def reconstruct(
    scene_id: str,
    preset: str = "quality",
    max_steps: int | None = None,
    cap_max: int = 3_000_000,
    force: bool = False,
    intrinsics: dict | None = None,
    sfm: bool = True,
    grow_grad2d: float = 0.0003,
) -> dict:
    scenes_vol.reload()
    root = Path(VOLUME_ROOT) / "scenes" / scene_id
    frames, recon, result = root / "frames", root / "recon", root / "gsplat"
    recon_ma = root / "recon_ma"
    if not frames.is_dir():
        raise FileNotFoundError(f"{frames} missing; run extract_frames.py and upload first")
    timings: dict[str, float] = {}
    fhash = _frames_hash(frames)
    if intrinsics:  # a new camera must redo MapAnything too
        fhash += " " + json.dumps(intrinsics, sort_keys=True)
    # Two stamps: recon_ma/ records the frames and camera MapAnything saw, recon/ adds the
    # SfM switch, so turning SfM on or off redoes recon/ onwards and keeps MapAnything.
    ma_stamp, stamp = recon_ma / "frames_hash.txt", recon / "frames_hash.txt"
    rhash = fhash + (" sfm" if sfm else " no-sfm")
    old = (_read(stamp) or "").strip()
    seen = (_read(ma_stamp) or "").strip() or old.rsplit(" ", 1)[0]  # scenes from before the split
    if not force and seen and seen != fhash:
        print("frames changed since the last reconstruction; redoing every stage")
        force = True
    elif not force and old and old != rhash:
        print("SfM setting changed; redoing the cameras and later stages")
        shutil.rmtree(recon, ignore_errors=True)

    # 1. MapAnything: metric cameras + points with one shared camera, exported as a COLMAP
    # model (Apache weights).
    k_args = [f"--{k}={intrinsics[k]}" for k in ("fx", "fy", "cx", "cy")] if intrinsics else []
    if force or not (recon_ma / "sparse").exists():
        force = True  # new MapAnything cameras invalidate everything after it
        shutil.rmtree(recon_ma, ignore_errors=True)
        timings["mapanything_s"] = _run(
            ["python", "-m", "movable_reality.ma_colmap", f"--images_dir={frames}",
             f"--output_dir={recon_ma}", *k_args]
        )  # fmt: skip
        ma_stamp.write_text(fhash)

    # 1a. COLMAP SfM on the same frames, fitted to MapAnything's metric frame (sfm.py). It also
    # fixes the focal length: MapAnything's export writes its own predicted intrinsics even when
    # given known ones (blender01: fx 1050 px against the true 1387 px).
    if force or not (recon / "sparse").exists():
        force = True  # recon_full/ and the splats were built from the old cameras
        shutil.rmtree(recon, ignore_errors=True)
        if sfm:
            scenes_vol.commit()  # the SfM container reads recon_ma/ from the volume
            t0 = time.time()
            print("sfm:", sfm_refine.remote(scene_id, intrinsics), flush=True)
            timings["sfm_s"] = round(time.time() - t0, 1)
            scenes_vol.reload()
        else:
            shutil.copytree(recon_ma, recon, symlinks=True)
        stamp.write_text(rhash)

    # 1b. MapAnything's model stores ~518 px images and intrinsics. Train on the full-res
    # frames instead: recon_full/ = frames as images/ plus the cameras rescaled to them.
    full = root / "recon_full"
    if force or not (full / "sparse" / "0" / "cameras.bin").exists():
        timings["full_res_s"] = _full_res_model(recon, frames, full)

    # 2. gsplat: keep the metric frame (no world-space normalisation), export PLY. Every 8th
    # frame is held out (gsplat's test_every) and scored at the last step.
    args = _train_args(preset, max_steps, cap_max, grow_grad2d)
    train_stamp = result / "train_args.json"
    if force or not (result / "ply").exists() or _read(train_stamp) != json.dumps(args):
        shutil.rmtree(result, ignore_errors=True)
        timings["gsplat_s"] = _run(
            ["python", "simple_trainer.py", *args, "--data_dir", str(full),
             "--result_dir", str(result)],
            cwd="/opt/gsplat/examples",
        )  # fmt: skip
        train_stamp.write_text(json.dumps(args))

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
def main(
    scene: str,
    preset: str = "quality",
    max_steps: int = 0,
    cap_max: int = 3_000_000,
    force: bool = False,
    skip_upload: bool = False,
    no_sfm: bool = False,
    eval_walk: str = "",
    grow_grad2d: float = 0.0003,
    objects: bool = False,
    obj_steps: int = 4000,
    bg_steps: int = 8000,
):
    if eval_walk:  # only render and score eval/views_<walk>.json (scripts/blender_view_eval.py)
        eval_views(scene, eval_walk)
        return
    if objects:  # only the per-object training (src/movable_reality/objtrain.py)
        train_objects_local(scene, obj_steps, bg_steps)
        return
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

    # Known camera intrinsics in full-resolution pixels, {"fx", "fy", "cx", "cy"}; without the
    # file MapAnything self-calibrates one shared camera (src/movable_reality/ma_colmap.py).
    kfile = local / "intrinsics.json"
    intrinsics = json.loads(kfile.read_text()) if kfile.exists() else None
    out = reconstruct.remote(
        scene, preset=preset, max_steps=max_steps or None, cap_max=cap_max, force=force,
        intrinsics=intrinsics, sfm=not no_sfm, grow_grad2d=grow_grad2d,
    )  # fmt: skip
    print("remote result:", out)

    (local / "splat").mkdir(parents=True, exist_ok=True)
    for name in ("scene.ply", "timings.json"):
        with open(local / "splat" / name, "wb") as f:
            for chunk in scenes_vol.read_file(f"/scenes/{scene}/splat/{name}"):
                f.write(chunk)
    # Cameras: the refined model (recon/), its full-res copy and MapAnything's own (recon_ma/).
    for model in ("recon", "recon_full", "recon_ma"):
        for entry in scenes_vol.listdir(f"/scenes/{scene}/{model}", recursive=True):
            rel = entry.path.lstrip("/").removeprefix(f"scenes/{scene}/")
            if not rel.endswith((".bin", ".txt", ".ply", ".json")) or "/images/" in rel:
                continue
            if "/colmap/" in rel:  # SfM's working model, already written to recon/sparse
                continue
            dest = local / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in scenes_vol.read_file(entry.path):
                    f.write(chunk)
    stats_dir = local / "gsplat_stats"  # PSNR/SSIM/LPIPS, written at gsplat eval steps
    shutil.rmtree(stats_dir, ignore_errors=True)  # stats of an older run would mislead
    stats_dir.mkdir()
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


@app.function(image=sfm_image, cpu=8.0, memory=16384, volumes={VOLUME_ROOT: scenes_vol},
              timeout=60 * 60)  # fmt: skip
def sfm_refine(scene_id: str, intrinsics: dict | None) -> dict:
    """COLMAP SfM on the frames, fitted to MapAnything's cameras: recon_ma/ -> recon/."""
    import numpy as np  # noqa: PLC0415

    from movable_reality.sfm import refine  # noqa: PLC0415

    scenes_vol.reload()
    root = Path(VOLUME_ROOT) / "scenes" / scene_id
    K = None
    if intrinsics:
        k = intrinsics
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
    report = refine(root / "frames", root / "recon_ma", root / "recon", K)
    scenes_vol.commit()
    return report


@app.function(gpu=GPU, volumes={VOLUME_ROOT: scenes_vol, "/cache": cache_vol}, timeout=30 * 60)
def render_eval(scene_id: str, walk: str, n_side: int = 8) -> dict:
    """Render the last gsplat checkpoint from eval/views_<walk>.json and score it against the
    uploaded GT images with the metrics gsplat's own eval uses (LPIPS on AlexNet)."""
    import math  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from gsplat import rasterization  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from torchmetrics.image import (  # noqa: PLC0415
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: PLC0415

    scenes_vol.reload()
    root = Path(VOLUME_ROOT) / "scenes" / scene_id
    spec = json.loads((root / "eval" / f"views_{walk}.json").read_text())
    ckpts = sorted(
        (root / "gsplat" / "ckpts").glob("ckpt_*_rank0.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    if not ckpts:
        raise FileNotFoundError(f"no gsplat checkpoint for {scene_id}; reconstruct it first")
    sp = torch.load(ckpts[-1], map_location="cuda", weights_only=False)["splats"]
    colors = torch.cat([sp["sh0"], sp["shN"]], 1)
    sh_degree = round(math.sqrt(colors.shape[1])) - 1
    aa = "--antialiased" in json.loads((root / "gsplat" / "train_args.json").read_text())
    psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    out_dir = root / "eval" / f"views_{walk}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    W, H = spec["width"], spec["height"]
    per = []
    with torch.no_grad():
        for i, v in enumerate(spec["views"]):
            img, _, _ = rasterization(
                sp["means"], sp["quats"], torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"]),
                colors, torch.tensor(v["w2c"], device="cuda")[None].float(),
                torch.tensor(v["K"], device="cuda")[None].float(), W, H, sh_degree=sh_degree,
                rasterize_mode="antialiased" if aa else "classic",
            )  # fmt: skip
            pred = img[0].clamp(0, 1).permute(2, 0, 1)[None]
            gt_np = np.asarray(Image.open(root / "eval_gt" / walk / v["image"]).convert("RGB"))
            gt = torch.tensor(gt_np, device="cuda").float().div(255).permute(2, 0, 1)[None]
            scores = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "lpips": lpips(pred, gt)}
            per.append({"name": v["name"], **{k: float(x) for k, x in scores.items()}})
            if i % max(1, len(spec["views"]) // n_side) == 0:
                side = np.concatenate([gt_np, (pred[0].permute(1, 2, 0).cpu().numpy() * 255)
                                       .round().astype(np.uint8)], axis=1)  # fmt: skip
                Image.fromarray(side).save(out_dir / f"{v['name']}.jpg", quality=90)
    summary = {
        "walk": walk,
        "views": len(per),
        "checkpoint": ckpts[-1].name,
        "num_gaussians": int(sp["means"].shape[0]),
        **{f"{k}_mean": float(np.mean([p[k] for p in per])) for k in ("psnr", "ssim", "lpips")},
        "psnr_median": float(np.median([p["psnr"] for p in per])),
        "alignment": spec["alignment"],
        "per_view": per,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1))
    scenes_vol.commit()
    return {k: v for k, v in summary.items() if k != "per_view"}


def eval_views(scene: str, walk: str = "test"):
    """Upload eval/views_<walk>.json (scripts/blender_view_eval.py) and its GT images, render
    and score on Modal, download metrics.json and side-by-side images (GT left, render right)."""
    data = Path(__file__).resolve().parents[1] / "data"
    local = data / "scenes" / scene
    spec_path = local / "eval" / f"views_{walk}.json"
    spec = json.loads(spec_path.read_text())
    rgb = data / spec["rgb_dir"]
    with scenes_vol.batch_upload(force=True) as batch:
        batch.put_file(spec_path, f"/scenes/{scene}/eval/{spec_path.name}")
        for v in spec["views"]:
            batch.put_file(rgb / v["image"], f"/scenes/{scene}/eval_gt/{walk}/{v['image']}")
    print(json.dumps(render_eval.remote(scene, walk), indent=1))
    dest = local / "eval" / f"views_{walk}"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for entry in scenes_vol.listdir(f"/scenes/{scene}/eval/views_{walk}"):
        with open(dest / Path(entry.path).name, "wb") as f:
            for chunk in scenes_vol.read_file(entry.path):
                f.write(chunk)
    print(f"downloaded to {dest}")


@app.function(gpu=GPU, volumes={VOLUME_ROOT: scenes_vol}, timeout=2 * 60 * 60)
def train_objects(scene_id: str, obj_steps: int, bg_steps: int) -> dict:
    """Every object and the background fine-tuned on their own pixels (objtrain.py)."""
    scenes_vol.reload()
    root = Path(VOLUME_ROOT) / "scenes" / scene_id
    out = root / "objects_trained"
    shutil.rmtree(out, ignore_errors=True)
    seconds = _run(["python", "-m", "movable_reality.objtrain", f"--scene_dir={root}",
                    f"--obj_steps={obj_steps}", f"--bg_steps={bg_steps}"])  # fmt: skip
    scenes_vol.commit()
    return {"seconds": seconds, **json.loads((out / "report.json").read_text())}


def train_objects_local(scene: str, obj_steps: int, bg_steps: int):
    """Upload the lift outputs, train on Modal, download objects_trained/ and install it."""
    from movable_reality.objtrain import install  # noqa: PLC0415

    local = Path(__file__).resolve().parents[1] / "data" / "scenes" / scene
    # Always lift from the joint split: a second run must not start from trained PLYs.
    src = local / "objects_lifted" if (local / "objects_lifted").exists() else local / "objects"
    try:
        scenes_vol.remove_file(f"/scenes/{scene}/ids", recursive=True)
    except Exception:  # noqa: BLE001 - nothing uploaded yet
        pass
    with scenes_vol.batch_upload(force=True) as batch:
        batch.put_directory(local / "ids" / "masks", f"/scenes/{scene}/ids/masks")
        batch.put_file(local / "ids" / "object_labels.json",
                       f"/scenes/{scene}/ids/object_labels.json")  # fmt: skip
        batch.put_directory(src, f"/scenes/{scene}/objects")
        batch.put_file(local / "scene.json", f"/scenes/{scene}/scene.json")
    print("uploaded masks, objects and scene.json", flush=True)
    report = train_objects.remote(scene, obj_steps, bg_steps)
    print(json.dumps(report, indent=1))
    dest = local / "objects_trained"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for entry in scenes_vol.listdir(f"/scenes/{scene}/objects_trained"):
        with open(dest / Path(entry.path).name, "wb") as f:
            for chunk in scenes_vol.read_file(entry.path):
                f.write(chunk)
    print("installed:", install(local))
