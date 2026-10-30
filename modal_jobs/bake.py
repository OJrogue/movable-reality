"""Plane bake with gsplat's orthographic rasterizer on Modal (GPU path of stage 12).

    uv run modal run modal_jobs/bake.py --scene room01
    uv run modal run modal_jobs/bake.py --scene room01 --force --texel 0.004

Same stage as scripts/bake_planes.py: the local entrypoint runs the CPU pipeline (plane fit,
rectangle, hole filling, GLB and PLY export, scene.json) on the local scene folder and sends
only the rasterisation to the GPU, as arrays. No files go through a Modal volume. Use it when
the CPU bake is too slow (texture sizes near the 4096 px cap, millions of plane splats).

gsplat 1.5.3 (pinned; source read 29 Oct): rasterization(..., camera_model="ortho") projects
mean2d = (fx * x + cx, fy * y + cy) and cov2d = J cov J^T with J = [[fx, 0, 0], [0, fy, 0]]
(gsplat/cuda/include/Utils.cuh, ortho_proj), samples pixel centres at (j + 0.5, i + 0.5),
classic mode clamps alpha at 0.999, drops alpha < 1/255 and stops at transmittance 1e-4,
and with backgrounds=None returns the premultiplied colour sum. `raster.ortho_camera` builds
viewmat and K so this equals the CPU reference (tests/test_bake_raster.py checks the camera).

STATUS: written, not yet run (no Modal credentials in the session that wrote it). The image
build compiles gsplat's CUDA kernels; that step is the most likely to need a fix.
"""

from __future__ import annotations

import modal

APP_NAME = "movable-reality-bake"
GPU = "L4"  # a bake is seconds of GPU time; the smallest GPU with enough memory is fine
GSPLAT_VERSION = "1.5.3"
CUDA_ARCHS = "8.0;8.6;8.9;9.0"  # A100, A10/3090, L4/L40S/4090, H100

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch==2.9.1", "numpy>=2.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(f"gsplat=={GSPLAT_VERSION}")
    # The PyPI wheel compiles its CUDA kernels on first import (torch JIT). Do it once in the
    # image so containers start warm; TORCH_CUDA_ARCH_LIST avoids needing a GPU at build time.
    .run_commands(
        "python -c 'from gsplat.cuda._backend import _C; assert _C is not None'",
        env={"TORCH_CUDA_ARCH_LIST": CUDA_ARCHS, "MAX_JOBS": "8"},
    )
    .env({"TORCH_CUDA_ARCH_LIST": CUDA_ARCHS})
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu=GPU, timeout=20 * 60)
def rasterize_ortho(
    means,  # (N, 3) float32, world
    covs,  # (N, 3, 3) float32, world
    opacities,  # (N,) float32
    rgb,  # (N, 3) float32, degree-0 colour
    viewmat,  # (4, 4) world-to-camera
    k,  # (3, 3)
    width: int,
    height: int,
):
    """Returns (premultiplied colour (H, W, 3), alpha (H, W)) as float16 numpy arrays."""
    import numpy as np
    import torch
    from gsplat import rasterization

    dev = torch.device("cuda")

    def t(a):
        return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=dev)

    n = len(means)
    colors, alphas, _ = rasterization(
        t(means),
        None,  # quats: ignored when covars is given
        None,  # scales
        t(opacities),
        t(rgb),
        t(viewmat)[None],
        t(k)[None],
        width,
        height,
        near_plane=0.01,
        far_plane=1e4,
        eps2d=0.3,
        sh_degree=None,
        packed=False,
        backgrounds=None,
        render_mode="RGB",
        rasterize_mode="classic",
        camera_model="ortho",
        covars=t(covs),
    )
    print(f"rasterized {n} Gaussians into {width} x {height}")
    premult = colors[0].float().cpu().numpy().astype(np.float16)
    alpha = alphas[0, ..., 0].float().cpu().numpy().astype(np.float16)
    return premult, alpha


def gpu_raster(means, covs, opacities, rgb, view):
    """RasterFn for movable_reality.bake.pipeline.bake_scene that runs on Modal."""
    import numpy as np

    from movable_reality.bake import raster

    # Camera 1 m in front of the plane; baked Gaussians lie within the bake band (cm), so
    # every depth is well inside (near_plane, far_plane).
    viewmat, k = raster.ortho_camera(view, height=1.0)
    premult, alpha = rasterize_ortho.remote(
        np.asarray(means, np.float32),
        np.asarray(covs, np.float32),
        np.asarray(opacities, np.float32),
        np.asarray(rgb, np.float32),
        viewmat.astype(np.float32),
        k.astype(np.float32),
        int(view.width),
        int(view.height),
    )
    return premult.astype(np.float64), alpha.astype(np.float64)


@app.local_entrypoint()
def main(
    scene: str,
    force: bool = False,
    labels: str = "floor,ceiling",
    texel: float = 0.005,
    max_px: int = 4096,
    threshold: float = 0.03,
    no_mask: bool = False,
):
    from movable_reality.bake.pipeline import BakeOptions, bake_scene
    from movable_reality.config import scene_dir

    opts = BakeOptions(
        labels=tuple(s.strip() for s in labels.split(",") if s.strip()),
        texel=texel,
        max_px=max_px,
        threshold=threshold,
        alpha_mask=not no_mask,
    )
    report = bake_scene(scene_dir(scene), opts, force=force, raster_fn=gpu_raster)
    print(f"{scene}: {report.status}, removed {report.removed_splats} splats")
