"""Per-Gaussian photometric uncertainty on Modal (U-3DGS-style, post hoc, scene frozen).

    uv run modal run modal_jobs/coverage.py --scene room01
    uv run modal run modal_jobs/coverage.py --scene room01 --iters 3000 --lam 0.2

Inputs, uploaded from data/scenes/<scene>/: splat/scene.ply (gsplat result, recon frame),
recon/ (COLMAP model) and frames/ (the training images; matched to COLMAP image names, or
recon/images/ if frames/ is missing). Output: coverage/uncertainty.npz, aligned with
splat/scene.ply: `coeffs` (N, 16) float16 SH coefficients of u_k, `u_mean` and `u_max`
(N,) float32, plus coverage/uncertainty.json (timings, loss curve, settings).
scripts/score_coverage.py folds `u_mean` into the coverage score when the file exists.

Method (src/movable_reality/coverage/uncertainty.py has the maths and the CPU tests): the
trained Gaussians are frozen; a per-Gaussian SH uncertainty channel (degree 3) is rendered
with gsplat's own alpha blending, U = sum_k u_k(d) alpha_k T_k, and fitted by Adam to the
per-pixel residual L = 0.8 L1 + 0.2 DSSIM of the frozen render against the training
image, with the prior lambda_reg * sum_k int_S2 (b - u_k)^2 (b = 1) computed in closed form.
The colour pass runs once per view under no_grad; the uncertainty pass renders a
single-channel "colour" from per-Gaussian values, so A = alpha T stays fixed and the
problem stays linear. gsplat's SH path clamps colours to >= 0 after adding 0.5, which would
make it non-linear, so u_k(d) is evaluated here (d = mean - camera centre, as in gsplat's
SH colours) and passed in as a post-activation colour.

STATUS: written 29 Oct, not yet run (no Modal credentials in the coding session). The CPU
self-check `check_against_numpy()` compares the torch SH basis, residual and regulariser
with the numpy reference at start-up; a mismatch aborts the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "movable-reality-coverage"
GPU = "L40S"
VOLUME_ROOT = "/data"
GSPLAT_VERSION = "1.5.3"  # rasterization() API checked against the v1.5.3 source
CUDA_ARCHS = "8.0;8.6;8.9;9.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.1", "torchvision==0.24.1", index_url="https://download.pytorch.org/whl/cu128"
    )
    .pip_install("numpy>=2.0", "pillow>=11.0", "plyfile>=1.1", "pydantic>=2.8", "python-dotenv")
    .run_commands(
        f"pip install --no-build-isolation 'gsplat=={GSPLAT_VERSION}'",
        env={"TORCH_CUDA_ARCH_LIST": CUDA_ARCHS, "MAX_JOBS": "8"},
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("movable_reality")
)

app = modal.App(APP_NAME, image=image)
scenes_vol = modal.Volume.from_name("movable-reality-scenes", create_if_missing=True)


# ---------------------------------------------------------------- torch mirrors of the maths


def sh_basis_torch(dirs, degree: int = 3):
    """Torch copy of uncertainty.sh_basis (same constants and order)."""
    import torch

    from movable_reality.coverage.uncertainty import C0, C1, C2, C3

    x, y, z = dirs.unbind(-1)
    out = [torch.full_like(x, C0)]
    if degree >= 1:
        out += [-C1 * y, C1 * z, -C1 * x]
    if degree >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        out += [C2[0] * xy, C2[1] * yz, C2[2] * (2 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]
    if degree >= 3:
        out += [
            C3[0] * y * (3 * xx - yy),
            C3[1] * xy * z,
            C3[2] * y * (4 * zz - xx - yy),
            C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
            C3[4] * x * (4 * zz - xx - yy),
            C3[5] * z * (xx - yy),
            C3[6] * x * (xx - 3 * yy),
        ]
    return torch.stack(out, dim=-1)


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None):
    import torch

    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def pixel_residual_torch(render, target, lam: float = 0.2):
    """Torch copy of uncertainty.pixel_residual; images (H, W, 3) in [0, 1] -> (H, W)."""
    import torch.nn.functional as F

    k = _gauss_window(device=render.device)
    a = render.permute(2, 0, 1)[:, None]  # (3, 1, H, W)
    b = target.permute(2, 0, 1)[:, None]

    def filt(img):
        img = F.conv2d(img, k.view(1, 1, 1, -1), padding=(0, 5))
        return F.conv2d(img, k.view(1, 1, -1, 1), padding=(5, 0))

    mx, my = filt(a), filt(b)
    sxx = filt(a * a) - mx * mx
    syy = filt(b * b) - my * my
    sxy = filt(a * b) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    ssim = ssim[:, 0].mean(dim=0)
    l1 = (render - target).abs().mean(dim=-1)
    return (1 - lam) * l1 + lam * ((1 - ssim) / 2).clamp(0, 1)


def regulariser_torch(coeffs, b: float = 1.0):
    """Closed form of sum_k int_S2 (b - u_k)^2 from SH orthonormality."""
    import math

    return (
        coeffs.shape[0] * 4 * math.pi * b * b
        - 4 * math.sqrt(math.pi) * b * coeffs[:, 0].sum()
        + (coeffs**2).sum()
    )


def check_against_numpy() -> None:
    """Abort early if the torch mirrors disagree with the tested numpy reference."""
    import numpy as np
    import torch

    from movable_reality.coverage import uncertainty as U

    rng = np.random.default_rng(0)
    d = rng.normal(size=(64, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ok = np.allclose(sh_basis_torch(torch.from_numpy(d)).numpy(), U.sh_basis(d), atol=1e-12)
    a, b = rng.uniform(size=(2, 32, 40, 3))
    r_t = pixel_residual_torch(torch.from_numpy(a).float(), torch.from_numpy(b).float()).numpy()
    ok &= np.allclose(r_t, U.pixel_residual(a, b), atol=1e-4)
    c = rng.normal(size=(5, 16))
    ok &= np.isclose(float(regulariser_torch(torch.from_numpy(c))), U.regulariser(c).sum())
    if not ok:
        raise RuntimeError("torch mirrors of the uncertainty maths disagree with numpy")


# ---------------------------------------------------------------- the job


def _load_gaussians(ply: Path, device):
    """Means, quats (wxyz), scales, opacities, SH (N, K, 3) from a gsplat/3DGS PLY."""
    import numpy as np
    import torch
    from plyfile import PlyData

    v = PlyData.read(str(ply))["vertex"].data
    names = v.dtype.names
    f = lambda *ks: np.stack([v[k] for k in ks], axis=1).astype(np.float32)  # noqa: E731
    means = f("x", "y", "z")
    quats = f("rot_0", "rot_1", "rot_2", "rot_3")
    scales = np.exp(f("scale_0", "scale_1", "scale_2"))
    opac = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    dc = f("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :]
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if rest_names:
        rest = f(*rest_names).reshape(len(v), 3, -1).transpose(0, 2, 1)  # channel-major in PLY
        sh = np.concatenate([dc, rest], axis=1)
    else:
        sh = dc
    sh_degree = int(round(np.sqrt(sh.shape[1]))) - 1
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    return t(means), t(quats), t(scales), t(opac), t(sh), sh_degree


def _load_views(root: Path, long_side: int):
    """(Camera rescaled to the image actually loaded, image (H, W, 3) float32) per view.

    Prefers the full-resolution frames/ when their aspect ratio matches the COLMAP camera
    (intrinsics are then rescaled), else recon/images/ (MapAnything's ~518 px export, which
    may be cropped to a different aspect ratio)."""
    import numpy as np
    from PIL import Image

    from movable_reality.coverage.colmap import read_model

    out = []
    for cam in read_model(root / "recon"):
        img = None
        for d in (root / "frames", root / "recon" / "images"):
            p = d / cam.name
            if not p.exists():
                continue
            cand = Image.open(p).convert("RGB")
            if abs(cand.size[0] / cand.size[1] - cam.width / cam.height) < 0.01:
                img = cand
                break
        if img is None:
            print(f"skip {cam.name}: no image with the camera's aspect ratio", flush=True)
            continue
        s = long_side / max(img.size)
        if s < 1:
            img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
        cam = cam.scaled(max(img.size))
        out.append((cam, np.asarray(img, dtype=np.float32) / 255.0))
    return out


@app.function(gpu=GPU, volumes={VOLUME_ROOT: scenes_vol}, timeout=60 * 60)
def fit_uncertainty(
    scene_id: str,
    iters: int = 3000,
    lam: float = 0.2,
    prior_b: float = 1.0,
    lr: float = 0.01,
    long_side: int = 960,
    degree: int = 3,
    force: bool = False,
) -> dict:
    import numpy as np
    import torch
    from gsplat import rasterization

    check_against_numpy()
    scenes_vol.reload()
    root = Path(VOLUME_ROOT) / "scenes" / scene_id
    out = root / "coverage" / "uncertainty.npz"
    if out.exists() and not force:
        return {"skipped": str(out)}
    dev = torch.device("cuda")
    t0 = time.time()
    means, quats, scales, opac, sh, sh_degree = _load_gaussians(root / "splat" / "scene.ply", dev)
    views = _load_views(root, long_side)
    if not views:
        raise RuntimeError(f"no training images found for {scene_id}")
    n, s = means.shape[0], (degree + 1) ** 2
    t_load = time.time() - t0

    # 1. Frozen colour renders -> per-pixel residual targets (kept on the GPU, fp16).
    t0 = time.time()
    targets, mats = [], []
    with torch.no_grad():
        for cam, img in views:
            viewmat = torch.eye(4, device=dev)
            viewmat[:3, :3] = torch.from_numpy(cam.R).float()
            viewmat[:3, 3] = torch.from_numpy(cam.t).float()
            K = torch.from_numpy(cam.K).float().to(dev)
            rgb, _, _ = rasterization(
                means, quats, scales, opac, sh, viewmat[None], K[None],
                cam.width, cam.height, sh_degree=sh_degree, packed=False,
            )  # fmt: skip
            target = torch.from_numpy(img).to(dev)
            res = pixel_residual_torch(rgb[0].clamp(0, 1), target)
            targets.append(res.half())
            mats.append((viewmat, K, cam.width, cam.height, torch.from_numpy(cam.center).float()))
    t_residual = time.time() - t0

    # 2. Fit the SH uncertainty channel (Adam, one view per step, regulariser scaled by
    #    1 / n_views so the stochastic gradient matches the full objective).
    from movable_reality.coverage.uncertainty import C0

    coeffs = torch.zeros((n, s), device=dev)
    coeffs[:, 0] = prior_b / C0  # start at the prior
    coeffs.requires_grad_(True)
    opt = torch.optim.Adam([coeffs], lr=lr)
    n_views = len(mats)
    losses = []
    t0 = time.time()
    rng = np.random.default_rng(0)
    for it in range(iters):
        i = int(rng.integers(n_views))
        viewmat, K, w, h, centre = mats[i]
        d = means - centre.to(dev)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        u = (coeffs * sh_basis_torch(d, degree)).sum(-1, keepdim=True)  # (N, 1)
        U, _, _ = rasterization(
            means, quats, scales, opac, u, viewmat[None], K[None], w, h, packed=False
        )
        data = ((U[0, ..., 0] - targets[i].float()) ** 2).sum()
        loss = data + lam / n_views * regulariser_torch(coeffs, prior_b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 100 == 0 or it == iters - 1:
            losses.append([it, float(data), float(loss)])
            print(f"it {it} data {float(data):.4f} loss {float(loss):.4f}", flush=True)
    t_fit = time.time() - t0

    # 3. Summaries for coverage.
    c = coeffs.detach()
    dirs = torch.from_numpy(_fib(64)).float().to(dev)
    u_dirs = c @ sh_basis_torch(dirs, degree).T  # (N, 64)
    u_mean = (c[:, 0] * C0 / prior_b).clamp(0, 1)
    u_max = (u_dirs.max(dim=1).values / prior_b).clamp(0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        coeffs=c.cpu().numpy().astype(np.float16),
        u_mean=u_mean.cpu().numpy().astype(np.float32),
        u_max=u_max.cpu().numpy().astype(np.float32),
    )
    info = {
        "gaussians": n,
        "views": n_views,
        "iters": iters,
        "lam": lam,
        "prior_b": prior_b,
        "lr": lr,
        "degree": degree,
        "long_side": long_side,
        "gsplat": GSPLAT_VERSION,
        "timings_s": {
            "load": round(t_load, 1),
            "residual": round(t_residual, 1),
            "fit": round(t_fit, 1),
        },
        "loss": losses,
        "u_mean_percentiles": np.percentile(u_mean.cpu().numpy(), [5, 25, 50, 75, 95]).tolist(),
    }
    (out.parent / "uncertainty.json").write_text(json.dumps(info, indent=2))
    scenes_vol.commit()
    return info


def _fib(n: int):
    from movable_reality.coverage.voxels import fibonacci_sphere

    return fibonacci_sphere(n)


@app.local_entrypoint()
def main(
    scene: str,
    iters: int = 3000,
    lam: float = 0.2,
    lr: float = 0.01,
    long_side: int = 960,
    force: bool = False,
    skip_upload: bool = False,
):
    local = Path(__file__).resolve().parents[1] / "data" / "scenes" / scene
    if not skip_upload:
        for sub in ("splat/scene.ply",):
            if not (local / sub).exists():
                raise SystemExit(f"{local / sub} missing")
        with scenes_vol.batch_upload(force=True) as batch:
            batch.put_file(local / "splat" / "scene.ply", f"/scenes/{scene}/splat/scene.ply")
            batch.put_directory(local / "recon", f"/scenes/{scene}/recon")
            if (local / "frames").is_dir():
                batch.put_directory(local / "frames", f"/scenes/{scene}/frames")
        print(f"uploaded {local}")
    info = fit_uncertainty.remote(
        scene, iters=iters, lam=lam, lr=lr, long_side=long_side, force=force
    )
    print(json.dumps({k: v for k, v in info.items() if k != "loss"}, indent=2))
    (local / "coverage").mkdir(parents=True, exist_ok=True)
    for name in ("uncertainty.npz", "uncertainty.json"):
        with open(local / "coverage" / name, "wb") as f:
            for chunk in scenes_vol.read_file(f"/scenes/{scene}/coverage/{name}"):
                f.write(chunk)
    print(f"downloaded to {local / 'coverage'}; now run scripts/score_coverage.py --force")
