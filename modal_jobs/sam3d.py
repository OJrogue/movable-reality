"""SAM 3D Objects on Modal: one image + one object mask -> the object as 3D Gaussians.

    uv run modal deploy modal_jobs/sam3d.py   # once, then insert.py and complete_objects.py call it
    uv run modal run modal_jobs/sam3d.py --image data/x.png --mask data/x_mask.png --out x.ply

Weights: gated `facebook/sam-3d-objects` (SAM License) on Hugging Face, cached on the
`movable-reality-cache` volume. HF_TOKEN goes in as a Modal secret built from the local
environment (.env) at deploy time and is never written to the repo.

Only Meta models run here (AGENTS.md rule 2). SAM 3D Objects' default pipeline estimates the
pointmap of the image with MoGe (a Microsoft model). We switch that off (`depth_model: null`)
and pass a pointmap from MapAnything (Meta, Apache-2.0 weights) instead, through the
pipeline's own `pointmap=` argument. The MoGe *package* is installed because SAM 3D Objects
imports its geometry helpers (plain functions, no weights), and its network is never loaded.

Output of `Sam3D.run`: the Gaussians in the object's canonical frame as a standard 3DGS PLY
(x y z, nx ny nz, f_dc_0..2, opacity, scale_0..2, rot_0..3), plus the predicted pose that
places them in the camera frame (PyTorch3D convention). `movable_reality.gaussians` turns that
into OpenCV camera coordinates, the reconstruction frame or a floor placement.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import modal

APP_NAME = "movable-reality-sam3d"
# 32 GB VRAM needed (doc/setup.md). L40S (48 GB) is the cheapest Modal GPU above that.
GPU = "L40S"
SAM3D_SHA = "f91db411c50efee93d8db7aeb323885650f6f722"  # main on 2 Jun 2026
PYTORCH3D_SHA = "75ebeeaea0908c5527e7b1e305fbc7681382db47"  # pinned by sam-3d-objects
MOGE_SHA = "a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"  # pinned by sam-3d-objects (code only)
UTILS3D_SHA = "3913c65d81e05e47b9f367250cf8c0f7462a0900"  # pinned by MoGe
GSPLAT_SHA = "2323de5905d5e90e035f792fe65bad0fedd413e7"  # pinned by sam-3d-objects
MAPANYTHING_SHA = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"  # same as reconstruct.py
SAM3D_REPO = "facebook/sam-3d-objects"
MAPANYTHING_REPO = "facebook/map-anything-apache"
CACHE = "/cache"
CKPT_DIR = f"{CACHE}/sam3d/checkpoints"

# Packages from sam-3d-objects/requirements.txt that inference never imports (training,
# notebooks, cloud tooling). `dataclasses==0.6` would shadow the standard library.
_SKIP = (
    "bpy|sagemaker|auto_gptq|bitsandbytes|wandb|jupyter|conda-pack|Flask|Werkzeug|pymongo|hdfs|"
    "librosa|decord|av|sentence-transformers|optimum|peft|mosaicml-streaming|polyscope|pyrender|"
    "smplx|nvidia-pyindex|nvidia-cuda-nvcc-cu12|cuda-python|black|flake8|usort|autoflake|pdoc3|"
    "tensorboard|gdown|pip-system-certs|webdataset|fastavro|crcmod|panda3d-gltf|torchaudio|MoGe|"
    "hydra-submitit-launcher|pytest|OpenEXR|imath|python-pycg|dataclasses|xformers"
)
_GCC = {"CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -shared", "LDCXXSHARED": "g++ -shared"}

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
        "libxrender1",
        "libxext6",
        "libsm6",
        "libgomp1",
    )  # fmt: skip
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2",
        "setuptools<75",
        "wheel",
        "ninja",
        "hatchling",
        "hatch-requirements-txt",
        "trimesh",
        "pyvista",
        "plotly",
        "seaborn",
        "matplotlib",
        "scipy",
        "imageio",
        "einops",
        "safetensors",
        "omegaconf",
        "plyfile",
        "tqdm",
        "click",
        "huggingface_hub[hf_transfer]<1.0",
    )  # fmt: skip
    .run_commands(
        "git clone https://github.com/facebookresearch/sam-3d-objects /opt/sam3d",
        f"git -C /opt/sam3d checkout {SAM3D_SHA}",
        f"grep -v -E '^({_SKIP})([=<> @]|$)' /opt/sam3d/requirements.txt > /tmp/req.txt",
        "cat /tmp/req.txt",
        "pip install -r /tmp/req.txt 'numpy<2'",
        "pip install --no-deps -e /opt/sam3d",
        # Patch from the official setup (doc/setup.md): hydra PR 2863, pinned commit.
        "python /opt/sam3d/patching/hydra",
    )
    .pip_install(
        "kaolin==0.17.0",
        find_links="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html",
    )
    .run_commands(
        f"pip install --no-deps 'git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_SHA}'",
        f"pip install --no-deps 'git+https://github.com/microsoft/MoGe.git@{MOGE_SHA}'",
        # No CUDA build: gsplat is only imported (its rasteriser serves video renders we skip).
        f"BUILD_NO_CUDA=1 pip install --no-deps 'git+https://github.com/nerfstudio-project/gsplat.git@{GSPLAT_SHA}'",
        "pip install jaxtyping rich",
    )
    .run_commands(
        # PyTorch3D: only transforms and structures are used, so a CPU build is enough and
        # avoids an nvcc compile on the 1-CPU builder (see reconstruct.py).
        "git clone https://github.com/facebookresearch/pytorch3d /opt/pytorch3d",
        f"git -C /opt/pytorch3d checkout {PYTORCH3D_SHA}",
        "nproc",
        "cd /opt/pytorch3d && FORCE_CUDA=0 MAX_JOBS=8 pip install -v --no-build-isolation .",
        env=_GCC,
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/map-anything /opt/map-anything",
        f"git -C /opt/map-anything checkout {MAPANYTHING_SHA}",
        # --no-deps: its opencv-python-headless pin clashes with opencv-python above.
        "pip install --no-deps -e /opt/map-anything",
        "pip install uniception==0.1.7 natsort python-box pillow-heif orjson 'numpy<2'",
    )
    .env(
        {
            "LIDRA_SKIP_INIT": "true",
            "CUDA_HOME": "/usr/local/cuda",
            "HF_HOME": f"{CACHE}/hf",
            "TORCH_HOME": f"{CACHE}/torch",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .run_commands(
        "python -c 'import sam3d_objects.pipeline.inference_pipeline_pointmap, "
        'pytorch3d.transforms, spconv.pytorch, kaolin, mapanything.models; print("imports ok")\''
    )
)

app = modal.App(APP_NAME, image=image)
cache_vol = modal.Volume.from_name("movable-reality-cache", create_if_missing=True)

if modal.is_local():
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN missing from .env: the SAM 3D Objects weights are gated")
hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})


def _fixed_size(w: int, h: int, long_side: int = 518, patch: int = 14) -> tuple[int, int]:
    """MapAnything input size with the image's aspect ratio, both sides multiples of 14."""
    s = long_side / max(w, h)
    return max(patch, round(w * s / patch) * patch), max(patch, round(h * s / patch) * patch)


@app.cls(
    gpu=GPU,
    volumes={CACHE: cache_vol},
    secrets=[hf_secret],
    timeout=30 * 60,
    scaledown_window=120,
)
class Sam3D:
    @modal.enter()
    def load(self) -> None:
        import torch  # noqa: PLC0415 - container only
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from hydra.utils import instantiate  # noqa: PLC0415
        from mapanything.models import MapAnything  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        t0 = time.time()
        if not Path(CKPT_DIR, "pipeline.yaml").exists():
            dl = snapshot_download(
                SAM3D_REPO, allow_patterns=["checkpoints/*"], local_dir=f"{CACHE}/sam3d"
            )
            print("downloaded", dl)
            cache_vol.commit()
        self.timings = {"download_s": round(time.time() - t0, 1)}

        t0 = time.time()
        config = OmegaConf.load(f"{CKPT_DIR}/pipeline.yaml")
        config.depth_model = None  # no MoGe network: the pointmap comes from MapAnything
        # Both Gaussian decoders (32 or 4 splats per voxel). The mesh decoder runs only when
        # a call asks for it (run(mesh=True)), through the pipeline's decode_formats argument.
        config.decode_formats = ["gaussian", "gaussian_4"]
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = CKPT_DIR
        self.pipeline = instantiate(config)
        self.mapanything = MapAnything.from_pretrained(MAPANYTHING_REPO).to("cuda").eval()
        cache_vol.commit()
        self.timings["load_s"] = round(time.time() - t0, 1)
        print("loaded", self.timings, f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    def _pointmap(self, rgb):
        """MapAnything pointmap of one image, camera frame, PyTorch3D axes (x left, y up)."""
        import tempfile  # noqa: PLC0415

        import torch  # noqa: PLC0415
        from mapanything.utils.image import load_images  # noqa: PLC0415

        h, w = rgb.shape[:2]
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/view.png"
            from PIL import Image  # noqa: PLC0415

            Image.fromarray(rgb).save(path)
            views = load_images([path], resize_mode="fixed_size", size=_fixed_size(w, h))
        pred = self.mapanything.infer(
            views, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
            apply_mask=False, mask_edges=False,
        )[0]  # fmt: skip
        pts = pred["pts3d_cam"][0].float().clone()  # (h', w', 3) OpenCV: x right, y down, z forward
        # Invalid points as NaN, not 0: SAM 3D's focal solver skips non-finite points but
        # divides by z = 0 (crash on the first run, 30 Oct).
        bad = ~torch.isfinite(pts).all(-1) | (pts[..., 2] <= 1e-4)
        pts[bad] = float("nan")
        pts = pts * torch.tensor([-1.0, -1.0, 1.0], device=pts.device)  # to PyTorch3D camera
        K = pred["intrinsics"][0].float().cpu().numpy().tolist()
        return pts, K, tuple(pts.shape[:2])

    @modal.method()
    def run(
        self, image_bytes: bytes, mask_png: bytes, seed: int = 42, dense: bool = False,
        mesh: bool = False,
    ) -> dict:  # fmt: skip
        """dense=False returns the 4-splats-per-voxel decode (about 120k splats for a stool,
        8x fewer than dense), which suits the headset budget. mesh=True also decodes the
        object as a triangle mesh with vertex colours and returns it as GLB bytes ("glb", in
        the GLB's y-up canonical frame, which to_glb rotates from the Gaussians' z-up one).
        Texture baking is off: it renders 100 views through PyTorch3D, which is a CPU build
        here."""
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        t0 = time.time()
        rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        mask = np.asarray(Image.open(io.BytesIO(mask_png)).convert("L")) > 127
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"mask {mask.shape} does not match image {rgb.shape[:2]}")
        if mask.sum() < 64:
            raise ValueError("mask has fewer than 64 pixels")
        pointmap, K, pm_size = self._pointmap(rgb)
        t_pm = time.time() - t0

        t1 = time.time()
        rgba = np.concatenate([rgb, mask[..., None].astype(np.uint8) * 255], axis=-1)
        out = self.pipeline.run(
            rgba, None, seed, stage1_only=False, with_mesh_postprocess=False,
            with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True,
            pointmap=pointmap,
            decode_formats=["gaussian", "gaussian_4", "mesh"] if mesh else None,
        )  # fmt: skip
        t_gen = time.time() - t1

        gs = out["gaussian" if dense else "gaussian_4"][0]
        with __import__("tempfile").NamedTemporaryFile(suffix=".ply") as f:
            gs.save_ply(f.name)
            ply = Path(f.name).read_bytes()

        def vec(key: str) -> list[float]:
            return out[key].detach().float().reshape(-1).cpu().numpy().tolist()

        timings = {
            **self.timings,
            "pointmap_s": round(t_pm, 1),
            "generate_s": round(t_gen, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }
        glb = None
        if mesh and out.get("glb") is not None:
            glb = out["glb"].export(file_type="glb")
            timings["mesh_faces"] = int(len(out["glb"].faces))
        print("sam3d run", timings, f"{len(ply) / 1e6:.1f} MB")
        return {
            "ply": ply,
            "glb": glb,
            # Pose of the canonical Gaussians in the camera frame (PyTorch3D axes, row-vector
            # convention): x_cam = (s * x_local) @ R(q) + t, with q = (w, x, y, z).
            "rotation": vec("rotation"),
            "translation": vec("translation"),
            "scale": vec("scale"),
            "intrinsics": K,
            "pointmap_size": list(pm_size),
            "timings": timings,
        }


@app.local_entrypoint()
def main(
    image: str, mask: str, out: str = "sam3d_out.ply", seed: int = 42, dense: bool = False,
    mesh: bool = False,
):  # fmt: skip
    import json

    t0 = time.time()
    res = Sam3D().run.remote(Path(image).read_bytes(), Path(mask).read_bytes(), seed, dense, mesh)
    Path(out).write_bytes(res.pop("ply"))
    glb = res.pop("glb", None)
    if glb:
        Path(out).with_suffix(".glb").write_bytes(glb)
    Path(out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), f"\nwall {time.time() - t0:.0f} s -> {out}")
