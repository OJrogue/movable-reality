"""M4 insert path: "add a small wooden stool by the window" -> a new generated object.

Muse Spark (noun + typical size) -> Muse Image (one object, plain background) -> SAM 3.1
image mask -> SAM 3D Objects on Modal -> stand upright on the floor at (x, z), scale to the
typical size -> objects/gen_<n>.ply (recon frame, every splat `generated` = 1) -> SceneObject
"gen#<n>" plus a history entry "insert" (undo hides it).

Every paid call is cached under data/cache/ (chat/, insert/, sam/, sam3d/), so a rerun of the
same prompt costs nothing and gives the same object.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import splat_ops as so
from .agent.llm import cache_only, cached_chat
from .clients import meta_async_client, meta_client
from .config import CACHE_DIR, Models
from .sam import paste_mask, segment_image
from .sam3d import mask_png, run_sam3d
from .scene import Scene, SceneObject

IMAGE_SIZE = "1024x1024"
SIZE_PROMPT = (
    "The user wants to add this object to a room: {prompt!r}. Reply with JSON only: "
    '{{"noun": "<one or two word concrete noun for segmentation, e.g. stool>", '
    '"width_m": <typical width in metres>, "height_m": <typical height in metres>, '
    '"depth_m": <typical depth in metres>}}. Take adjectives like small or large into account.'
)
IMAGE_PROMPT = (
    "Product photo of a single {prompt}. The whole object is visible with margin on every "
    "side, centred, seen from a three-quarter front view at a slight downward angle. Plain "
    "light grey seamless background, soft even studio lighting, no other objects, no text."
)


def _run(coro):
    """Run a coroutine from sync code, also when called inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


def parse_size(text: str) -> dict[str, Any]:
    """Noun and size from Muse Spark's reply; tolerant of code fences and extra prose."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in size reply: {text[:200]!r}")
    d = json.loads(m.group(0))
    size = tuple(float(d[k]) for k in ("width_m", "height_m", "depth_m"))
    if not all(0.02 <= v <= 5.0 for v in size):
        raise ValueError(f"implausible size {size}")
    return {"noun": str(d.get("noun") or "object").strip().lower(), "size": size}


async def object_size(prompt: str) -> dict[str, Any]:
    data = await cached_chat(
        meta_async_client(),
        "meta",
        model=Models.FAST,
        messages=[{"role": "user", "content": SIZE_PROMPT.format(prompt=prompt)}],
        temperature=0,
    )
    return parse_size(data["choices"][0]["message"]["content"])


def generate_image(prompt: str) -> Path:
    """Muse Image text-to-image, cached as data/cache/insert/<hash>.png."""
    full = IMAGE_PROMPT.format(prompt=prompt)
    key = hashlib.sha256(json.dumps([Models.IMAGE, full, IMAGE_SIZE]).encode()).hexdigest()[:24]
    path = CACHE_DIR / "insert" / f"{key}.png"
    if path.exists():
        return path
    if cache_only():
        raise RuntimeError("image not in the cache and MR_CACHE_ONLY=1")
    resp = meta_client().images.generate(model=Models.IMAGE, prompt=full, size=IMAGE_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(resp.data[0].b64_json))
    return path


def object_mask(image: Path, noun: str) -> np.ndarray:
    """Largest SAM 3.1 instance of `noun` (falls back to the noun "object")."""
    w, h = Image.open(image).size
    for query in dict.fromkeys([noun, noun.split()[-1], "object"]):
        inst = _run(segment_image(image, query))
        if inst:
            masks = [paste_mask(i, w, h) for i in inst]
            return max(masks, key=lambda m: int(m.sum()))
    raise RuntimeError(f"SAM 3.1 found no '{noun}' in the generated image {image}")


def next_gen_index(scene: Scene) -> int:
    used = [int(o.id.split("#")[1]) for o in scene.objects if re.fullmatch(r"gen#\d+", o.id)]
    return max(used, default=0) + 1


def place_generated(
    local: np.ndarray,
    cam_from_local: np.ndarray,
    size: tuple[float, float, float] | None,
    x: float,
    z: float,
    world_from_recon: list[list[float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Canonical SAM 3D splats -> recon-frame splats standing on the floor at (x, z).
    Returns the splats (flag `generated` = 1) and placement info (world bbox, scale)."""
    R_up, tilt = so.upright_rotation(cam_from_local)
    p = so.xyz(local) @ R_up.T
    alpha = so.opacity(local)
    lo, hi = so.robust_bounds(p, alpha)
    s = so.fit_scale(hi - lo, size)
    t = so.floor_placement(p, alpha, x, z, s)
    world_from_local = so.similarity(R_up, t, s)
    recon_from_local = np.linalg.inv(np.asarray(world_from_recon, dtype=np.float64))
    out = so.apply_matrix(so.standard(local, generated=1), recon_from_local @ world_from_local)
    wlo, whi = so.robust_bounds(s * p + t, alpha, pct=0.5)
    info = {
        "bbox_min": [float(v) for v in wlo],
        "bbox_max": [float(v) for v in whi],
        "scale": s,
        "tilt_deg": round(tilt, 1),
    }
    return out, info


def insert_object(
    scene: Scene,
    folder: Path,
    prompt: str,
    x: float,
    z: float,
    instruction: str | None = None,
) -> dict[str, Any]:
    """Generate `prompt`, stand it on the floor at world (x, z), add it to `scene` (in memory;
    the caller saves scene.json). Returns a summary for the agent tool."""
    folder = Path(folder)
    t0 = time.time()
    timings: dict[str, float] = {}
    meta = _run(object_size(prompt))
    timings["size_s"] = round(time.time() - t0, 1)

    t1 = time.time()
    image = generate_image(prompt)
    timings["image_s"] = round(time.time() - t1, 1)

    t1 = time.time()
    mask = object_mask(image, meta["noun"])
    timings["sam_s"] = round(time.time() - t1, 1)

    t1 = time.time()
    res = run_sam3d(image.read_bytes(), mask_png(mask))
    timings["sam3d_s"] = round(time.time() - t1, 1)

    A, _, _ = res.cam_from_local()
    splats, info = place_generated(res.vertices, A, meta["size"], x, z, scene.world_from_recon)
    n = next_gen_index(scene)
    rel = f"objects/gen_{n}.ply"
    so.write_vertices(folder / rel, splats)
    Image.open(image).save(folder / "objects" / f"gen_{n}_source.png")
    obj = SceneObject(
        id=f"gen#{n}",
        label=meta["noun"],
        ply=rel,
        bbox_min=tuple(info["bbox_min"]),  # type: ignore[arg-type]
        bbox_max=tuple(info["bbox_max"]),  # type: ignore[arg-type]
        source="generated",
        prompt=prompt,
    )
    scene.objects.append(obj)
    if scene.coverage is not None:
        scene.coverage.generated_splats += len(splats)
    size = [round(v, 2) for v in obj.size]
    scene._log(  # noqa: SLF001 - the insert op is logged like the other edits
        "insert",
        obj.id,
        {"prompt": prompt, "x": x, "z": z, "size": size, "splats": len(splats)},
        instruction,
    )
    timings["total_s"] = round(time.time() - t0, 1)
    return {
        "id": obj.id,
        "label": obj.label,
        "size": size,
        "position": [round(v, 2) for v in obj.center],
        "splats": len(splats),
        "tilt_deg": info["tilt_deg"],
        "timings": timings,
    }
