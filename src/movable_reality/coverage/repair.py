"""Repair groundwork (C3): render a view of a small gap, repair it with Muse Image edit using
the nearest real frames as references, and cache the result.

Only Meta models are called (AGENTS.md): Muse Image (`config.Models.IMAGE`) through the
OpenAI-compatible Meta Model API client in clients.py, endpoint `/v1/images/edits`:
`client.images.edit(model=..., image=[files...], prompt=..., n=1, size=...)`, result in
`data[0].b64_json` (docs: dev.meta.ai, "Build with Muse Image"; size only sets the aspect
ratio, output is up to about 1600 px; $0.01 per image).

Flow per gap:
1. `repair_camera`: a camera at the gap's next-best view (or a walkable pose near it)
   looking at the gap centre.
2. `render_gap_view`: CPU preview render of the scene from that camera plus a mask of
   pixels that show the gap (inside the gap sphere, or empty).
3. `nearest_frames`: the training frames whose cameras look most like the repair camera
   (position and direction), as references for the scene's look.
4. `repair_view`: Muse Image edit; the rendered view is the first image, the references
   follow. Cached under data/cache/muse_image/ by a hash of the model, prompt, size and
   input image bytes; reruns hit the cache and make no call.
5. Distilling the repaired view back into splats is the next step (gsplat fine-tune of
   the gap region on the repaired view, new splats tagged with `ply.append_generated`).
   Not implemented yet.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..config import CACHE_DIR, Models
from ..scene import Gap
from .cameras import Camera, look_at
from .render import render_points, to_uint8
from .voxels import room_bounds

DEFAULT_PROMPT = (
    "This is a rendered view of a 3D scan of a real room. The part of the image marked in "
    "magenta was never seen by the camera and is missing or broken. Fill only that part so "
    "it continues the surfaces, furniture, materials and lighting around it, matching the "
    "reference photos of the same room. Keep every other pixel, the viewpoint and the "
    "perspective unchanged. Do not add new objects. Photorealistic, no text."
)
MASK_COLOR = (255, 0, 255)


@dataclass(frozen=True)
class RepairRequest:
    gap_id: str
    view: Path  # rendered view, PNG
    references: tuple[Path, ...]  # nearest real frames, JPG/PNG
    prompt: str = DEFAULT_PROMPT
    size: str = "1536x1024"  # aspect ratio hint for Muse Image


@dataclass(frozen=True)
class RepairResult:
    gap_id: str
    image: Path  # repaired view on disk
    cached: bool
    cache_key: str


# ---------------------------------------------------------------- views


def repair_camera(
    gap: Gap,
    width: int = 768,
    height: int = 512,
    hfov_deg: float = 65.0,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    obstacles: np.ndarray | None = None,
    fill: float = 0.5,
    margin: float = 0.15,
    clearance: float = 0.3,
) -> Camera:
    """Camera at the gap's best next view (else 1.5 m in front of the gap at eye height)
    looking at the gap centre.

    A next-best view stands close to its gap, so the gap sphere can fill the whole frame and
    Muse Image gets no surroundings to continue. With `bounds` (room min and max corners,
    world frame) the eye backs off horizontally, at its own height, away from the gap until
    the gap's angular radius is at most `fill` times the vertical half FOV. It stops
    `margin` inside the bounds and `clearance` before the first of `obstacles` (opaque splat
    centres) on its way, so it never ends up in a wall or a piece of furniture. A fallback
    eye is clamped into the bounds."""
    c = np.asarray(gap.center, dtype=np.float64)
    if gap.next_best_views:
        eye = np.asarray(gap.next_best_views[0].eye, dtype=np.float64)
        target = np.asarray(gap.next_best_views[0].target, dtype=np.float64)
    else:
        eye = c + np.array([0.0, 0.0, 1.5 + gap.radius])
        eye[1] = 1.6
        target = c
    if bounds is not None:
        lo = np.asarray(bounds[0], dtype=np.float64) + margin
        hi = np.asarray(bounds[1], dtype=np.float64) - margin
        if not gap.next_best_views:
            eye = np.clip(eye, lo, np.maximum(lo, hi))
        vhalf = np.arctan(np.tan(np.radians(hfov_deg) / 2) * height / width)
        want = gap.radius / np.sin(min(fill * vhalf, np.pi / 2 - 1e-3))
        eye = _back_off(eye, c, want, lo, hi, obstacles, clearance)
    return look_at(eye, target, width, height, hfov_deg, name=f"repair_{gap.id}")


def _back_off(
    eye: np.ndarray,
    centre: np.ndarray,
    want: float,
    lo: np.ndarray,
    hi: np.ndarray,
    obstacles: np.ndarray | None,
    clearance: float,
    step: float = 0.05,
) -> np.ndarray:
    """Move `eye` horizontally away from `centre` until it is `want` metres from it, it
    reaches [lo, hi], or the next step would come within `clearance` of an obstacle. An
    eye straight below or above the gap, or already outside [lo, hi], stays put."""
    away = eye - centre
    away[1] = 0.0
    h0 = float(np.linalg.norm(away))
    dy = float(eye[1] - centre[1])
    if want <= np.hypot(h0, dy) or h0 < 1e-6 or np.any(eye < lo) or np.any(eye > hi):
        return eye
    away /= h0
    need = float(np.sqrt(want * want - dy * dy)) - h0  # horizontal step that reaches `want`
    # Largest step s >= 0 along `away` that keeps eye + s * away inside [lo, hi].
    with np.errstate(divide="ignore", invalid="ignore"):
        s_hi = np.where(away > 0, (hi - eye) / away, np.inf)
        s_lo = np.where(away < 0, (lo - eye) / away, np.inf)
    s_max = float(max(0.0, min(need, s_hi.min(), s_lo.min())))
    if obstacles is None or len(obstacles) == 0 or s_max == 0.0:
        return eye + away * s_max
    pts = np.asarray(obstacles, dtype=np.float64)
    end = eye + away * s_max
    near = np.all(
        (pts >= np.minimum(eye, end) - clearance) & (pts <= np.maximum(eye, end) + clearance),
        axis=1,
    )
    pts = pts[near]
    s_ok = 0.0
    for s in np.append(np.arange(step, s_max, step), s_max):
        p = eye + away * s
        if len(pts) and np.min(np.sum((pts - p) ** 2, axis=1)) < clearance * clearance:
            break
        s_ok = float(s)
    return eye + away * s_ok


def gap_mask(cam: Camera, gap: Gap, depth: np.ndarray) -> np.ndarray:
    """Bool (H, W): pixels whose first surface lies inside the gap sphere, or that show
    nothing at all where the ray passes through the sphere."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack(
        [(u - cam.K[0, 2]) / cam.K[0, 0], (v - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(u)], -1
    )
    d_world = d_cam @ cam.R  # camera -> world for row vectors
    o = cam.center
    c = np.asarray(gap.center, dtype=np.float64)
    r = float(gap.radius)
    # Ray-sphere: |o + t d - c|^2 = r^2 with t in units of camera z (d has z = 1).
    oc = o - c
    a = np.einsum("hwk,hwk->hw", d_world, d_world)
    b = 2 * np.einsum("hwk,k->hw", d_world, oc)
    cc = oc @ oc - r * r
    disc = b * b - 4 * a * cc
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = (-b - sq) / (2 * a)
    t1 = (-b + sq) / (2 * a)
    hit &= t1 > 0
    z = np.where(np.isfinite(depth), depth, np.inf)
    inside = hit & (z >= t0) & (z <= t1)
    empty = hit & ~np.isfinite(depth)
    return inside | empty


def render_gap_view(
    cam: Camera,
    gap: Gap,
    means: np.ndarray,
    rgb: np.ndarray,
    scales: np.ndarray,
    opacity: np.ndarray,
    generated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(image uint8 (H, W, 3) with the gap painted magenta, mask bool (H, W)). World-frame
    splats; generated splats are left out so a repair never builds on a repair."""
    keep = np.ones(len(means), bool) if generated is None else np.asarray(generated) == 0
    img, depth = render_points(
        cam, means[keep], rgb[keep], scales[keep], opacity[keep], background=(0.0, 0.0, 0.0)
    )
    mask = gap_mask(cam, gap, depth)
    out = to_uint8(img)
    out[mask] = MASK_COLOR
    return out, mask


def nearest_frames(
    cameras: list[Camera], target: Camera, k: int = 3, focus: np.ndarray | None = None
) -> list[int]:
    """Indices of the k training cameras most similar to `target`: distance between
    centres plus 1.5 m per radian between viewing directions. With `focus` (the gap
    centre), frames whose image contains it come first, so references show the gap's
    surroundings when any frame saw them."""
    if not cameras:
        return []
    centres = np.stack([c.center for c in cameras])
    fwd = np.stack([c.forward for c in cameras])
    d_pos = np.linalg.norm(centres - target.center, axis=1)
    ang = np.arccos(np.clip(fwd @ target.forward, -1.0, 1.0))
    cost = d_pos + 1.5 * ang
    if focus is not None:
        f = np.asarray(focus, dtype=np.float64)[None]
        sees = np.array([bool(c.in_frustum(f)[0]) for c in cameras])
        cost = cost + np.where(sees, 0.0, 100.0)
    return [int(i) for i in np.argsort(cost, kind="stable")[:k]]


# ---------------------------------------------------------------- Muse Image edit (cached)


def _cache_key(req: RepairRequest) -> str:
    h = hashlib.sha256()
    meta = {"model": Models.IMAGE, "prompt": req.prompt, "size": req.size}
    h.update(json.dumps(meta, sort_keys=True).encode())
    for p in (req.view, *req.references):
        h.update(hashlib.sha256(Path(p).read_bytes()).digest())
    return h.hexdigest()[:24]


def cache_path(req: RepairRequest) -> Path:
    return CACHE_DIR / "muse_image" / f"edit_{_cache_key(req)}.webp"


def repair_view(
    req: RepairRequest, out: Path, client: Any | None = None, force: bool = False
) -> RepairResult:
    """Muse Image edit of a rendered gap view; the rendered view is the first image and the
    real frames follow as references. Cached by request hash: a cache hit makes no call."""
    key = _cache_key(req)
    cached = cache_path(req)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists() and not force:
        out.write_bytes(cached.read_bytes())
        return RepairResult(req.gap_id, out, True, key)
    if client is None:
        from ..clients import meta_client

        client = meta_client()
    files = [open(p, "rb") for p in (req.view, *req.references)]  # noqa: SIM115
    try:
        resp = client.images.edit(
            model=Models.IMAGE, image=files, prompt=req.prompt, n=1, size=req.size
        )
    finally:
        for f in files:
            f.close()
    data = base64.b64decode(resp.data[0].b64_json)
    Image.open(io.BytesIO(data)).verify()  # fail on a broken payload before caching it
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    out.write_bytes(data)
    meta = {"gap_id": req.gap_id, "model": Models.IMAGE, "size": req.size, "cache_key": key}
    meta["usage"] = _usage(resp)
    cached.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return RepairResult(req.gap_id, out, False, key)


def _usage(resp: Any) -> Any:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    dump = getattr(usage, "model_dump", None)
    return dump() if callable(dump) else str(usage)


def prepare_request(
    folder: Path,
    gap: Gap,
    cameras: list[Camera],
    splats_world: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    k_refs: int = 3,
) -> RepairRequest:
    """Render the gap view and pick reference frames for one gap; files go under
    coverage/repair/<gap_id>/ in the scene folder."""
    folder = Path(folder)
    means, rgb, scales, opacity, generated = splats_world
    real = np.ones(len(means), bool) if generated is None else np.asarray(generated) == 0
    solid = real & (np.asarray(opacity) >= 0.3)
    bounds = room_bounds(means[real], opacity[real]) if real.any() else None
    cam = repair_camera(gap, bounds=bounds, obstacles=means[solid])
    img, mask = render_gap_view(cam, gap, means, rgb, scales, opacity, generated)
    out = folder / "coverage" / "repair" / gap.id
    out.mkdir(parents=True, exist_ok=True)
    view = out / "view.png"
    Image.fromarray(img).save(view)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "mask.png")
    refs = []
    for i in nearest_frames(cameras, cam, k_refs, focus=np.asarray(gap.center)):
        p = folder / "frames" / cameras[i].name
        if p.exists():
            refs.append(p)
    (out / "request.json").write_text(
        json.dumps(
            {
                "gap_id": gap.id,
                "camera": {"eye": cam.center.tolist(), "forward": cam.forward.tolist()},
                "references": [p.relative_to(folder).as_posix() for p in refs],
                "mask_pixels": int(mask.sum()),
            },
            indent=2,
        )
    )
    return RepairRequest(gap_id=gap.id, view=view, references=tuple(refs))
