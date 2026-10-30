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
5. Distilling the repaired views back into splats: distil.py packs them with seed points
   in the gap voxels, modal_jobs/coverage.py (--job distil) fits new Gaussians only and
   returns them as coverage/generated.ply, every splat flagged generated = 1.

The gap view comes from gsplat on Modal (modal_jobs/coverage.py --job render, alpha
blended) when coverage_repair.py runs with --gpu, else from the CPU point preview.
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
    view_index: int = 0,
) -> Camera:
    """Camera at the gap's next-best view number `view_index` (clamped to the views there
    are) (else 1.5 m in front of the gap at eye height)
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
        nbv = gap.next_best_views[min(view_index, len(gap.next_best_views) - 1)]
        eye = np.asarray(nbv.eye, dtype=np.float64)
        target = np.asarray(nbv.target, dtype=np.float64)
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


MAX_ALIGNMENT_L1 = 0.08


def alignment_error(view: Path, repaired: Path, mask: Path) -> float:
    """Mean absolute difference (0..1) between the rendered view and its repair over the
    pixels Muse Image was asked to keep (outside the gap mask, not empty black). With several
    reference photos Muse Image tends to redraw the room from a reference's viewpoint
    (covtest, 30 Oct: 3 references moved the camera, 1 kept it). Such a repair is not
    aligned with the camera and must not be distilled."""
    a = np.asarray(Image.open(view).convert("RGB"), np.float32) / 255.0
    b = Image.open(repaired).convert("RGB")
    b = np.asarray(b.resize((a.shape[1], a.shape[0]), Image.LANCZOS), np.float32) / 255.0
    m = np.asarray(Image.open(mask).convert("L")) > 127
    keep = ~m & (a.max(axis=-1) > 0.02)
    if not keep.any():
        return 1.0
    return float(np.abs(a - b).mean(axis=-1)[keep].mean())


def write_check(out_dir: Path, max_l1: float = MAX_ALIGNMENT_L1) -> dict:
    """coverage/repair/<view>/check.json: alignment error and whether distil may use it."""
    err = alignment_error(out_dir / "view.png", out_dir / "repaired.webp", out_dir / "mask.png")
    check = {"alignment_l1": round(err, 4), "max_l1": max_l1, "accepted": err <= max_l1}
    (out_dir / "check.json").write_text(json.dumps(check, indent=2))
    return check


def edit_files(files: list[Any]) -> list[tuple[str, Any]]:
    """Multipart parts for the image list. The Meta Model API wants `image[0]`,
    `image[1]`, ... and rejects the `image[]` keys the OpenAI SDK sends for a list (400,
    30 Oct), so the request is built here."""
    parts = []
    for i, f in enumerate(files):
        name = Path(getattr(f, "name", f"image_{i}.png")).name
        mime = "image/png" if name.lower().endswith(".png") else "image/jpeg"
        parts.append((f"image[{i}]", (name, f, mime)))
    return parts


def images_edit(client: Any, files: list[Any], prompt: str, size: str) -> Any:
    """POST /images/edits with indexed image parts through the OpenAI client (auth, retries,
    base URL). A client without `post` (a test double) gets the SDK-style call."""
    if not hasattr(client, "post"):
        return client.images.edit(model=Models.IMAGE, image=files, prompt=prompt, n=1, size=size)
    from openai.types import ImagesResponse

    return client.post(
        "/images/edits",
        body={"model": Models.IMAGE, "prompt": prompt, "n": 1, "size": size},
        files=edit_files(files),
        options={"headers": {"Content-Type": "multipart/form-data"}},
        cast_to=ImagesResponse,
    )


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
        resp = images_edit(client, files, req.prompt, req.size)
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


def view_dir(folder: Path, gap_id: str, view_index: int = 0) -> Path:
    """coverage/repair/<gap_id>/ for a gap's first view, <gap_id>_v<k>/ for view k."""
    name = gap_id if view_index == 0 else f"{gap_id}_v{view_index}"
    return Path(folder) / "coverage" / "repair" / name


def plan_camera(
    gap: Gap,
    splats_world: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    view_index: int = 0,
) -> Camera:
    """The repair camera of one gap view (see repair_camera), world frame."""
    means, _, _, opacity, generated = splats_world
    real = np.ones(len(means), bool) if generated is None else np.asarray(generated) == 0
    solid = real & (np.asarray(opacity) >= 0.3)
    bounds = room_bounds(means[real], opacity[real]) if real.any() else None
    return repair_camera(gap, bounds=bounds, obstacles=means[solid], view_index=view_index)


def load_render(path: Path, world_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rgb, depth in metres, alpha) from render.npz (modal_jobs/coverage.py --job render).
    gsplat renders in the reconstruction frame, so its depth is in recon units: multiply by
    the scale of world_from_recon before comparing it with world-frame rays."""
    z = np.load(path)
    depth = z["depth"].astype(np.float32) * np.float32(world_scale)
    return z["rgb"], depth, z["alpha"].astype(np.float32)


def view_from_render(
    cam: Camera, gap: Gap, rgb: np.ndarray, depth: np.ndarray, alpha: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gap view from a gsplat render (uint8 RGB, expected depth, accumulated alpha): pixels
    with alpha below 0.5 count as empty, then the same mask rule as the CPU preview."""
    d = np.where(alpha >= 0.5, depth, np.inf).astype(np.float32)
    mask = gap_mask(cam, gap, d)
    out = np.array(rgb, dtype=np.uint8, copy=True)
    out[mask] = MASK_COLOR
    return out, mask


def camera_dict(cam: Camera) -> dict:
    """World frame, OpenCV convention: x_cam = R @ x_world + t."""
    return {
        "eye": cam.center.tolist(),
        "forward": cam.forward.tolist(),
        "width": cam.width,
        "height": cam.height,
        "K": cam.K.tolist(),
        "R": cam.R.tolist(),
        "t": cam.t.tolist(),
    }


def camera_from_dict(d: dict, name: str = "") -> Camera:
    return Camera(
        name=name,
        width=int(d["width"]),
        height=int(d["height"]),
        K=np.asarray(d["K"], np.float64),
        R=np.asarray(d["R"], np.float64),
        t=np.asarray(d["t"], np.float64),
    )


def prepare_request(
    folder: Path,
    gap: Gap,
    cameras: list[Camera],
    splats_world: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    k_refs: int = 3,
    view_index: int = 0,
    cam: Camera | None = None,
    render: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> RepairRequest:
    """Render one gap view and pick reference frames; files go under view_dir(). `render`
    is a gsplat render (rgb, depth, alpha) of `cam`. Without it the CPU preview is used."""
    folder = Path(folder)
    means, rgb, scales, opacity, generated = splats_world
    if cam is None:
        cam = plan_camera(gap, splats_world, view_index)
    if render is not None:
        img, mask = view_from_render(cam, gap, *render)
    else:
        img, mask = render_gap_view(cam, gap, means, rgb, scales, opacity, generated)
    out = view_dir(folder, gap.id, view_index)
    out.mkdir(parents=True, exist_ok=True)
    view = out / "view.png"
    old = view.read_bytes() if view.exists() else None
    Image.fromarray(img).save(view)
    if old != view.read_bytes():  # a repair of another view must not reach the distil step
        (out / "repaired.webp").unlink(missing_ok=True)
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
                # Gap ids are renumbered on every rescore; distil checks the gap still matches.
                "gap_center": list(gap.center),
                "gap_radius": gap.radius,
                "view_index": view_index,
                "renderer": "gsplat" if render is not None else "cpu-preview",
                "camera": camera_dict(cam),
                "references": [p.relative_to(folder).as_posix() for p in refs],
                "mask_pixels": int(mask.sum()),
            },
            indent=2,
        )
    )
    return RepairRequest(gap_id=gap.id, view=view, references=tuple(refs))
