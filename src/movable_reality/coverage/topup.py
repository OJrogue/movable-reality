"""C2 recapture, full-rerun version: merge a short top-up video into a scene's frames.

The top-up frames get the next free indices after the original ones, so frame i of the
rebuilt sam_input.mp4 is still frames/{i:06d}.jpg and every earlier index keeps its image
(SAM masks and ids of the original frames stay valid). reconstruct.py then reruns
MapAnything on all frames and gsplat from scratch (PLAN "C2 recapture", step 1).

MapAnything's new reconstruction has its own arbitrary frame. `align_world` maps it back
onto the old one with a similarity fitted to the camera centres of the original frames
(Umeyama), so `world_from_recon`, the floor and the object boxes in scene.json stay where
they were and the rescored gaps are comparable with the old ones.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .cameras import Camera

SOURCE_KEY = "source"


def sharpness(path: Path) -> float:
    """Variance of a 4-neighbour Laplacian on a downscaled grey copy (as extract_frames.py)."""
    img = Image.open(path).convert("L")
    img.thumbnail((480, 480))
    a = np.asarray(img, dtype=np.float32)
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())


def select_sharp(scores: list[float], oversample: int, max_frames: int) -> list[int]:
    """The sharpest candidate of every window of `oversample` consecutive candidates,
    evenly thinned to at most `max_frames` (same rule as scripts/extract_frames.py)."""
    chosen: list[int] = []
    for start in range(0, len(scores), max(1, oversample)):
        window = range(start, min(start + oversample, len(scores)))
        chosen.append(max(window, key=lambda i: scores[i]))
    if len(chosen) > max_frames:
        keep = np.linspace(0, len(chosen) - 1, max_frames).round().astype(int)
        chosen = [chosen[k] for k in keep]
    return chosen


def video_source(video: Path) -> str:
    """Stable label of a top-up video (content hash), stored with each of its frames."""
    h = hashlib.sha256()
    with Path(video).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"topup:{h.hexdigest()[:12]}"


def frame_indices(frames: Path) -> list[int]:
    """Indices of frames/{i:06d}.jpg. Raises unless they are exactly 0..N-1."""
    idx = sorted(int(p.stem) for p in Path(frames).glob("*.jpg") if p.stem.isdigit())
    if idx != list(range(len(idx))):
        raise ValueError(f"{frames}: frame indices are not contiguous from 0")
    return idx


def fit_to_size(img: Image.Image, size: tuple[int, int], tol: float = 0.01) -> Image.Image:
    """Resize a top-up frame to the original frame size. reconstruct.py rescales every
    camera from the size of one frame, so all frames must share it. A different aspect
    ratio (portrait against landscape) is an error, never a silent crop."""
    w, h = img.size
    if abs(w / h - size[0] / size[1]) > tol * size[0] / size[1]:
        raise ValueError(
            f"top-up frame is {w}x{h}, original frames are {size[0]}x{size[1]}: "
            "film the top-up in the same orientation"
        )
    return img if img.size == size else img.resize(size, Image.LANCZOS)


def merge_frames(
    folder: Path, new_images: list[Path], new_meta: list[dict], source: str
) -> list[int]:
    """Copy `new_images` into folder/frames with the next free indices and append their
    entries (with `source`) to frames.json. Returns the new indices. Idempotent: when
    frames.json already lists frames from `source`, nothing is copied and their indices are
    returned."""
    folder = Path(folder)
    frames = folder / "frames"
    meta_path = folder / "frames.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"frames": []}
    done = [f["index"] for f in meta["frames"] if f.get(SOURCE_KEY) == source]
    if done:
        return done
    if len(new_images) != len(new_meta):
        raise ValueError("one metadata entry per new image")
    existing = frame_indices(frames)
    if len(existing) != len(meta["frames"]):
        raise ValueError(f"{meta_path} lists {len(meta['frames'])} frames, found {len(existing)}")
    size = Image.open(frames / f"{existing[0]:06d}.jpg").size if existing else None
    start = len(existing)
    out = []
    for k, (src, m) in enumerate(zip(new_images, new_meta, strict=True)):
        i = start + k
        dst = frames / f"{i:06d}.jpg"
        img = Image.open(src)
        if size is not None and img.size != size:
            fit_to_size(img.convert("RGB"), size).save(dst, quality=95)
        else:
            shutil.copyfile(src, dst)
        meta["frames"].append({**m, "index": i, SOURCE_KEY: source})
        out.append(i)
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(meta_path)
    return out


def sam_clip_cmd(folder: Path, fps: float) -> list[str]:
    """ffmpeg command that rebuilds sam_input.mp4 from all frames (same settings as
    scripts/extract_frames.py), so SAM frame i is frames/{i:06d}.jpg again."""
    folder = Path(folder)
    return [
        "ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
        "-i", str(folder / "frames" / "%06d.jpg"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
        str(folder / "sam_input.mp4"),
    ]  # fmt: skip


# ---------------------------------------------------------------- frame alignment


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |dst - (s R src + t)|^2 over point pairs (N, 3)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if len(src) < 3:
        raise ValueError("need at least 3 point pairs")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var)
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_world(
    prev: list[Camera], new: list[Camera], world_from_prev: np.ndarray
) -> tuple[np.ndarray, dict]:
    """world_from_recon for the new reconstruction: world_from_prev @ prev_from_new, with
    prev_from_new fitted to the centres of the cameras both models share (by image name).
    Returns the 4x4 matrix and fit statistics (pairs, scale, RMS residual in prev units)."""
    by_name = {c.name: c for c in prev}
    pairs = [(by_name[c.name].center, c.center) for c in new if c.name in by_name]
    if len(pairs) < 3:
        raise ValueError(f"only {len(pairs)} cameras shared with the previous model")
    p = np.array([a for a, _ in pairs])
    q = np.array([b for _, b in pairs])
    s, R, t = umeyama(q, p)
    prev_from_new = np.eye(4)
    prev_from_new[:3, :3] = s * R
    prev_from_new[:3, 3] = t
    rms = float(np.sqrt(((p - (s * q @ R.T + t)) ** 2).sum(1).mean()))
    W = np.asarray(world_from_prev, np.float64) @ prev_from_new
    return W, {"pairs": len(pairs), "scale": s, "rms": rms}
