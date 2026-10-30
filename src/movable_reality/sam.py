"""SAM 3.1 on the Meta Model API: segment an image or a video for one noun, with disk caching.

Each request takes one short concrete noun ("armchair"). The model returns every match as a
box plus a box-local mask; in video, an instance keeps its object id across frames.
Docs: https://dev.meta.ai/docs/sam/overview
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import openai
from meta_sam_parser import (
    SegmentationMask,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from PIL import Image

from .clients import meta_async_client
from .config import CACHE_DIR, Models

MaskEncoding = Literal["one_bit", "lossless"]


@dataclass(frozen=True)
class Instance:
    """One object in one frame."""

    object_id: str  # SAM id, stable across frames within one request
    frame_index: int  # 0 for images
    box: tuple[int, int, int, int]  # left, top, right, bottom (half-open), source pixels
    mask: np.ndarray  # bool, box-local raster; paste with paste_mask()
    confidence: float | None


def paste_mask(inst: Instance, width: int, height: int) -> np.ndarray:
    """Return a full-frame bool mask of shape (height, width) for a box-local instance mask."""
    left, top, right, bottom = inst.box
    box_w, box_h = right - left, bottom - top
    local = inst.mask
    if local.shape != (box_h, box_w):
        img = Image.fromarray(local.astype(np.uint8) * 255)
        local = np.asarray(img.resize((box_w, box_h), Image.Resampling.NEAREST)) > 127
    full = np.zeros((height, width), dtype=bool)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(right, width), min(bottom, height)
    if x1 > x0 and y1 > y0:
        full[y0:y1, x0:x1] = local[y0 - top : y1 - top, x0 - left : x1 - left]
    return full


# ---------------------------------------------------------------- caching helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(kind: str, key: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
    return CACHE_DIR / "sam" / f"{kind}_{digest}.json"


def _record_to_json(r: SegmentationMaskRecord) -> dict[str, Any]:
    return {
        "object_id": r.object_id,
        "frame_index": 0 if r.frame is None else r.frame.frame_index,
        "box": [int(r.bounds.left), int(r.bounds.top), int(r.bounds.right), int(r.bounds.bottom)],
        "mask": {
            "encoding": r.mask.encoding,
            "payload": r.mask.payload,
            "width": r.mask.width,
            "height": r.mask.height,
        },
        "confidence": r.confidence,
    }


def _instance_from_json(d: dict[str, Any]) -> Instance:
    m = d["mask"]
    raster = decode_mask_to_raster(
        SegmentationMask(
            encoding=m["encoding"], payload=m["payload"], width=m["width"], height=m["height"]
        )
    )
    mask = np.frombuffer(raster, dtype=np.uint8).reshape(m["height"], m["width"]).astype(bool)
    return Instance(
        object_id=str(d["object_id"]),
        frame_index=int(d["frame_index"]),
        box=tuple(d["box"]),  # type: ignore[arg-type]
        mask=mask,
        confidence=d.get("confidence"),
    )


def _load_cached(cache: Path) -> list[Instance]:
    """Instances from a cached response. meta_sam_parser decodes and re-encodes every mask in
    pure Python (~10 min for blender01's 13 nouns), so the decoded masks are kept bit-packed
    next to the JSON and reused."""
    packed = cache.with_suffix(".npz")
    if packed.exists() and packed.stat().st_mtime >= cache.stat().st_mtime:
        z = np.load(packed)
        meta = json.loads(str(z["meta"]))
        bits, out, at = z["bits"], [], 0
        for d in meta:
            h, w = d["shape"]
            n = (h * w + 7) // 8
            mask = np.unpackbits(bits[at : at + n])[: h * w].reshape(h, w).astype(bool)
            at += n
            out.append(
                Instance(str(d["object_id"]), int(d["frame_index"]), tuple(d["box"]), mask,
                         d.get("confidence"))
            )  # fmt: skip
        return out
    out = [_instance_from_json(d) for d in json.loads(cache.read_text())]
    meta = [
        {"object_id": i.object_id, "frame_index": i.frame_index, "box": list(i.box),
         "confidence": i.confidence, "shape": list(i.mask.shape)}
        for i in out
    ]  # fmt: skip
    bits = [np.packbits(i.mask.ravel()) for i in out]
    np.savez(packed, meta=json.dumps(meta), bits=np.concatenate(bits) if bits else np.zeros(0))
    return out


# ---------------------------------------------------------------- API calls


async def _segment(
    media_part: dict[str, Any], noun: str, media: Literal["image", "video"], encoding: MaskEncoding
) -> list[dict[str, Any]]:
    client = meta_async_client()
    events = await client.responses.create(
        model=Models.SAM,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": noun}, media_part],
            }
        ],
        stream=True,
        metadata={"mask_encoding": encoding},
    )
    fmt = image_segmentation_format() if media == "image" else video_segmentation_format()
    result = await parse_responses_stream(events, fmt).final_result()
    if result.outcome.status != "completed":
        raise RuntimeError(f"SAM 3.1 response incomplete: {result.outcome}")
    # Keep only the latest revision of each (frame, object) mask.
    latest: dict[tuple[int, str], SegmentationMaskRecord] = {}
    for r in result.records:
        if r.kind == "mask":
            frame = 0 if r.frame is None else r.frame.frame_index
            latest[(frame, r.object_id)] = r
    return [_record_to_json(r) for _, r in sorted(latest.items())]


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


async def segment_image(
    image: Path, noun: str, encoding: MaskEncoding = "one_bit", force: bool = False
) -> list[Instance]:
    image = Path(image)
    key = {"model": Models.SAM, "noun": noun, "sha": sha256_file(image), "enc": encoding}
    cache = _cache_path("image", key)
    if cache.exists() and not force:
        return [_instance_from_json(d) for d in json.loads(cache.read_text())]
    part = {"type": "input_image", "image_url": _image_data_url(image)}
    records = await _segment(part, noun, "image", encoding)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records))
    return [_instance_from_json(d) for d in records]


async def upload_video(video: Path, force: bool = False) -> str:
    """Upload once with purpose=user_data and remember the file id by content hash."""
    video = Path(video)
    index_path = CACHE_DIR / "files.json"
    index: dict[str, str] = json.loads(index_path.read_text()) if index_path.exists() else {}
    sha = sha256_file(video)
    if sha in index and not force:
        return index[sha]
    client = meta_async_client()
    with video.open("rb") as f:
        uploaded = await client.files.create(file=f, purpose="user_data")
    index[sha] = uploaded.id
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2))
    return uploaded.id


async def segment_video(
    video: Path, noun: str, encoding: MaskEncoding = "one_bit", force: bool = False
) -> list[Instance]:
    """Track every instance of `noun` through an MP4/MOV (H.264/H.265), max 15,000 frames."""
    video = Path(video)
    key = {"model": Models.SAM, "noun": noun, "sha": sha256_file(video), "enc": encoding}
    cache = _cache_path("video", key)
    if cache.exists() and not force:
        return _load_cached(cache)
    file_id = await upload_video(video)
    try:
        records = await _segment(
            {"type": "input_video", "file_id": file_id}, noun, "video", encoding
        )
    except (openai.NotFoundError, openai.BadRequestError):
        # The remembered file id may have expired or belong to another key: upload again once.
        file_id = await upload_video(video, force=True)
        records = await _segment(
            {"type": "input_video", "file_id": file_id}, noun, "video", encoding
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records))
    return _load_cached(cache)
