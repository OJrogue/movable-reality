"""Scene manifest (scene.json): written by the pipeline, edited by the agent, read by the viewer.

Mirror of this schema for the viewer: web/src/scene.ts. Change both together.
Conventions: metres; world frame +Y up, floor at y = 0; `world_from_recon` maps the
reconstruction frame (OpenCV: +Y down, +Z forward) into the world frame.
"""

from __future__ import annotations

import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
Vec3 = tuple[float, float, float]

# Rotate 180 degrees about X: OpenCV (+Y down, +Z forward) to +Y up. Used until the floor
# plane is estimated; then the pipeline writes the real alignment.
OPENCV_TO_Y_UP: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class Transform(BaseModel):
    """Edit applied to an object about its bbox centre, in world metres."""

    translation: Vec3 = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0  # rotation about world +Y
    scale: float = 1.0


class SceneObject(BaseModel):
    id: str  # "<noun>#<sam id>", e.g. "armchair#2", or "gen#<n>" for inserted objects
    label: str
    ply: str  # relative to the scene folder
    completed_ply: str | None = None
    lod: str | None = None  # lane lod: Spark .RAD tree for this object, relative path
    bbox_min: Vec3  # world frame, before `transform`
    bbox_max: Vec3
    transform: Transform = Field(default_factory=Transform)
    visible: bool = True
    source: Literal["scan", "generated"] = "scan"
    prompt: str | None = None  # text prompt for generated objects

    @property
    def center(self) -> Vec3:
        return tuple((a + b) / 2 for a, b in zip(self.bbox_min, self.bbox_max, strict=True))  # type: ignore[return-value]

    @property
    def size(self) -> Vec3:
        s = self.transform.scale
        return tuple((b - a) * s for a, b in zip(self.bbox_min, self.bbox_max, strict=True))  # type: ignore[return-value]

    def world_aabb(self) -> tuple[Vec3, Vec3]:
        """Axis-aligned bbox after the edit (yaw about the centre, then scale, then move)."""
        cx, cy, cz = self.center
        hx, hy, hz = (v / 2 for v in self.size)
        yaw = math.radians(self.transform.yaw_deg)
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ez = c * hx + s * hz, s * hx + c * hz
        tx, ty, tz = self.transform.translation
        mn = (cx + tx - ex, cy + ty - hy, cz + tz - ez)
        mx = (cx + tx + ex, cy + ty + hy, cz + tz + ez)
        return mn, mx


class HistoryEntry(BaseModel):
    ts: str
    op: str
    object_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    instruction: str | None = None  # the user command this edit came from
    verified: bool | None = None  # verifier verdict, None if not checked


# ---------------------------------------------------------------- lane: bake
# Owned by the bake lane (AGENTS.md). Extend here; keep web/src/scene.ts in step.


class BakedPlane(BaseModel):
    """Opaque textured quad that replaces the Gaussians of a flat surface (floor, ceiling)."""

    id: str  # e.g. "floor"
    label: str
    mesh: str  # .glb relative to the scene folder, same frame as the PLYs (recon frame)
    removed_splats: int = 0  # Gaussians this plane replaced
    # Geometry in the world frame, written by scripts/bake_planes.py.
    normal: Vec3 | None = None  # unit, points into the room
    offset: float | None = None  # plane is {p : normal . p = offset}, metres
    corners: list[Vec3] = Field(default_factory=list)  # quad corners, world frame, in order
    texel_m: float | None = None  # texture texel size, metres
    texture_px: tuple[int, int] | None = None  # width, height
    fit_rms_m: float | None = None  # RMS distance of the fitted Gaussian centres to the plane
    filled_fraction: float = 0.0  # share of texels no Gaussian covered, filled by hole filling


class PlaneBake(BaseModel):
    """How the planes were baked. The bake sets Scene.background_ply to `background_ply` (the
    background without the replaced Gaussians) and keeps the original for re-bakes and undo."""

    source_background_ply: str  # original background, unchanged
    background_ply: str  # background without the baked Gaussians
    ids: str = "ids/gaussian_ids.npy"  # per-Gaussian ids that selected the plane Gaussians
    ids_ply: str = "splat/scene.ply"  # the PLY those ids index
    removed_splats: int = 0  # all planes together


# ---------------------------------------------------------------- lane: coverage
# Owned by the coverage lane (AGENTS.md). Extend here; keep web/src/scene.ts in step.

GapStatus = Literal["open", "recapture_requested", "recaptured", "generated", "accepted"]


class NextBestView(BaseModel):
    """Where to stand and point the phone for a top-up video of one gap (world frame)."""

    eye: Vec3  # phone position, metres; the user stands at (eye[0], 0, eye[2])
    target: Vec3  # point to aim the phone at
    visible: int = 0  # low-coverage gap points this view would see
    gain: float = 0.0  # share of the gap's low-coverage points it would see, 0..1


class Gap(BaseModel):
    """A region the capture did not observe well enough."""

    id: str
    label: str | None = None  # Muse Wonder's name for it, e.g. "behind the sofa"
    near_object: str | None = None  # SceneObject.id, when the gap belongs to one
    center: Vec3  # world frame, metres
    radius: float  # metres
    score: float  # 0 = well covered, 1 = never observed
    status: GapStatus = "open"
    next_best_views: list[NextBestView] = Field(default_factory=list)  # best first


class Coverage(BaseModel):
    gap_map: str | None = None  # per-Gaussian or voxel scores, relative path
    gaps: list[Gap] = Field(default_factory=list)
    generated_splats: int = 0  # splats with PLY property generated = 1, whole scene
    # Share of the VR-visible non-empty voxels (surfaces and unobserved space) inside gaps.
    gap_share: float | None = None


# ---------------------------------------------------------------- manifest


class Scene(BaseModel):
    version: int = SCHEMA_VERSION
    scene_id: str
    units: Literal["m"] = "m"
    world_from_recon: list[list[float]] = Field(
        default_factory=lambda: [r[:] for r in OPENCV_TO_Y_UP]
    )
    background_ply: str = "objects/background.ply"
    background_lod: str | None = None  # lane lod: Spark .RAD tree for the background
    objects: list[SceneObject] = Field(default_factory=list)
    planes: list[BakedPlane] = Field(default_factory=list)  # lane bake
    plane_bake: PlaneBake | None = None  # lane bake
    coverage: Coverage | None = None  # lane coverage
    history: list[HistoryEntry] = Field(default_factory=list)

    # ------------------------------------------------------------ io

    @classmethod
    def load(cls, path: Path) -> Scene:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        """Atomic write so the viewer never reads a half-written file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------ queries

    def get(self, object_id: str) -> SceneObject:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        known = ", ".join(o.id for o in self.objects) or "none"
        raise KeyError(f"No object '{object_id}'. Known objects: {known}")

    def overlaps(self, a: str, b: str, margin: float = 0.0) -> bool:
        (amin, amax), (bmin, bmax) = self.get(a).world_aabb(), self.get(b).world_aabb()
        return all(amin[i] < bmax[i] + margin and bmin[i] < amax[i] + margin for i in range(3))

    # ------------------------------------------------------------ edits

    def _log(self, op: str, object_id: str | None, args: dict[str, Any], instruction: str | None):
        self.history.append(
            HistoryEntry(
                ts=datetime.now(UTC).isoformat(timespec="seconds"),
                op=op,
                object_id=object_id,
                args=args,
                instruction=instruction,
            )
        )

    def move(
        self, object_id: str, dx: float, dy: float, dz: float, instruction: str | None = None
    ) -> None:
        obj = self.get(object_id)
        tx, ty, tz = obj.transform.translation
        obj.transform.translation = (tx + dx, ty + dy, tz + dz)
        self._log("move", object_id, {"dx": dx, "dy": dy, "dz": dz}, instruction)

    def rotate(self, object_id: str, yaw_deg: float, instruction: str | None = None) -> None:
        obj = self.get(object_id)
        obj.transform.yaw_deg = (obj.transform.yaw_deg + yaw_deg) % 360.0
        self._log("rotate", object_id, {"yaw_deg": yaw_deg}, instruction)

    def set_visible(self, object_id: str, visible: bool, instruction: str | None = None) -> None:
        self.get(object_id).visible = visible
        self._log("restore" if visible else "remove", object_id, {}, instruction)

    def mark_last_verified(self, verified: bool) -> None:
        if self.history:
            self.history[-1].verified = verified
