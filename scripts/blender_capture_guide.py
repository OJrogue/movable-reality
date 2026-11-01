"""Filming guide videos from the synthetic room: how to walk and point the phone.

    uv run python scripts/blender_capture_guide.py                 # both walks
    uv run python scripts/blender_capture_guide.py --walk main
    uv run python scripts/blender_capture_guide.py --scene blender02
    uv run python scripts/blender_capture_guide.py --scene blender04 --layout   # plan PNG only

Inputs come from scripts/blender_room.py: the rendered frames (data/raw/<scene>/<walk>/rgb)
and the ground-truth cameras and objects (data/raw/<scene>_gt/). Each output frame shows
the phone's view next to a floor plan with the room outline (room_outline_xz in objects.json,
else the room_size_m box), the path, the phone position and the direction it points, plus
the current instruction, phone height, walking speed and turn rate.
Writes data/raw/<scene>_guide_main.mp4 (training capture) and _test.mp4 (test walk).
--layout writes data/raw/<scene>_layout.png instead: the floor plan with both walks, the
furniture labels, columns, ceiling heights and any windows and doors. It needs no render.
Without objects.json it takes the furniture footprints from the scene definition in
blender_room.py.
CPU only, no API calls.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from movable_reality.config import DATA_DIR

W, H = 1920, 1080
VIDEO_W, VIDEO_H = 1280, 720
PLAN = 640  # floor plan panel, square
FPS = 30

# Instruction per time span, seconds, per scene. Matches the walk functions in
# scripts/blender_room.py (_main_eye_target, _test_eye_target, _main2_eye_target, ...).
PHASES_01 = {
    "main": [
        (0.0, 25.0, "1/5  Chest-height loop",
         ["Walk slowly along the walls, about 0.5 m from them.",
          "Phone landscape at chest height (1.5 m),",
          "pointing into the room and a little down (about 20 deg).",
          "One loop in about 25 s, slow pace (0.4 m/s), turn gently."]),
        (25.0, 27.0, "2/5  Crouch",
         ["Lower the phone to knee height.",
          "Keep filming, keep the phone pointed inwards."]),
        (27.0, 38.0, "3/5  Knee-height loop",
         ["A smaller loop closer to the furniture at 0.6 m,",
          "phone nearly level, aimed at the middle of the room,",
          "so the sides and the space under tables are seen."]),
        (38.0, 40.0, "4/5  Stand up",
         ["Stand up again at the start of the first loop.",
          "Keep filming, tilt the phone a little up."]),
        (40.0, 56.0, "5/5  Upward loop",
         ["Walk the first loop again, phone at 1.5 m,",
          "pointing into the room and about 15 deg UP,",
          "so the tops of the walls and the ceiling are seen.",
          "Keep walking: turning on the spot gives no 3D."]),
    ],
    "test": [
        (0.0, 20.0, "Test walk (never used for training)",
         ["A figure eight through the middle of the room at 1.2 m.",
          "Turn the phone slowly so it sweeps every piece of",
          "furniture from angles the main loop did not use.",
          "Film it right after the main capture, same light."]),
    ],
}  # fmt: skip
PHASES_02 = {
    "main": [
        (0.0, 25.0, "1/3  Chest-height loop along the L",
         ["Walk slowly round the L, about 0.5 m from walls and furniture,",
          "past the desk and armchair, up into the bed alcove and back.",
          "Phone landscape at chest height (1.5 m), pointing across the room",
          "and a little down (about 20 deg). Slow pace (0.4 m/s)."]),
        (25.0, 27.0, "2/3  Crouch",
         ["Lower the phone to knee height.",
          "Keep filming, keep the phone pointed across the room."]),
        (27.0, 40.0, "3/3  Knee-height loop",
         ["A smaller loop in the open middle of the study at 0.6 m,",
          "phone nearly level, aimed across the loop,",
          "so the sides and the space under the furniture are seen."]),
    ],
    "test": [
        (0.0, 20.0, "Test walk (never used for training)",
         ["A loop at 1.2 m through the study and into the alcove.",
          "Turn the phone slowly so it sweeps every piece of",
          "furniture from angles the main loop did not use.",
          "Film it right after the main capture, same light."]),
    ],
}  # fmt: skip
PHASES_03 = {
    "main": [
        (0.0, 25.0, "1/3  Chest-height loop round the column",
         ["Walk slowly round the column through the open middle, about 0.5 m",
          "from walls and furniture, past the angled wall and the bay window.",
          "Phone landscape at chest height (1.5 m), pointing across the room",
          "and a little down (about 20 deg). Slow pace (0.4 m/s)."]),
        (25.0, 27.0, "2/3  Crouch",
         ["Lower the phone to knee height.",
          "Keep filming, keep the phone pointed across the room."]),
        (27.0, 40.0, "3/3  Knee-height loop",
         ["A smaller loop round the column at 0.6 m, phone nearly level,",
          "aimed across the loop, so the back of the column, the bay",
          "and the space under the furniture are seen."]),
    ],
    "test": [
        (0.0, 20.0, "Test walk (never used for training)",
         ["A loop at 1.2 m past the column and the bay.",
          "Turn the phone slowly so it sweeps every piece of",
          "furniture from angles the main loop did not use.",
          "Film it right after the main capture, same light."]),
    ],
}  # fmt: skip
PHASES_04 = {
    "main": [
        (0.0, 25.0, "1/3  Chest-height loop under the flat ceiling",
         ["Walk slowly round the middle of the attic, where you can stand,",
          "and step into the mouth of the dormer. Mind your head under the slopes.",
          "Phone landscape at chest height (1.5 m), pointing across the room",
          "and a little down (about 20 deg). Slow pace (0.4 m/s)."]),
        (25.0, 27.0, "2/3  Crouch",
         ["Lower the phone to knee height.",
          "Keep filming, keep the phone pointed across the room."]),
        (27.0, 43.0, "3/3  Knee-height loop under the slopes",
         ["A loop at 0.6 m that reaches in under both slopes and past the dormer,",
          "phone nearly level, aimed across the loop,",
          "so the knee walls and the space under the furniture are seen."]),
    ],
    "test": [
        (0.0, 20.0, "Test walk (never used for training)",
         ["A loop at 1.2 m through the middle of the attic.",
          "Turn the phone slowly so it sweeps every piece of",
          "furniture from angles the main loop did not use.",
          "Film it right after the main capture, same light."]),
    ],
}  # fmt: skip
PHASES = {"blender01": PHASES_01, "blender02": PHASES_02, "blender03": PHASES_03,
          "blender04": PHASES_04, "blender05": PHASES_01}  # fmt: skip
TIPS = [
    "Before filming: lock focus and exposure (long-press on iPhone), 1080p60 or 4K30,",
    "lamps on, curtains still, no people or pets, screens off. Move slowly, no fast turns.",
]
TITLES = {"main": "Training capture", "test": "Test walk"}


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Helvetica.ttc", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):  # fmt: skip
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def poses(gt: Path, walk: str, scene: str = "blender01") -> np.ndarray:
    """(N, 4, 4) cam_to_world, OpenCV camera, world +Y up. From the GT file, or, while a
    render is still running, from the walk definition in blender_room.py."""
    path = gt / walk / "cameras.json"
    if path.exists():
        cams = json.loads(path.read_text())
        return np.array([f["cam_to_world"] for f in cams["frames"]], dtype=np.float64)
    sys.path.insert(0, str(Path(__file__).parent))
    import blender_room as br  # noqa: PLC0415 - pure helpers, no bpy needed

    n = int(round(br.SCENES[scene]["walks"][walk] * br.FPS))
    return np.array(
        [br.cam_to_world_cv(br.camera_matrix(walk, i / br.FPS, scene)) for i in range(n)]
    )


def motion(c2w: np.ndarray, i: int, span: int = 8) -> tuple[float, float, float, float]:
    """Phone height (m), walking speed (m/s), turn rate (deg/s) and tilt (deg, positive =
    pointing down) around frame i."""
    a, b = max(0, i - span), min(len(c2w) - 1, i + span)
    dt = max(b - a, 1) / FPS
    pa, pb = c2w[a, :3, 3], c2w[b, :3, 3]
    speed = float(np.linalg.norm((pb - pa)[[0, 2]]) / dt)
    fa, fb = c2w[a, :3, 2], c2w[b, :3, 2]
    ya, yb = math.atan2(fa[0], fa[2]), math.atan2(fb[0], fb[2])
    turn = abs((yb - ya + math.pi) % (2 * math.pi) - math.pi)
    f = c2w[i, :3, 2]
    tilt = math.degrees(math.atan2(-f[1], math.hypot(f[0], f[2])))
    return float(c2w[i, 1, 3]), speed, math.degrees(turn) / dt, tilt


def room_outline(objs: dict) -> list[tuple[float, float]]:
    """Room outline in world (x, z) from objects.json: room_outline_xz if present (L-shaped
    rooms), else the room_size_m box centred on the origin (blender01)."""
    if "room_outline_xz" in objs:
        return [(float(x), float(z)) for x, z in objs["room_outline_xz"]]
    rx, rz = objs["room_size_m"][0] / 2, objs["room_size_m"][2] / 2
    return [(-rx, -rz), (rx, -rz), (rx, rz), (-rx, rz)]


class Plan:
    """Top-down floor plan, north (world -z) up, in a size x size panel. Polygon rooms add
    ceiling zones (every zone edge drawn as a thin line) and columns."""

    def __init__(
        self,
        outline: list[tuple[float, float]],
        objects: list[dict],
        size: int = PLAN,
        zones: list[dict] | None = None,
        columns: list[list[float]] | None = None,
        openings: list[dict] | None = None,
    ):
        pts = np.asarray(outline, dtype=float)
        lo, hi = pts.min(0), pts.max(0)
        self.size = size
        self.centre = (lo + hi) / 2
        self.s = (size - 80 * size / PLAN) / float((hi - lo).max())
        self.outline = [tuple(p) for p in pts]
        self.objects = objects
        self.zones = zones or []
        self.columns = columns or []
        self.openings = openings or []

    def px(self, x: float, z: float) -> tuple[float, float]:
        return (self.size / 2 + (x - self.centre[0]) * self.s,
                self.size / 2 + (z - self.centre[1]) * self.s)  # fmt: skip

    def base(self, path: np.ndarray) -> Image.Image:
        img = Image.new("RGB", (self.size, self.size), (24, 26, 30))
        d = ImageDraw.Draw(img)
        poly = [self.px(x, z) for x, z in self.outline]
        d.polygon(poly, fill=(52, 48, 44))
        for zone in self.zones:  # ceiling zone edges (slope starts, dormer, bay)
            zp = [self.px(x, z) for x, z in zone["outline_xz"]]
            d.line([*zp, zp[0]], fill=(88, 84, 78), width=2)
        for x0, z0, x1, z1 in self.columns:
            d.rectangle([*self.px(x0, z0), *self.px(x1, z1)], fill=(220, 210, 195))
        d.line([*poly, poly[0]], fill=(220, 210, 195), width=6, joint="curve")
        for op in self.openings:  # windows teal, doors brown, drawn over the wall line
            colour = (70, 200, 190) if op["kind"] == "window" else (190, 130, 70)
            d.line([self.px(*p) for p in op["outline_xz"]], fill=colour, width=12)
        f = font(15)
        big = [o for o in self.objects if o.get("support") is None]
        small = [o for o in self.objects if o.get("support") is not None]
        for o in big + small:  # items on furniture drawn on top of it
            (x0, _, z0), (x1, _, z1) = o["bbox_min"], o["bbox_max"]
            d.rectangle([*self.px(x0, z0), *self.px(x1, z1)],
                        fill=(95, 90, 82) if o in big else (130, 120, 105),
                        outline=(160, 150, 135))  # fmt: skip
        for o in big:  # labels last, so no box covers them
            (x0, _, z0), (x1, _, z1) = o["bbox_min"], o["bbox_max"]
            cx, cz = self.px((x0 + x1) / 2, (z0 + z1) / 2)
            d.text((cx, cz), o["noun"], font=f, fill=(245, 240, 230), anchor="mm",
                   stroke_width=2, stroke_fill=(40, 38, 34))  # fmt: skip
        pts = [self.px(p[0], p[2]) for p in path]
        d.line(pts, fill=(70, 140, 220), width=3)
        d.text((self.size / 2, 18), "floor plan (north up)", font=font(16), fill=(170, 170, 170),
               anchor="mm")  # fmt: skip
        return img

    def layout(self, walks: dict[str, np.ndarray], title: str) -> Image.Image:
        """Plan with every walk's eye path (world positions, (N, 3)), ceiling heights per
        zone and a 1 m scale bar. Main walk: blue above 1 m, violet below; test: green."""
        img = self.base(np.zeros((0, 3)))
        d = ImageDraw.Draw(img)
        f = font(int(self.size / 40))
        for zone in self.zones:
            pts = np.asarray(zone["outline_xz"], dtype=float)
            if (pts.max(0) - pts.min(0)).min() < 0.6:
                continue  # too narrow to label
            hx, hz, h0 = zone["height"]
            hs = [hx * x + hz * z + h0 for x, z in pts]
            label = f"{min(hs):.1f} m" if max(hs) - min(hs) < 1e-6 else (
                f"{min(hs):.1f}-{max(hs):.1f} m")  # fmt: skip
            cx, cz = self.px(*pts.mean(0))
            d.text((cx, cz + self.size / 30), f"ceiling {label}", font=f, fill=(150, 145, 135),
                   anchor="mm")  # fmt: skip
        colours = {"test": (90, 200, 120)}
        for walk, eyes in sorted(walks.items(), reverse=True):  # test first, main on top
            for a, b in zip(eyes[:-1], eyes[1:], strict=True):
                c = colours.get(walk) or ((90, 150, 235) if a[1] > 1.0 else (190, 110, 230))
                d.line([self.px(a[0], a[2]), self.px(b[0], b[2])], fill=c, width=4)
            sx, sz = self.px(eyes[0][0], eyes[0][2])
            d.ellipse([sx - 8, sz - 8, sx + 8, sz + 8], outline="white", width=3)
        x0, y0 = self.size * 0.06, self.size - self.size * 0.05
        d.line([(x0, y0), (x0 + self.s, y0)], fill="white", width=4)
        d.text((x0 + self.s / 2, y0 - 6), "1 m", font=f, fill="white", anchor="mb")
        d.rectangle([0, 0, self.size, self.size * 0.06], fill=(24, 26, 30))
        d.text((self.size / 2, self.size * 0.03), title, font=font(int(self.size / 32)),
               fill=(230, 230, 230), anchor="mm")  # fmt: skip
        d.text((self.size - 12, self.size - 12),
               "blue: main walk at chest height   violet: main walk at knee height   "
               "green: test walk   circle: start", font=font(int(self.size / 55)),
               fill=(170, 170, 170), anchor="rd")  # fmt: skip
        return img

    def frame(self, base: Image.Image, c2w: np.ndarray, i: int) -> Image.Image:
        img = base.copy().convert("RGBA")
        d = ImageDraw.Draw(img)
        done = [self.px(p[0], p[2]) for p in c2w[: i + 1, :3, 3]]
        if len(done) > 1:
            d.line(done, fill=(255, 190, 60), width=5)
        x, z = c2w[i, 0, 3], c2w[i, 2, 3]
        fwd = c2w[i, :3, 2]
        yaw = math.atan2(fwd[2], fwd[0])
        cx, cz = self.px(x, z)
        r, half = 1.2 * self.s, math.radians(34)  # view cone, ~69 deg horizontal FOV
        cone = [(cx, cz)] + [(cx + r * math.cos(yaw + a), cz + r * math.sin(yaw + a))
                             for a in np.linspace(-half, half, 12)]  # fmt: skip
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(cone, fill=(255, 190, 60, 80), outline=(255, 210, 120, 255))
        img = Image.alpha_composite(img, layer)
        d = ImageDraw.Draw(img)
        d.ellipse(
            [cx - 11, cz - 11, cx + 11, cz + 11], fill=(255, 120, 40), outline="white", width=2
        )
        d.text((self.size / 2, self.size - 16),
               "orange: phone, view and path so far   blue: whole path",
               font=font(15), fill=(160, 160, 160), anchor="mm")  # fmt: skip
        return img.convert("RGB")


def gauge(d: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str, frac: float, warn: bool):
    d.text((x, y), label, font=font(20), fill=(170, 170, 170))
    d.text((x, y + 26), value, font=font(34), fill=(255, 120, 90) if warn else (240, 240, 240))
    d.rectangle([x, y + 72, x + 260, y + 80], fill=(60, 60, 60))
    d.rectangle([x, y + 72, x + 260 * min(max(frac, 0), 1), y + 80],
                fill=(255, 120, 90) if warn else (90, 180, 120))  # fmt: skip


def render(
    walk: str, raw: Path, gt: Path, out: Path, limit: int | None = None, scene: str = "blender01"
) -> Path:
    c2w = poses(gt, walk, scene)
    path_all = c2w[:, :3, 3]
    if limit:
        c2w = c2w[:limit]
    rgb = raw / walk / "rgb"
    frames = sorted(rgb.glob("*.png"))
    if len(frames) < len(c2w):
        raise SystemExit(f"{rgb}: {len(frames)} of {len(c2w)} frames rendered")
    objs = json.loads((gt / "objects.json").read_text())
    plan = Plan(room_outline(objs), [o for o in objs["objects"] if "bbox_min" in o],
                zones=objs.get("ceiling_zones"), columns=objs.get("columns_xz"),
                openings=objs.get("openings"))  # fmt: skip
    base = plan.base(path_all)
    phases = PHASES[scene][walk]
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    total = len(path_all) / FPS
    for i in range(len(c2w)):
        t = i / FPS
        canvas = Image.new("RGB", (W, H), (16, 17, 20))
        view = Image.open(frames[i]).convert("RGB").resize((VIDEO_W, VIDEO_H), Image.BILINEAR)
        canvas.paste(view, (0, 0))
        canvas.paste(plan.frame(base, c2w, i), (VIDEO_W, 0))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, 470, 50], fill=(0, 0, 0))
        d.text((14, 10), f"{TITLES[walk]}, {total:.0f} s   {t:5.1f} s", font=font(26),
               fill=(255, 255, 255))  # fmt: skip
        d.text((VIDEO_W - 14, VIDEO_H - 14), "what the phone sees", font=font(20),
               fill=(255, 255, 255), anchor="rd")  # fmt: skip
        _, _, title, lines = next((p for p in phases if p[0] <= t < p[1]), phases[-1])
        d.text((30, VIDEO_H + 24), title, font=font(38), fill=(255, 200, 90))
        for k, line in enumerate(lines):
            d.text((30, VIDEO_H + 80 + 34 * k), line, font=font(27), fill=(235, 235, 235))
        for k, line in enumerate(TIPS):
            d.text((30, H - 64 + 26 * k), line, font=font(19), fill=(140, 140, 140))
        height, speed, turn, tilt = motion(c2w, i)
        for k, (label, value, frac, warn) in enumerate([
            ("phone height", f"{height:.2f} m", height / 2.0, False),
            ("tilt down", f"{tilt:.0f} deg", tilt / 45, False),
            ("walking speed", f"{speed:.2f} m/s", speed / 1.0, speed > 0.6),
            ("turning", f"{turn:.0f} deg/s", turn / 60, turn > 40),
        ]):  # fmt: skip
            gauge(d, VIDEO_W + 30 + 310 * (k % 2), PLAN + 30 + 200 * (k // 2), label, value,
                  frac, warn)  # fmt: skip
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"{walk}: wrote {out} ({len(c2w)} frames, {total:.0f} s)")
    return out


def scene_layout(scene: str, gt: Path, out: Path, size: int = 1400) -> Path:
    """Top-down layout PNG of a scene: outline, furniture, columns, ceiling heights and both
    walks. Uses objects.json when a build exists, else the scene definition."""
    sys.path.insert(0, str(Path(__file__).parent))
    import blender_room as br  # noqa: PLC0415 - pure helpers, no bpy needed

    cfg = br.SCENES[scene]
    path = gt / "objects.json"
    if path.exists():
        objs = json.loads(path.read_text())
        objects = [o for o in objs["objects"] if "bbox_min" in o]
    else:  # footprints from the definition; Blender (x0, y0, x1, y1) -> world x, z
        rx, ry, rh = cfg["room"]
        objs = {"room_size_m": [rx, rh, ry]}
        if cfg["outline"] is not None:
            objs["room_outline_xz"] = [[x, -y] for x, y in cfg["outline"]]
        if cfg.get("zones"):
            objs.update(br.world_room_extras(scene))
        if cfg.get("openings"):
            objs["openings"] = br.world_openings(scene)
        foot = br.furniture_footprints(scene) if scene != "blender01" else {}
        objects = [
            {"noun": noun, "support": None, "bbox_min": [b[0], 0.0, -b[3]],
             "bbox_max": [b[2], 1.0, -b[1]]}
            for name, noun, *_ in cfg["objects"] if (b := foot.get(name))
        ]  # fmt: skip
    plan = Plan(room_outline(objs), objects, size=size, zones=objs.get("ceiling_zones"),
                columns=objs.get("columns_xz"), openings=objs.get("openings"))  # fmt: skip
    walks = {walk: poses(gt, walk, scene)[:, :3, 3] for walk in cfg["walks"]}
    img = plan.layout(walks, f"{scene}: layout (north up), {len(objects)} objects")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene", default="blender01", choices=sorted(PHASES))
    ap.add_argument("--walk", default="main,test")
    ap.add_argument("--raw", type=Path, default=None, help="default data/raw/<scene>")
    ap.add_argument("--limit", type=int, default=None, help="first N frames only (preview)")
    ap.add_argument("--layout", action="store_true", help="write <scene>_layout.png only")
    args = ap.parse_args()
    raw = args.raw or DATA_DIR / "raw" / args.scene
    gt = raw.parent / (raw.name + "_gt")
    if args.layout:
        scene_layout(args.scene, gt, raw.parent / f"{raw.name}_layout.png")
        return 0
    for walk in args.walk.split(","):
        out = raw.parent / f"{raw.name}_guide_{walk}.mp4"
        render(walk, raw, gt, out, args.limit, args.scene)
    return 0


if __name__ == "__main__":
    sys.exit(main())
