// Deterministic walk paths for bench mode. Pure functions of time, so a run on the desktop
// and a run in the headset see the same sequence of rig poses.
//
// The rig is the camera's parent. On the desktop the camera sits 1.6 m above it; in XR the
// head pose (local-floor) is added on top, so the user stands still and the rig moves them.
// Poses are in the world frame (+Y up, floor at y = 0), inside the synthetic bench room
// (scripts/bench_make_splats.py: x in [-3, 3], z in [-2.5, 2.5]). Speeds stay under
// 0.5 m/s and turns under 30 deg/s, so a standing user is not thrown around.

export interface RigPose {
  x: number;
  z: number;
  /** Rotation about +Y in radians; 0 looks along -Z. */
  yaw: number;
}

export interface BenchPath {
  name: string;
  description: string;
  pose(tS: number): RigPose;
}

const TAU = 2 * Math.PI;

/** Yaw that looks from (x, z) towards (tx, tz); 0 looks along -Z. */
export function yawTowards(x: number, z: number, tx: number, tz: number): number {
  return Math.atan2(-(tx - x), -(tz - z));
}

/** Ellipse round the room centre, looking at the centre: most of the room in view. */
const orbit: BenchPath = {
  name: "orbit",
  description: "ellipse 1.8 x 1.2 m round the centre, 40 s per lap, facing the centre",
  pose(t) {
    const a = (TAU * t) / 40;
    const x = 1.8 * Math.sin(a);
    const z = 1.2 * Math.cos(a);
    return { x, z, yaw: yawTowards(x, z, 0, 0) };
  },
};

/**
 * Walk through the room: an off-centre loop that passes close to the furniture and the
 * walls (large splats on screen) while the view sweeps slowly across the room. Yaw is its
 * own slow function of time rather than tied to the walking direction, which keeps turns
 * gentle for a standing user.
 */
const walk: BenchPath = {
  name: "walk",
  description: "loop 4.0 x 2.4 m every 60 s; view sweeps +-75 deg about -Z every 45 s",
  pose(t) {
    const w = TAU / 60;
    const x = 2.0 * Math.sin(w * t);
    const z = 1.2 * Math.sin(2 * w * t);
    const yaw = (75 * Math.PI) / 180 * Math.sin((TAU * t) / 45);
    return { x, z, yaw };
  },
};

/** Stand at the centre and turn on the spot: every wall in turn, sort changes each frame. */
const pan: BenchPath = {
  name: "pan",
  description: "stand at the centre, turn 360 deg every 30 s",
  pose(t) {
    return { x: 0, z: 0, yaw: (-TAU * t) / 30 };
  },
};

/** No motion: render cost without re-sorting or LOD re-traversal. */
const still: BenchPath = {
  name: "still",
  description: "stand 1.5 m behind the centre looking at it",
  pose() {
    return { x: 0, z: 1.5, yaw: 0 };
  },
};

export const PATHS: Record<string, BenchPath> = { walk, orbit, pan, still };
export const DEFAULT_PATH = "walk";

export function getPath(name: string | null): BenchPath | null {
  return PATHS[name ?? DEFAULT_PATH] ?? null;
}
