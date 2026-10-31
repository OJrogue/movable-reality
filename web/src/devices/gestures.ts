// Lane devices: touch gesture maths. Pure functions, tested in gestures.test.ts.
//
// One finger looks around (yaw and pitch of the rig, the scene follows the finger). Two
// fingers pan (the midpoint moves the rig in the screen plane, the scene follows the
// fingers) and pinch (spreading the fingers moves the rig forward along the view).

export interface Point {
  x: number;
  y: number;
}

export interface TouchTuning {
  /** Radians of yaw or pitch per CSS pixel of one-finger drag. */
  lookRadPerPx: number;
  /** Metres of rig movement per CSS pixel of two-finger pan. */
  panMPerPx: number;
  /** Metres forward per unit of ln(finger distance ratio). */
  pinchM: number;
  /** Largest pitch either way (radians); straight up or down flips the yaw. */
  maxPitch: number;
}

export const DEFAULT_TOUCH: Readonly<TouchTuning> = {
  lookRadPerPx: 0.005,
  panMPerPx: 0.004,
  pinchM: 2.5,
  maxPitch: (85 * Math.PI) / 180,
};

/** Yaw and pitch after a one-finger drag of (dx, dy) pixels. */
export function look(
  yaw: number,
  pitch: number,
  dx: number,
  dy: number,
  t: TouchTuning = DEFAULT_TOUCH,
): { yaw: number; pitch: number } {
  const p = pitch + dy * t.lookRadPerPx;
  return { yaw: yaw + dx * t.lookRadPerPx, pitch: Math.max(-t.maxPitch, Math.min(t.maxPitch, p)) };
}

export interface TwoFingerStep {
  /** Rig movement in camera space: +x right, +y up, +z forward (metres). */
  right: number;
  up: number;
  forward: number;
}

/**
 * Rig movement for two fingers going from (a0, b0) to (a1, b1). The midpoint delta pans:
 * fingers moving right move the camera left, so the scene follows the fingers. The ratio
 * of finger distances pinches: spreading moves forward. Fingers closer than 1 px give no
 * pinch (the ratio is meaningless).
 */
export function twoFinger(
  a0: Point,
  b0: Point,
  a1: Point,
  b1: Point,
  t: TouchTuning = DEFAULT_TOUCH,
): TwoFingerStep {
  const mx = (a1.x + b1.x - a0.x - b0.x) / 2;
  const my = (a1.y + b1.y - a0.y - b0.y) / 2;
  const d0 = Math.hypot(b0.x - a0.x, b0.y - a0.y);
  const d1 = Math.hypot(b1.x - a1.x, b1.y - a1.y);
  const forward = d0 >= 1 && d1 >= 1 ? Math.log(d1 / d0) * t.pinchM : 0;
  // Screen y grows downwards: fingers moving down move the camera up.
  return { right: -mx * t.panMPerPx, up: my * t.panMPerPx, forward };
}

/** Pointer travel under this many CSS pixels is a tap, not a drag (fingers wobble). */
export const TAP_SLOP_PX = 10;

export function isTap(down: Point, up: Point, slop = TAP_SLOP_PX): boolean {
  return Math.hypot(up.x - down.x, up.y - down.y) < slop;
}
