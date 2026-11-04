// Lane ui: Blender-style view moves for the mouse. The rig carries the camera, so turning the rig
// about a point in the world orbits the camera about that point. Pure and testable.
import * as THREE from "three";

export const ORBIT_RAD_PER_PX = 0.005;
/** Orbit stops short of straight up and down, so the view never flips over. */
export const MAX_PITCH_RAD = THREE.MathUtils.degToRad(85);

const UP = new THREE.Vector3(0, 1, 0);
const _yaw = new THREE.Quaternion();
const _pitch = new THREE.Quaternion();
const _right = new THREE.Vector3();
const _offset = new THREE.Vector3();
const _euler = new THREE.Euler(0, 0, 0, "YXZ");

/**
 * Orbit by a mouse movement in pixels. The room follows the mouse: drag right and the point in
 * front of you moves right, drag down and you look from higher up.
 */
export function orbitRig(rig: THREE.Object3D, pivot: THREE.Vector3, dx: number, dy: number, k = ORBIT_RAD_PER_PX): void {
  _yaw.setFromAxisAngle(UP, -dx * k);
  _euler.setFromQuaternion(rig.quaternion);
  const pitch = THREE.MathUtils.clamp(_euler.x - dy * k, -MAX_PITCH_RAD, MAX_PITCH_RAD) - _euler.x;
  _right.set(1, 0, 0).applyQuaternion(rig.quaternion);
  _pitch.setFromAxisAngle(_right, pitch);
  _yaw.multiply(_pitch);
  _offset.copy(rig.position).sub(pivot).applyQuaternion(_yaw);
  rig.position.copy(pivot).add(_offset);
  rig.quaternion.premultiply(_yaw).normalize();
}

/** Slide the view so the point at `distance` metres follows the mouse. */
export function panRig(rig: THREE.Object3D, dx: number, dy: number, distance: number, fovDeg: number, heightPx: number): void {
  const metresPerPx = (2 * distance * Math.tan(THREE.MathUtils.degToRad(fovDeg) / 2)) / Math.max(heightPx, 1);
  rig.position.add(new THREE.Vector3(-dx * metresPerPx, dy * metresPerPx, 0).applyQuaternion(rig.quaternion));
}

export type Outline = readonly (readonly [number, number])[];

/** Room outline corners are in order; only a convex room can be clamped edge by edge. */
export function isConvex(outline: Outline): boolean {
  let sign = 0;
  for (let i = 0; i < outline.length; i++) {
    const [ax, az] = outline[i];
    const [bx, bz] = outline[(i + 1) % outline.length];
    const [cx, cz] = outline[(i + 2) % outline.length];
    const cross = (bx - ax) * (cz - bz) - (bz - az) * (cx - bx);
    if (Math.abs(cross) < 1e-9) continue;
    if (sign !== 0 && Math.sign(cross) !== sign) return false;
    sign = Math.sign(cross);
  }
  return outline.length >= 3 && sign !== 0;
}

/** The nearest point at least `inset` metres inside a convex outline. */
export function clampToOutline(x: number, z: number, outline: Outline, inset: number): [number, number] {
  const n = outline.length;
  const cx = outline.reduce((a, p) => a + p[0], 0) / n;
  const cz = outline.reduce((a, p) => a + p[1], 0) / n;
  // Twice, so a push off one wall cannot leave the point outside the next one in a corner.
  for (let pass = 0; pass < 2; pass++) {
    for (let i = 0; i < n; i++) {
      const [ax, az] = outline[i];
      const [bx, bz] = outline[(i + 1) % n];
      const len = Math.hypot(bx - ax, bz - az);
      if (len < 1e-9) continue;
      let nx = -(bz - az) / len;
      let nz = (bx - ax) / len;
      if (nx * (cx - ax) + nz * (cz - az) < 0) {
        nx = -nx; // the normal points inwards
        nz = -nz;
      }
      const d = nx * (x - ax) + nz * (z - az);
      if (d < inset) {
        x += nx * (inset - d);
        z += nz * (inset - d);
      }
    }
  }
  return [x, z];
}

