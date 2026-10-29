// Lane lod: which object a ray points at, tested against the manifest bboxes (not splats).
// The box is tested in the object's pivot frame, so yaw and scale edits are exact: the
// pivot sits at the bbox centre plus the edit (see SceneView in ../scene.ts), so in pivot
// coordinates the object occupies [-half, +half] with half = (bbox_max - bbox_min) / 2.
import * as THREE from "three";
import type { SceneObject, Vec3 } from "../scene";

export function halfSize(o: Pick<SceneObject, "bbox_min" | "bbox_max">): Vec3 {
  return [0, 1, 2].map((i) => Math.abs(o.bbox_max[i] - o.bbox_min[i]) / 2) as Vec3;
}

const _inv = new THREE.Matrix4();
const _local = new THREE.Ray();
const _box = new THREE.Box3();
const _hit = new THREE.Vector3();

/**
 * Distance in world units from the ray origin to where it enters the object's box, or
 * null if it misses. 0 when the origin is inside the box. `pivotWorld` is the pivot's
 * matrixWorld (translation, yaw, uniform scale).
 */
export function rayBoxDistance(ray: THREE.Ray, pivotWorld: THREE.Matrix4, half: Vec3): number | null {
  if (Math.abs(pivotWorld.determinant()) < 1e-12) return null; // scale 0: nothing to hit
  _inv.copy(pivotWorld).invert();
  _local.copy(ray).applyMatrix4(_inv);
  _local.direction.normalize();
  _box.min.set(-half[0], -half[1], -half[2]);
  _box.max.set(half[0], half[1], half[2]);
  // intersectBox returns the exit point from inside; the object is under the ray then.
  if (_box.containsPoint(_local.origin)) return 0;
  if (!_local.intersectBox(_box, _hit)) return null;
  return _hit.applyMatrix4(pivotWorld).distanceTo(ray.origin);
}

export interface Pickable {
  id: string;
  pivotWorld: THREE.Matrix4;
  half: Vec3;
}

/** Nearest object hit by the ray, or null. Ties keep the first object. */
export function pickNearest(ray: THREE.Ray, objects: Iterable<Pickable>): { id: string; distance: number } | null {
  let best: { id: string; distance: number } | null = null;
  for (const o of objects) {
    const d = rayBoxDistance(ray, o.pivotWorld, o.half);
    if (d !== null && (best === null || d < best.distance)) best = { id: o.id, distance: d };
  }
  return best;
}
