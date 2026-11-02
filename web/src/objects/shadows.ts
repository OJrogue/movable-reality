// Contact shadows for objects standing on the floor, drawn when the room shell replaces the
// scan's surfaces (room/shell.json replaces_splats). The shell floor is built without the
// furniture's shadows (scripts/room_shell.py regenerates the floor under and around each
// footprint), so they would stay behind when an object moves; here each object carries its
// own soft shadow instead. The decal hangs under the object's offset group (scene frame of
// the unedited object), so a move, turn or scale takes it along.
import * as THREE from "three";
import type { SceneObject } from "../scene";

export interface ShadowRect {
  cx: number;
  cz: number;
  sx: number; // footprint size, metres
  sz: number;
}

/** Footprint of an object that stands on the floor (bbox bottom within `floorTol` of
 * y = 0), else null: a shelf item casts no contact shadow on the floor. */
export function shadowRect(o: Pick<SceneObject, "bbox_min" | "bbox_max">, floorTol = 0.08): ShadowRect | null {
  if (o.bbox_min[1] > floorTol) return null;
  const sx = o.bbox_max[0] - o.bbox_min[0];
  const sz = o.bbox_max[2] - o.bbox_min[2];
  if (!(sx > 0 && sz > 0)) return null;
  return { cx: (o.bbox_min[0] + o.bbox_max[0]) / 2, cz: (o.bbox_min[2] + o.bbox_max[2]) / 2, sx, sz };
}

/** Alpha at (x, z) metres from the footprint centre: `strength` inside the footprint shrunk
 * by `inset`, falling smoothly to 0 at `feather` metres past the footprint edge. */
export function shadowAlpha(x: number, z: number, r: ShadowRect, strength = 0.16, feather = 0.06, inset = 0.12): number {
  const dx = Math.max(Math.abs(x) - (r.sx / 2 - inset), 0);
  const dz = Math.max(Math.abs(z) - (r.sz / 2 - inset), 0);
  const t = Math.min(Math.hypot(dx, dz) / (feather + inset), 1);
  return strength * (1 - t * t * (3 - 2 * t)); // smoothstep fall-off
}

const PX_PER_M = 64;
const FEATHER = 0.06;

/** A horizontal quad 3 mm above the floor under the footprint, with its alpha texture. */
export function makeShadow(r: ShadowRect): THREE.Mesh {
  const w = r.sx + 2 * FEATHER;
  const d = r.sz + 2 * FEATHER;
  const pw = Math.max(8, Math.round(w * PX_PER_M));
  const pd = Math.max(8, Math.round(d * PX_PER_M));
  const data = new Uint8Array(pw * pd * 4);
  for (let j = 0; j < pd; j++) {
    for (let i = 0; i < pw; i++) {
      const a = shadowAlpha(((i + 0.5) / pw - 0.5) * w, ((j + 0.5) / pd - 0.5) * d, r, undefined, FEATHER);
      data[(j * pw + i) * 4 + 3] = Math.round(a * 255);
    }
  }
  const tex = new THREE.DataTexture(data, pw, pd, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  const mat = new THREE.MeshBasicMaterial({ color: 0x000000, map: tex, transparent: true, depthWrite: false });
  // The texture's colour is black; MeshBasicMaterial multiplies, so only alpha matters.
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(r.cx, 0.003, r.cz);
  mesh.renderOrder = 1; // after the opaque shell floor
  return mesh;
}
