// Lane bake: pure logic for baked planes (no WebGL), so vitest can cover it.
import * as THREE from "three";
import type { BakedPlane, SceneManifest } from "../scene";

/** Draw before every other opaque mesh; Spark's splats are in the transparent pass anyway. */
export const PLANE_RENDER_ORDER = -100;

/** Identity of a baked plane: a re-bake changes geometry or texture, so it changes the key. */
export function planeKey(p: BakedPlane): string {
  return JSON.stringify([p.mesh, p.corners ?? null, p.texel_m ?? null, p.texture_px ?? null, p.fit_rms_m ?? null]);
}

export function planeUrl(sceneBase: string, p: BakedPlane): string {
  // The key in the query string keeps a re-baked file from coming out of the HTTP cache.
  return `${sceneBase}/${p.mesh}?v=${hash(planeKey(p))}`;
}

export interface PlaneDiff {
  add: BakedPlane[];
  remove: string[]; // keys
}

/** Which planes to load and which loaded ones to drop for a new manifest. */
export function diffPlanes(loaded: Iterable<string>, m: SceneManifest): PlaneDiff {
  const want = new Map((m.planes ?? []).map((p) => [planeKey(p), p]));
  const have = new Set(loaded);
  return {
    add: [...want].filter(([k]) => !have.has(k)).map(([, p]) => p),
    remove: [...have].filter((k) => !want.has(k)),
  };
}

/**
 * Make every mesh under `root` an opaque occluder drawn first: depth test and depth write
 * on, no blending, unlit colours as baked. Returns the number of meshes changed.
 */
export function makeOpaque(root: THREE.Object3D, maxAnisotropy = 1): number {
  let n = 0;
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const mat of mats) {
      mat.transparent = false;
      mat.opacity = 1;
      mat.alphaTest = 0;
      mat.depthTest = true;
      mat.depthWrite = true;
      mat.blending = THREE.NoBlending;
      mat.side = THREE.DoubleSide;
      const map = (mat as THREE.MeshBasicMaterial).map;
      if (map) {
        map.anisotropy = Math.max(1, Math.min(maxAnisotropy, 8)); // grazing views of the floor
        map.needsUpdate = true;
      }
      mat.needsUpdate = true;
    }
    mesh.renderOrder = PLANE_RENDER_ORDER;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    n += 1;
  });
  return n;
}

/** Group that maps the recon frame (OpenCV, like the PLYs) into the world frame. */
export function reconGroup(worldFromRecon: THREE.Matrix4): THREE.Group {
  const g = new THREE.Group();
  g.matrixAutoUpdate = false;
  g.matrix.copy(worldFromRecon);
  g.matrixWorldNeedsUpdate = true;
  return g;
}

function hash(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(36);
}
