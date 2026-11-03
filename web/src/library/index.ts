// Lane library: draws the premade meshes (src/movable_reality/library.py) that a scene holds as
// objects with source "library". Each hangs under its object's pivot, so move, rotate, remove and
// undo work as for every other object. The files stand on y = 0, centred on x = z = 0, in the
// world frame, so there is no world_from_recon here. They are lit PBR models (the scan's splats
// carry their light, these do not): the first one sets a neutral image-based light on the scene,
// which unlit planes and splats ignore.
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ViewerContext } from "../context";
import type { SceneObject, Vec3 } from "../scene";

export const LIBRARY_BASE = "/library";
const ENV_INTENSITY = 0.75;

export interface LibraryItem {
  id: string;
  label: string;
  category: string;
  on_surface: boolean;
  size: Vec3; // width x, height y, depth z, metres
  triangles: number;
  glb: string; // relative to LIBRARY_BASE
  thumb: string | null;
}

/** The catalogue, or [] when this server has no library (scripts/build_library.py). */
export async function loadCatalog(): Promise<LibraryItem[]> {
  try {
    const r = await fetch(`${LIBRARY_BASE}/catalog.json`);
    if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) return [];
    return (await r.json()) as LibraryItem[];
  } catch {
    return [];
  }
}

/** Catalogue items grouped by category, in catalogue order. */
export function byCategory(items: LibraryItem[]): [string, LibraryItem[]][] {
  const groups = new Map<string, LibraryItem[]>();
  for (const item of items) groups.set(item.category, [...(groups.get(item.category) ?? []), item]);
  return [...groups];
}

const MAX_DROP_M = 3.5;
const FALLBACK_DROP_M = 2;

/** Where a new item lands: where the view's centre hits the floor when that is near the
 * user, else a short way ahead on the floor (the user looks at the horizon or up). */
export function dropPoint(eye: Vec3, forward: Vec3): { x: number; z: number } {
  if (forward[1] < -1e-3) {
    const t = -eye[1] / forward[1];
    const x = eye[0] + forward[0] * t;
    const z = eye[2] + forward[2] * t;
    if (Math.hypot(x - eye[0], z - eye[2]) <= MAX_DROP_M) return { x, z };
  }
  const n = Math.hypot(forward[0], forward[2]);
  const [fx, fz] = n > 1e-6 ? [forward[0] / n, forward[2] / n] : [0, -1];
  return { x: eye[0] + fx * FALLBACK_DROP_M, z: eye[2] + fz * FALLBACK_DROP_M };
}

/** Yaw that turns an item's front (+Z in the file) towards the user, in 90 degree steps:
 * furniture stands square to the walls, and a diagonal sofa finds no free spot in a bedroom. */
export function yawFacing(x: number, z: number, eye: Vec3): number {
  const deg = THREE.MathUtils.radToDeg(Math.atan2(eye[0] - x, eye[2] - z));
  return ((Math.round(deg / 90) * 90) % 360 + 360) % 360;
}

export function installLibrary(ctx: ViewerContext): void {
  const loader = new GLTFLoader();
  const files = new Map<string, Promise<THREE.Group>>();
  const placed = new Set<string>();
  let lit = false;

  const light = () => {
    if (lit) return;
    lit = true;
    const pmrem = new THREE.PMREMGenerator(ctx.renderer);
    ctx.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    ctx.scene.environmentIntensity = ENV_INTENSITY;
    pmrem.dispose();
  };

  const file = (id: string): Promise<THREE.Group> => {
    let f = files.get(id);
    if (!f) {
      f = loader.loadAsync(`${LIBRARY_BASE}/${id}.glb`).then((g) => g.scene);
      files.set(id, f);
    }
    return f;
  };

  const place = (o: SceneObject, offset: THREE.Object3D, libraryId: string) => {
    placed.add(o.id);
    light();
    const group = new THREE.Group();
    group.name = `library:${o.id}`;
    // Under the offset group (minus the bbox centre), so this is the unedited world position.
    group.position.set((o.bbox_min[0] + o.bbox_max[0]) / 2, o.bbox_min[1], (o.bbox_min[2] + o.bbox_max[2]) / 2);
    offset.add(group);
    file(libraryId)
      .then((scene) => group.add(scene.clone()))
      .catch((err) => console.warn(`[library] ${o.id}: failed to load ${libraryId}: ${String(err)}`));
  };

  ctx.onManifest((m, view) => {
    for (const o of m.objects) {
      if (o.source !== "library" || !o.library_id || !o.visible || placed.has(o.id)) continue;
      const offset = view.objectNode(o.id)?.pivot.children[0];
      if (offset) place(o, offset, o.library_id);
    }
  });
}
