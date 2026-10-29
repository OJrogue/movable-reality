// Mirror of src/movable_reality/scene.py. Change both together.
import * as THREE from "three";
import { SplatMesh, type SplatMeshOptions } from "@sparkjsdev/spark";

export type Vec3 = [number, number, number];

export interface Transform {
  translation: Vec3;
  yaw_deg: number;
  scale: number;
}

export interface SceneObject {
  id: string;
  label: string;
  ply: string;
  completed_ply: string | null;
  lod?: string | null; // lane lod: Spark .RAD tree for this object
  bbox_min: Vec3;
  bbox_max: Vec3;
  transform: Transform;
  visible: boolean;
  source: "scan" | "generated";
  prompt: string | null;
}

// ---- lane: bake (owned by the bake lane, see AGENTS.md)
export interface BakedPlane {
  id: string;
  label: string;
  mesh: string; // .glb relative to the scene folder, recon frame like the PLYs
  removed_splats: number;
  // Geometry in the world frame, written by scripts/bake_planes.py.
  normal?: Vec3 | null; // unit, points into the room
  offset?: number | null; // plane is {p : normal . p = offset}, metres
  corners?: Vec3[]; // quad corners, world frame, in order
  texel_m?: number | null;
  texture_px?: [number, number] | null; // width, height
  fit_rms_m?: number | null;
  filled_fraction?: number; // share of texels no Gaussian covered, filled by hole filling
}

/** How the planes were baked; background_ply in the manifest already points at the reduced PLY. */
export interface PlaneBake {
  source_background_ply: string; // original background, unchanged
  background_ply: string; // background without the baked Gaussians
  ids: string;
  ids_ply: string;
  removed_splats: number;
}

// ---- lane: coverage (owned by the coverage lane, see AGENTS.md)
export type GapStatus = "open" | "recapture_requested" | "recaptured" | "generated" | "accepted";

/** Where to stand and point the phone for a top-up video of one gap (world frame). */
export interface NextBestView {
  eye: Vec3; // phone position, metres; the user stands at (eye[0], 0, eye[2])
  target: Vec3; // point to aim the phone at
  visible: number; // low-coverage gap points this view would see
  gain: number; // share of the gap's low-coverage points it would see, 0..1
}

export interface Gap {
  id: string;
  label: string | null;
  near_object: string | null;
  center: Vec3; // world frame, metres
  radius: number;
  score: number; // 0 = well covered, 1 = never observed
  status: GapStatus;
  next_best_views?: NextBestView[]; // best first; missing in older manifests
}

export interface Coverage {
  gap_map: string | null;
  gaps: Gap[];
  generated_splats: number;
  gap_share?: number | null; // share of VR-visible non-empty voxels (surfaces, unobserved space) in gaps
}

// Optional sections may be missing in older manifests; fall back to the plain PLY path.
export interface SceneManifest {
  version: number;
  scene_id: string;
  units: "m";
  world_from_recon: number[][]; // 4x4, row-major
  background_ply: string;
  background_lod?: string | null; // lane lod
  objects: SceneObject[];
  planes?: BakedPlane[]; // lane bake
  plane_bake?: PlaneBake | null; // lane bake
  coverage?: Coverage | null; // lane coverage
  history: unknown[];
}

/** Rotate 180 degrees about X: OpenCV (+Y down, +Z forward) to +Y up. */
export const OPENCV_TO_Y_UP: number[][] = [
  [1, 0, 0, 0],
  [0, -1, 0, 0],
  [0, 0, -1, 0],
  [0, 0, 0, 1],
];

export function matrixFromRows(rows: number[][]): THREE.Matrix4 {
  // Matrix4.fromArray is column-major, so load the rows and transpose.
  return new THREE.Matrix4().fromArray(rows.flat()).transpose();
}

function center(o: SceneObject): Vec3 {
  return [0, 1, 2].map((i) => (o.bbox_min[i] + o.bbox_max[i]) / 2) as Vec3;
}

export interface ObjectNode {
  pivot: THREE.Group; // world position of the bbox centre, carries the edit
  mesh: SplatMesh;
}

/**
 * Scene graph for one manifest:
 * root ─ recon(background) ─ SplatMesh
 *      └ pivot(obj) ─ offset(-centre) ─ recon ─ SplatMesh
 * `recon` groups hold world_from_recon; `pivot` holds the object's edit about its centre.
 */
/** Splat meshes whose load failed (404, corrupt file); they never initialise. */
export const failedMeshes = new WeakSet<SplatMesh>();

export class SceneView {
  readonly root = new THREE.Group();
  private readonly nodes = new Map<string, ObjectNode>();
  private background?: SplatMesh;
  private backgroundPly?: string;

  /** Called when a splat file fails to load. */
  onLoadError?: (ply: string, err: unknown) => void;

  /**
   * Hook for lanes that need to change how splat meshes are created (lod: LOD trees,
   * lodScale). Called once per mesh; objectId is null for the background. Return extra
   * SplatMesh options; the url is filled in unless the hook sets it.
   */
  meshOptions?: (ply: string, objectId: string | null, m: SceneManifest) => Partial<SplatMeshOptions>;

  constructor(private readonly baseUrl: string) {}

  /** The background splat mesh, once loaded. */
  backgroundMesh(): SplatMesh | undefined {
    return this.background;
  }

  /** The splat mesh and pivot of one object, once created. */
  objectNode(id: string): Readonly<ObjectNode> | undefined {
    return this.nodes.get(id);
  }

  apply(m: SceneManifest): void {
    const worldFromRecon = matrixFromRows(m.world_from_recon);
    // A bake swaps background_ply for the reduced PLY: replace the mesh, or the baked floor
    // and ceiling splats stay drawn over the new quads.
    if (this.backgroundPly !== m.background_ply) {
      const old = this.background;
      if (old) {
        old.parent?.removeFromParent();
        old.dispose();
      }
      this.background = this.addRecon(this.root, m.background_ply, worldFromRecon, null, m);
      this.backgroundPly = m.background_ply;
    }
    for (const obj of m.objects) {
      const node = this.nodes.get(obj.id) ?? this.create(obj, worldFromRecon, m);
      const c = center(obj);
      const t = obj.transform;
      node.pivot.position.set(c[0] + t.translation[0], c[1] + t.translation[1], c[2] + t.translation[2]);
      node.pivot.rotation.set(0, THREE.MathUtils.degToRad(t.yaw_deg), 0);
      node.pivot.scale.setScalar(t.scale);
      node.pivot.visible = obj.visible;
      // TODO(M4): swap to obj.completed_ply once the object has been moved or rotated.
    }
  }

  private create(obj: SceneObject, worldFromRecon: THREE.Matrix4, m: SceneManifest): ObjectNode {
    const pivot = new THREE.Group();
    pivot.name = obj.id;
    const offset = new THREE.Group();
    const c = center(obj);
    offset.position.set(-c[0], -c[1], -c[2]);
    pivot.add(offset);
    const mesh = this.addRecon(offset, obj.ply, worldFromRecon, obj.id, m);
    this.root.add(pivot);
    const node = { pivot, mesh };
    this.nodes.set(obj.id, node);
    return node;
  }

  private addRecon(
    parent: THREE.Object3D,
    ply: string,
    worldFromRecon: THREE.Matrix4,
    objectId: string | null,
    m: SceneManifest,
  ): SplatMesh {
    const recon = new THREE.Group();
    recon.matrixAutoUpdate = false;
    recon.matrix.copy(worldFromRecon);
    recon.matrixWorldNeedsUpdate = true;
    const extra = this.meshOptions?.(ply, objectId, m) ?? {};
    const mesh = new SplatMesh({ url: `${this.baseUrl}/${ply}`, ...extra });
    mesh.initialized.catch((err: unknown) => {
      failedMeshes.add(mesh);
      this.onLoadError?.(ply, err);
    });
    recon.add(mesh);
    parent.add(recon);
    return mesh;
  }
}
