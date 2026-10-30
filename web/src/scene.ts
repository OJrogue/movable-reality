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
  completed_lod?: string | null; // lane lod: Spark .RAD tree for completed_ply
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
  transparent_fraction?: number; // share of texels outside the outline (alpha mask)
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
  generated_ply?: string | null; // C3 repairs, every splat generated = 1, recon frame
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

/** True when the edit moves, rotates or scales the object (tolerance 1e-6). */
export function isEdited(t: Transform): boolean {
  const eps = 1e-6;
  return (
    t.translation.some((v) => Math.abs(v) > eps) ||
    Math.min(Math.abs(t.yaw_deg % 360), 360 - Math.abs(t.yaw_deg % 360)) > eps ||
    Math.abs(t.scale - 1) > eps
  );
}

/**
 * The PLY an object shows. That is its completed version (SAM 3D Objects fills the unseen
 * back) once it has been edited, since only then can the viewer see the missing side, and
 * the scan otherwise. An undo back to the identity transform returns to the scan.
 */
export function shownPly(o: SceneObject): string {
  return o.completed_ply && isEdited(o.transform) ? o.completed_ply : o.ply;
}

export interface ObjectNode {
  pivot: THREE.Group; // world position of the bbox centre, carries the edit
  mesh: SplatMesh;
  ply: string; // the PLY `mesh` shows: obj.ply, or obj.completed_ply after an edit
}

/** Show or hide a splat mesh through its recon group (Spark skips hidden subtrees). */
function setShown(mesh: SplatMesh, shown: boolean): void {
  if (mesh.parent) mesh.parent.visible = shown;
}

/** Splat meshes whose load failed (404, corrupt file); they never initialise. */
export const failedMeshes = new WeakSet<SplatMesh>();

/**
 * Scene graph for one manifest:
 * root ─ recon(background) ─ SplatMesh
 *      └ pivot(obj) ─ offset(-centre) ─ recon ─ SplatMesh (one recon per variant: scan, completed)
 * `recon` groups hold world_from_recon; `pivot` holds the object's edit about its centre.
 */

export class SceneView {
  readonly root = new THREE.Group();
  private readonly nodes = new Map<string, ObjectNode>();
  /** Every mesh created per object, keyed by PLY (scan and completed), or "failed". */
  private readonly variants = new Map<string, Map<string, SplatMesh | "failed">>();
  /** Variants that have finished loading. */
  private readonly loaded = new WeakSet<SplatMesh>();
  /** Latest manifest entry per object, for swaps that finish after a later edit. */
  private readonly objects = new Map<string, SceneObject>();
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
      this.objects.set(obj.id, obj);
      const node = this.nodes.get(obj.id) ?? this.create(obj, worldFromRecon, m);
      const c = center(obj);
      const t = obj.transform;
      node.pivot.position.set(c[0] + t.translation[0], c[1] + t.translation[1], c[2] + t.translation[2]);
      node.pivot.rotation.set(0, THREE.MathUtils.degToRad(t.yaw_deg), 0);
      node.pivot.scale.setScalar(t.scale);
      node.pivot.visible = obj.visible;
      if (shownPly(obj) !== node.ply) this.swap(node, obj, worldFromRecon, m);
    }
  }

  /**
   * Show the variant `shownPly(obj)` asks for (scan to completed after an edit, back on
   * undo). Each variant is created once and then kept hidden when unused, so swapping back
   * is instant and Spark keeps at most two LOD trees per object (D7 keeps hidden trees in
   * memory, so a disposed mesh would leave its tree behind in the worker). The previous
   * variant stays drawn until the new one has loaded, so the object never blinks out. A
   * variant that fails to load is remembered and the scan is shown in its place.
   */
  private swap(node: ObjectNode, obj: SceneObject, worldFromRecon: THREE.Matrix4, m: SceneManifest): void {
    const variants = this.variants.get(obj.id)!;
    node.ply = shownPly(obj);
    // A completed PLY that failed to load falls back to the scan.
    const ply = variants.get(node.ply) === "failed" ? obj.ply : node.ply;
    let mesh = variants.get(ply);
    if (mesh === "failed") {
      for (const v of variants.values()) if (v !== "failed") setShown(v, false);
      return; // the scan itself is broken; onLoadError has reported it
    }
    if (!mesh) {
      const created = this.addRecon(node.pivot.children[0], ply, worldFromRecon, obj.id, m);
      variants.set(ply, created);
      // Re-run the choice with the latest manifest once the load settles: the edit may
      // have been undone meanwhile, or the load may have failed.
      const settle = (ok: boolean) => {
        if (ok) this.loaded.add(created);
        else {
          variants.set(ply, "failed");
          created.parent?.removeFromParent();
        }
        const latest = this.objects.get(obj.id);
        if (latest) this.swap(node, latest, worldFromRecon, m);
      };
      created.initialized.then(
        () => settle(true),
        () => settle(false),
      );
      mesh = created;
    }
    node.mesh = mesh;
    setShown(mesh, true);
    // Hide the others only once this one has loaded, so the object never blinks out.
    if (!this.loaded.has(mesh)) return;
    for (const v of variants.values()) if (v !== mesh && v !== "failed") setShown(v, false);
  }

  private create(obj: SceneObject, worldFromRecon: THREE.Matrix4, m: SceneManifest): ObjectNode {
    const pivot = new THREE.Group();
    pivot.name = obj.id;
    const offset = new THREE.Group();
    const c = center(obj);
    offset.position.set(-c[0], -c[1], -c[2]);
    pivot.add(offset);
    this.root.add(pivot);
    this.variants.set(obj.id, new Map());
    this.objects.set(obj.id, obj);
    // swap() fills in mesh and ply straight away (it creates the first variant).
    const node = { pivot, ply: "" } as ObjectNode;
    this.swap(node, obj, worldFromRecon, m);
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
