// Lane bake: baked floor and ceiling quads (scripts/bake_planes.py) in the viewer.
//
// Each manifest plane is a .glb quad in the recon frame, like the PLYs. It goes under a
// group holding world_from_recon, as SceneView does for splats, and is drawn opaque with
// depth writes on. three.js draws opaque meshes before transparent ones and Spark's
// SparkRenderer is one transparent mesh (depthTest on, depthWrite off), so the quads are in
// the depth buffer before any splat is blended and splats behind them are rejected.
//
// The bake points background_ply in scene.json at the background without the replaced
// splats, so SceneView (and the lod lane's meshOptions) already load the reduced PLY;
// nothing here replaces splat meshes.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ViewerContext } from "../context";
import { matrixFromRows, type BakedPlane } from "../scene";
import { diffPlanes, makeOpaque, planeKey, planeUrl, reconGroup, staleBackgroundLod } from "./planes";

export interface PlanesHandle {
  readonly group: THREE.Group;
  /** Plane ids currently shown. */
  loaded(): string[];
  /** Resolves when every plane requested so far has loaded or failed. */
  idle(): Promise<void>;
}

export function installPlanes(ctx: ViewerContext): PlanesHandle {
  const group = new THREE.Group();
  group.name = "baked-planes";
  const loader = new GLTFLoader();
  const nodes = new Map<string, { id: string; node: THREE.Object3D }>();
  const pending = new Set<Promise<void>>();
  const maxAniso = ctx.renderer.capabilities.getMaxAnisotropy();
  let attached = false;
  let warnedLod = false;

  const load = (plane: BakedPlane, key: string, worldFromRecon: THREE.Matrix4) => {
    const url = planeUrl(ctx.sceneBase ?? "", plane);
    const p = loader
      .loadAsync(url)
      .then((gltf) => {
        const recon = reconGroup(worldFromRecon);
        recon.name = `plane:${plane.id}`;
        const meshes = makeOpaque(gltf.scene, maxAniso);
        recon.add(gltf.scene);
        group.add(recon);
        nodes.set(key, { id: plane.id, node: recon });
        console.info(`[planes] ${plane.id}: ${plane.mesh} (${meshes} mesh, ${plane.removed_splats} splats replaced)`);
      })
      .catch((err) => console.warn(`[planes] ${plane.id}: failed to load ${url}: ${String(err)}`))
      .finally(() => pending.delete(p));
    pending.add(p);
  };

  ctx.onManifest((m, view) => {
    if (!attached) {
      view.root.add(group);
      attached = true;
    }
    const worldFromRecon = matrixFromRows(m.world_from_recon);
    const { add, remove } = diffPlanes(nodes.keys(), m);
    for (const key of remove) {
      const entry = nodes.get(key);
      if (!entry) continue;
      group.remove(entry.node);
      dispose(entry.node);
      nodes.delete(key);
    }
    for (const plane of add) load(plane, planeKey(plane), worldFromRecon);
    if (!warnedLod && staleBackgroundLod(m)) {
      warnedLod = true;
      console.warn(
        `[planes] background_lod ${m.background_lod} predates the bake; it still holds the` +
          ` baked splats. Run scripts/build_lod.py ${m.scene_id} to rebuild it from ${m.plane_bake?.background_ply}.`,
      );
    }
    // Planes replace splats that a re-bake or revert brings back; SceneView builds the
    // background mesh once, so a changed background_ply shows after a reload.
  });

  return {
    group,
    loaded: () => [...nodes.values()].map((n) => n.id),
    idle: async () => {
      while (pending.size) await Promise.allSettled([...pending]);
    },
  };
}

function dispose(root: THREE.Object3D): void {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry.dispose();
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const mat of mats) {
      (mat as THREE.MeshBasicMaterial).map?.dispose();
      mat.dispose();
    }
  });
}
