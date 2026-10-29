// Read-only views of Spark 2.4.0 state for the HUD, the frame log and bench readiness.
// Every access to a Spark internal lives here, so a Spark upgrade touches one file.
// Checked against node_modules/@sparkjsdev/spark/dist/spark.module.js (2.4.0):
//   - SparkRenderer.activeSplats: splats drawn (the instance count), i.e. the last sort's
//     result after culling. Sorting runs in a worker, so after a change (LOD cut, toggle)
//     it lags by one sort, which on a headset is a few frames and in SwiftShader seconds.
//   - SparkRenderer.display.mapping[].count: splats each visible mesh put into the
//     displayed frame: the LOD cut for an LOD mesh, all of its splats otherwise. Same lag.
//   - SparkRenderer.lodInstances: Map<SplatMesh, {numSplats}> once a mesh has a cut.
//   - lodSplatCount, lodSplatScale, lodRenderScale, lodTraverseMode, enableLod and the cone
//     foveation fields are read on every LOD update, so they can change at runtime.
import * as THREE from "three";
import { SplatMesh, type SparkRenderer } from "@sparkjsdev/spark";
import type { LodSettings } from "./frameLog";

export interface SparkCounters {
  drawn: number;
  cut: number;
}

export function sparkCounters(spark: SparkRenderer): SparkCounters {
  let cut = 0;
  for (const m of spark.display.mapping) cut += m.count;
  return { drawn: spark.activeSplats, cut };
}

/** The splat budget Spark uses for LOD: explicit count or platform default, times scale. */
export function lodSplatCount(spark: SparkRenderer): { count: number; explicit: boolean } {
  const explicit = spark.lodSplatCount !== undefined;
  return { count: spark.lodSplatCount ?? spark.defaultSplatTarget(), explicit };
}

export function lodSettings(spark: SparkRenderer, budgetOn: boolean): LodSettings {
  const { count, explicit } = lodSplatCount(spark);
  return {
    enable_lod: spark.enableLod,
    lod_splat_count: count,
    lod_splat_count_explicit: explicit,
    lod_splat_scale: spark.lodSplatScale,
    lod_render_scale: spark.lodRenderScale,
    lod_traverse_mode: spark.lodTraverseMode,
    budget_controller: budgetOn,
    cone_fov0: spark.coneFov0,
    cone_fov: spark.coneFov,
    cone_foveate: spark.coneFoveate,
    behind_foveate: spark.behindFoveate,
  };
}

export function splatMeshes(scene: THREE.Object3D): SplatMesh[] {
  const out: SplatMesh[] = [];
  scene.traverse((o) => {
    if (o instanceof SplatMesh) out.push(o);
  });
  return out;
}

function visibleMeshes(scene: THREE.Object3D): Set<SplatMesh> {
  const out = new Set<SplatMesh>();
  scene.traverseVisible((o) => {
    if (o instanceof SplatMesh) out.add(o);
  });
  return out;
}

export function hasLodTree(m: SplatMesh): boolean {
  return Boolean(m.packedSplats?.lodSplats || m.extSplats?.lodSplats || m.paged);
}

/** Splats in the mesh's LOD tree (leaves and inner nodes), or null without a tree. */
export function treeSplats(m: SplatMesh): number | null {
  const tree = m.packedSplats?.lodSplats ?? m.extSplats?.lodSplats;
  return tree ? tree.numSplats : null;
}

/** Whether the mesh holds its full splat set next to its tree (SplatMesh `nonLod`). */
export function hasFullSplats(m: SplatMesh): boolean {
  if (m.paged) return false;
  return (m.packedSplats?.numSplats ?? 0) > 0 || (m.extSplats?.numSplats ?? 0) > 0;
}

/**
 * True once every splat mesh has loaded and, with LOD on, every visible LOD mesh has its
 * first cut. Before that, frame times measure loading, not rendering.
 */
export function splatsReady(scene: THREE.Object3D, spark: SparkRenderer): boolean {
  const meshes = splatMeshes(scene);
  if (meshes.length === 0) return false;
  if (meshes.some((m) => !m.isInitialized)) return false;
  if (!spark.enableLod) return true;
  for (const m of visibleMeshes(scene)) {
    if (!hasLodTree(m)) continue;
    const cut = m.paged ? m.paged.numSplats : spark.lodInstances.get(m)?.numSplats;
    if (!cut) return false;
  }
  return true;
}

/** Sum of the trees' sizes over all splat meshes (null if no mesh has a tree). */
export function totalTreeSplats(scene: THREE.Object3D): number | null {
  let sum = 0;
  let any = false;
  for (const m of splatMeshes(scene)) {
    const n = treeSplats(m);
    if (n !== null) {
      sum += n;
      any = true;
    }
  }
  return any ? sum : null;
}

/** Vertex count from a binary or ASCII PLY header; reads only the header bytes. */
export async function plyVertexCount(url: string): Promise<number | null> {
  const r = await fetch(url);
  if (!r.ok || !r.body) return null;
  const reader = r.body.getReader();
  const dec = new TextDecoder("latin1");
  let head = "";
  try {
    while (head.length < 65536) {
      const { done, value } = await reader.read();
      if (done) break;
      head += dec.decode(value, { stream: true });
      if (head.includes("end_header")) break;
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
  if (!head.startsWith("ply")) return null;
  const m = /element vertex (\d+)/.exec(head);
  return m ? Number(m[1]) : null;
}
