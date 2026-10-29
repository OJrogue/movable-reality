// Lane lod: per-object LOD trees and priority for the targeted object. See docs/LOD.md.
//
// installLod(ctx) makes every splat mesh a Spark LOD mesh, so SparkRenderer traverses all
// trees jointly under one splat budget (lodSplatCount * lodSplatScale):
//   - the manifest names a .RAD tree (scripts/build_lod.py) -> load it;
//   - otherwise load the PLY with `lod: true` and Spark builds the tree in a worker;
//   - ?mode=naive -> leave LOD off (plain PLYs) for the proof shot.
// Priority: the object under the controller ray (XR; head direction if no controller is
// tracked) or the mouse (desktop), and the object of the latest edit, ramp to a higher
// SplatMesh.lodScale; the rest return to 1. Rays are tested against manifest bboxes.
// URL options: ?lodPriority=<scale> (default 2), ?lodRamp=<ms> (default 300),
// ?lodBuild=quality (bhatt-lod instead of tiny-lod for load-time trees),
// ?lodSplatCount=<n> (test knob: shared splat budget, see below),
// ?lodPaged=1 (stream .RAD trees into Spark's shared page pool; see docs/LOD.md, Paging).
import * as THREE from "three";
import type { SplatMesh } from "@sparkjsdev/spark";
import type { ViewerContext } from "../context";
import type { SceneManifest } from "../scene";
import { lodChoice, type LodSource } from "./options";
import { halfSize, pickNearest, type Pickable } from "./pick";
import {
  focusReason,
  latestEditedObject,
  parsePriorityConfig,
  rampScale,
  targetScales,
  type Focus,
  type PriorityConfig,
} from "./priority";

export interface LodEvent {
  t: number; // performance.now()
  kind: "source" | "target" | "reached" | "dirty";
  id: string | null;
  detail: string;
}

/** Debug handle, also exposed as window.mrLod for the headless check and the perf lane. */
export interface LodHandle {
  readonly config: PriorityConfig;
  readonly sources: Map<string, { source: LodSource; path: string }>;
  focus(): Focus;
  /** Current lodScale of every object mesh. */
  scales(): Record<string, number>;
  /** Spark's LOD state: trees, the shared budget, and splats drawn per mesh. */
  stats(): LodStats;
  readonly events: LodEvent[];
}

export interface LodStats {
  lodMeshes: number; // meshes Spark traversed in the last LOD pass
  lodTrees: number; // trees held by Spark's LOD worker
  budget: number; // lodSplatCount (or platform default) * lodSplatScale
  drawn: number; // splats in the current cut, all meshes
  meshes: Record<string, { treeSplats: number | null; drawn: number | null; lodScale: number }>;
}

const BACKGROUND = "background";
const MAX_EVENTS = 500;

export function installLod(ctx: ViewerContext): LodHandle | null {
  if (ctx.params.get("mode") === "naive") {
    console.info("[lod] ?mode=naive: LOD off, plain PLYs");
    return null;
  }
  const config = parsePriorityConfig(ctx.params);
  const loadTimeLod = ctx.params.get("lodBuild") === "quality" ? "quality" : true;
  const paged = ctx.params.get("lodPaged") === "1";
  // Test knob: a small shared budget makes priority visible on small scenes. At runtime the
  // renderer lane's budget controller owns lodSplatCount and may override this.
  const budget = Number(ctx.params.get("lodSplatCount"));
  if (Number.isInteger(budget) && budget > 0) ctx.spark.lodSplatCount = budget;
  const sources = new Map<string, { source: LodSource; path: string }>();
  const events: LodEvent[] = [];
  const log = (kind: LodEvent["kind"], id: string | null, detail: string) => {
    events.push({ t: performance.now(), kind, id, detail });
    if (events.length > MAX_EVENTS) events.shift();
    console.info(`[lod] ${kind} ${id ?? ""} ${detail}`.replace(/\s+/g, " ").trim());
  };

  let focus: Focus = { hovered: null, edited: null };
  let targets = new Map<string, number>();

  ctx.meshOptions = (ply, objectId, m) => {
    const id = objectId ?? BACKGROUND;
    const edited = latestEditedObject(m);
    const scale = objectId !== null && objectId === edited ? config.priorityScale : config.baseScale;
    const choice = lodChoice(ply, objectId, m, ctx.sceneBase ?? "", loadTimeLod, scale, paged);
    sources.set(id, { source: choice.source, path: choice.path });
    log("source", id, `${choice.source} ${choice.path} lodScale=${scale.toFixed(2)}`);
    return choice.options;
  };

  // ---- pointer (desktop) and controller rays (XR)
  const pointer = new THREE.Vector2();
  let pointerActive = false;
  const canvas = ctx.renderer.domElement;
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    pointerActive = true;
  });
  canvas.addEventListener("pointerleave", () => (pointerActive = false));
  const raycaster = new THREE.Raycaster();
  const controllers = [0, 1].map((i) => ctx.renderer.xr.getController(i));
  const tmpMatrix = new THREE.Matrix4();

  const rays = (): THREE.Ray[] => {
    ctx.rig.updateMatrixWorld();
    if (!ctx.renderer.xr.isPresenting) {
      if (!pointerActive) return [];
      raycaster.setFromCamera(pointer, ctx.camera);
      return [raycaster.ray.clone()];
    }
    const out: THREE.Ray[] = [];
    const toRay = (world: THREE.Matrix4) =>
      new THREE.Ray(
        new THREE.Vector3().setFromMatrixPosition(world),
        new THREE.Vector3(0, 0, -1).transformDirection(world),
      );
    for (const c of controllers) {
      if (!c.visible) continue; // three.js hides a target-ray space that has no pose
      // Poses are in the XR reference space; three.js puts the XR camera under the
      // camera's parent (the rig), so the controllers live in the rig frame too.
      out.push(toRay(tmpMatrix.multiplyMatrices(ctx.rig.matrixWorld, c.matrix)));
    }
    // No tracked controller: use the head direction (WebXR exposes no eye gaze).
    if (out.length === 0) out.push(toRay(ctx.renderer.xr.getCamera().matrixWorld));
    return out;
  };

  const pickables = (m: SceneManifest): Pickable[] => {
    const out: Pickable[] = [];
    for (const o of m.objects) {
      const node = o.visible ? ctx.view?.objectNode(o.id) : undefined;
      if (!node) continue;
      node.pivot.updateWorldMatrix(true, false);
      out.push({ id: o.id, pivotWorld: node.pivot.matrixWorld, half: halfSize(o) });
    }
    return out;
  };

  const meshOf = (id: string): SplatMesh | undefined => ctx.view?.objectNode(id)?.mesh;

  // ?lodRetraverse=0 turns the nudge below off, to observe Spark's own behaviour.
  const retraverseOnEdit = ctx.params.get("lodRetraverse") !== "0";
  ctx.onManifest((m) => {
    // A move changes the pivot in one frame. That bumps SplatMesh.version once, and Spark's
    // driveLod only re-traverses when a version rises by more than one since its last pass,
    // so ask for one traversal and the cut is chosen for the new pose. Trees are untouched.
    if (!retraverseOnEdit) return;
    ctx.spark.lodDirty = true;
    log("dirty", null, `manifest change, ${m.history.length} history entries`);
  });

  let lastTime: number | null = null;
  ctx.onFrame((time) => {
    const m = ctx.manifest;
    if (!m || !ctx.view) return;
    const dt = lastTime === null ? 0 : Math.min(100, Math.max(0, time - lastTime));
    lastTime = time;

    const objs = pickables(m);
    let hovered: string | null = null;
    let best = Infinity;
    for (const ray of rays()) {
      const hit = pickNearest(ray, objs);
      if (hit && hit.distance < best) [hovered, best] = [hit.id, hit.distance];
    }
    const next: Focus = { hovered, edited: latestEditedObject(m) };
    const nextTargets = targetScales(m, next, config);
    for (const [id, target] of nextTargets) {
      const mesh = meshOf(id);
      const known = targets.get(id) ?? mesh?.lodScale;
      if (mesh && known !== target) {
        const from = mesh.lodScale.toFixed(2);
        log("target", id, `lodScale ${from} -> ${target.toFixed(2)} [${focusReason(id, next)}]`);
      }
    }
    focus = next;
    targets = nextTargets;

    let changed = false;
    for (const [id, target] of targets) {
      const mesh = meshOf(id);
      if (!mesh || mesh.lodScale === target) continue;
      mesh.lodScale = rampScale(mesh.lodScale, target, dt, config.rampMs);
      if (mesh.lodScale === target) log("reached", id, `lodScale = ${target.toFixed(2)}`);
      changed = true;
    }
    // Spark re-traverses on camera motion, budget or mesh-set changes, not on lodScale.
    if (changed) ctx.spark.lodDirty = true;
  });

  const stats = (): LodStats => {
    const spark = ctx.spark;
    const meshes: LodStats["meshes"] = {};
    let drawn = 0;
    const named: [string, SplatMesh | undefined][] = [[BACKGROUND, ctx.view?.backgroundMesh()]];
    for (const o of ctx.manifest?.objects ?? []) named.push([o.id, meshOf(o.id)]);
    for (const [id, mesh] of named) {
      if (!mesh) continue;
      const tree = mesh.packedSplats?.lodSplats ?? mesh.extSplats?.lodSplats;
      // Paged meshes keep their cut on mesh.paged; the others in spark.lodInstances.
      const cut = mesh.paged ? mesh.paged.numSplats : spark.lodInstances.get(mesh)?.numSplats;
      drawn += cut ?? 0;
      meshes[id] = { treeSplats: tree?.numSplats ?? null, drawn: cut ?? null, lodScale: mesh.lodScale };
    }
    return {
      lodMeshes: spark.lodMeshes.length,
      lodTrees: spark.lodIds.size,
      budget: (spark.lodSplatCount ?? spark.defaultSplatTarget()) * spark.lodSplatScale,
      drawn,
      meshes,
    };
  };

  const handle: LodHandle = {
    config,
    sources,
    events,
    stats,
    focus: () => focus,
    scales: () => {
      const out: Record<string, number> = {};
      for (const o of ctx.manifest?.objects ?? []) {
        const mesh = meshOf(o.id);
        if (mesh) out[o.id] = mesh.lodScale;
      }
      return out;
    },
  };
  (window as unknown as { mrLod: LodHandle }).mrLod = handle;
  console.info(
    `[lod] installed: priority lodScale ${config.priorityScale}, ramp ${config.rampMs} ms, ` +
      `load-time tree ${loadTimeLod === true ? "tiny-lod" : "bhatt-lod"}`,
  );
  return handle;
}
