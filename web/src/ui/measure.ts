// Lane ui: tape measure. Alt+click two points in the room (splats, baked planes or meshes)
// and the distance between them shows in metres, with its horizontal and vertical parts.
// It shows off the metric claim without a tape: click the floor and the top of the sofa.
// A third Alt+click starts a new measurement; Escape clears it.
//
// Where a click lands: Spark's own raycast only tests splats of the current LOD cut as exact
// ellipsoids, and a ray through the middle of the bed's blanket often misses them all and lands on
// the floor behind. So the ray is tested as a thin cone against the splat centres of the whole
// LOD tree (cached on first use), and a surface needs a few splats to agree, which also skips
// lone floaters. Meshes (room shell, baked planes, library furniture) use the normal raycast.
// SparkRenderer is itself a Mesh, a 2 x 2 m quad at the world origin that nobody sees, and the
// gizmo has large invisible planes: neither may count, or long measurements end in mid-air.
import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";
import type { ViewerContext } from "../context";

export interface Measurement {
  total: number;
  horizontal: number;
  vertical: number;
}

export function measureBetween(a: THREE.Vector3, b: THREE.Vector3): Measurement {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dz = b.z - a.z;
  return { total: Math.hypot(dx, dy, dz), horizontal: Math.hypot(dx, dz), vertical: Math.abs(dy) };
}

export function formatMeasurement(m: Measurement): string {
  const cm = (v: number) => (v < 1 ? `${Math.round(v * 100)} cm` : `${v.toFixed(2)} m`);
  const parts = [cm(m.total)];
  // Show the parts only when the line is clearly slanted.
  if (m.vertical > 0.05 && m.horizontal > 0.05) parts.push(`(${cm(m.horizontal)} across, ${cm(m.vertical)} up)`);
  return parts.join(" ");
}

/**
 * Depth of the nearest surface along a ray from the distances of the splats near it: the first
 * spot where at least `support` splats lie within `band` metres of each other. Null if none does.
 */
export function surfaceDepth(distances: number[], support = 4, band = 0.03): number | null {
  const d = [...distances].sort((a, b) => a - b);
  for (let i = 0; i + support <= d.length; i++) {
    if (d[i + support - 1] - d[i] <= band) return d[i + Math.floor(support / 2)];
  }
  return null;
}

/** Splats wider than this (metres) are coarse LOD levels, too vague to measure against. */
const MAX_SPLAT_SCALE = 0.08;
const MIN_SPLAT_OPACITY = 0.3;
const NEAR_M = 0.05;

interface Centres {
  xyz: Float32Array;
  box: THREE.Box3;
}
const centresCache = new WeakMap<object, Centres>();

function splatCentres(mesh: SplatMesh): Centres | null {
  const source = mesh.packedSplats?.lodSplats ?? mesh.packedSplats;
  if (!source || !source.numSplats) return null;
  let c = centresCache.get(source);
  if (c) return c;
  const xyz = new Float32Array(source.numSplats * 3);
  const box = new THREE.Box3();
  let n = 0;
  source.forEachSplat((_i, centre, scales, _q, opacity) => {
    if (opacity < MIN_SPLAT_OPACITY || Math.max(scales.x, scales.y, scales.z) > MAX_SPLAT_SCALE) return;
    xyz[n++] = centre.x;
    xyz[n++] = centre.y;
    xyz[n++] = centre.z;
    box.expandByPoint(centre);
  });
  c = { xyz: xyz.subarray(0, n), box };
  centresCache.set(source, c);
  return c;
}

const _inv = new THREE.Matrix4();
const _localRay = new THREE.Ray();

/** World distances of the splats of one mesh inside a thin cone around the ray. */
function splatDistances(mesh: SplatMesh, ray: THREE.Ray, out: number[]): void {
  const c = splatCentres(mesh);
  if (!c) return;
  mesh.updateWorldMatrix(true, false);
  _inv.copy(mesh.matrixWorld).invert();
  _localRay.origin.copy(ray.origin).applyMatrix4(_inv);
  _localRay.direction.copy(ray.direction).transformDirection(_inv);
  if (!_localRay.intersectsBox(c.box.clone().expandByScalar(0.1))) return;
  const s = mesh.matrixWorld.getMaxScaleOnAxis();
  const { x: ox, y: oy, z: oz } = _localRay.origin;
  const { x: dx, y: dy, z: dz } = _localRay.direction;
  const { xyz } = c;
  for (let i = 0; i < xyz.length; i += 3) {
    const px = xyz[i] - ox;
    const py = xyz[i + 1] - oy;
    const pz = xyz[i + 2] - oz;
    const t = px * dx + py * dy + pz * dz;
    if (t <= 0) continue;
    const world = t * s;
    if (world < NEAR_M) continue;
    const tol = 0.012 + 0.004 * world; // about 4 px across at 1 m, and it widens with distance
    if ((px * px + py * py + pz * pz - t * t) * s * s < tol * tol) out.push(world);
  }
}

/** Nearest opaque mesh under the ray (room shell, planes, library furniture), not a helper. */
export function meshDistance(ctx: ViewerContext, raycaster: THREE.Raycaster): number | null {
  const h = raycaster
    .intersectObjects(ctx.scene.children, true)
    .find((x) => x.object instanceof THREE.Mesh && !(x.object instanceof SplatMesh) && x.object !== ctx.spark && shown(x.object));
  return h ? h.distance : null;
}

/** Where a ray meets the room: the nearer of the splat surface and the meshes. */
export function surfacePoint(ctx: ViewerContext, raycaster: THREE.Raycaster): THREE.Vector3 | null {
  const distances: number[] = [];
  ctx.scene.traverseVisible((o) => {
    if (o instanceof SplatMesh && shown(o)) splatDistances(o, raycaster.ray, distances);
  });
  const splat = surfaceDepth(distances);
  const mesh = meshDistance(ctx, raycaster);
  const d = splat === null ? mesh : mesh === null ? splat : Math.min(splat, mesh);
  return d === null ? null : raycaster.ray.at(d, new THREE.Vector3());
}


const COLOR = 0x3df5a6;

export function installMeasure(ctx: ViewerContext, notify: (text: string) => void): void {
  const canvas = ctx.renderer.domElement;
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const group = new THREE.Group();
  group.name = "measure";
  ctx.scene.add(group);
  const dotGeo = new THREE.SphereGeometry(0.015, 16, 12);
  const mat = new THREE.MeshBasicMaterial({ color: COLOR, depthTest: false, transparent: true });
  const lineMat = new THREE.LineBasicMaterial({ color: COLOR, depthTest: false, transparent: true });
  const label = document.createElement("div");
  label.className = "mr-measure";
  label.style.cssText =
    "position:fixed;pointer-events:none;transform:translate(-50%,-130%);padding:3px 8px;" +
    "border-radius:6px;background:rgba(0,0,0,.75);color:#3df5a6;font:600 14px system-ui;" +
    "white-space:nowrap;display:none;z-index:20";
  document.body.appendChild(label);
  let points: THREE.Vector3[] = [];

  const clear = () => {
    for (const c of [...group.children]) {
      group.remove(c);
      if (c instanceof THREE.Line) c.geometry.dispose();
    }
    points = [];
    label.style.display = "none";
  };

  const hit = (e: PointerEvent): THREE.Vector3 | null => {
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, ctx.camera);
    return surfacePoint(ctx, raycaster);
  };

  window.addEventListener(
    "pointerdown",
    (e) => {
      if (!e.altKey || e.button !== 0 || e.target !== canvas || ctx.renderer.xr.isPresenting) return;
      e.preventDefault();
      e.stopImmediatePropagation(); // no orbit, no pick
      const p = hit(e);
      if (!p) {
        notify("measure: nothing under the cursor");
        return;
      }
      if (points.length >= 2) clear();
      points.push(p);
      const dot = new THREE.Mesh(dotGeo, mat);
      dot.position.copy(p);
      dot.renderOrder = 999;
      group.add(dot);
      if (points.length === 2) {
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), lineMat);
        line.renderOrder = 999;
        group.add(line);
        const text = formatMeasurement(measureBetween(points[0], points[1]));
        label.textContent = text;
        label.style.display = "block";
        notify(`measure: ${text}`);
      } else {
        notify("measure: Alt+click the second point");
      }
    },
    { capture: true },
  );
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && points.length) clear();
  });

  const mid = new THREE.Vector3();
  const eye = new THREE.Vector3();
  ctx.onFrameEnd(() => {
    // Dots keep their size on screen, so the far end of a long line stays easy to see.
    ctx.camera.getWorldPosition(eye);
    for (const c of group.children) {
      if (c instanceof THREE.Mesh) c.scale.setScalar(Math.max(1, c.position.distanceTo(eye) * 0.4));
    }
    if (points.length !== 2) return;
    mid.copy(points[0]).add(points[1]).multiplyScalar(0.5).project(ctx.camera);
    const r = canvas.getBoundingClientRect();
    const behind = mid.z > 1;
    label.style.display = behind ? "none" : "block";
    label.style.left = `${r.left + ((mid.x + 1) / 2) * r.width}px`;
    label.style.top = `${r.top + ((1 - mid.y) / 2) * r.height}px`;
  });
}

/** Overlays drawn over the room, never measured: gap overlay, the tape itself, gizmo, selection box. */
const HELPER_GROUPS = new Set(["coverage-overlay", "measure", "ui-gizmo", "ui-selection"]);

/** three.js raycasts hidden objects too; skip anything under a hidden parent (splats hidden
 * by ?objects=meshonly) or under a helper overlay. */
function shown(o: THREE.Object3D): boolean {
  for (let p: THREE.Object3D | null = o; p; p = p.parent) {
    if (!p.visible || HELPER_GROUPS.has(p.name)) return false;
  }
  return true;
}

