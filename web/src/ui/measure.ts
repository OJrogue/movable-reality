// Lane ui: tape measure. Alt+click two points in the room (splats, baked planes or meshes)
// and the distance between them shows in metres, with its horizontal and vertical parts.
// It shows off the metric claim without a tape: click the floor and the top of the sofa.
// A third Alt+click starts a new measurement; Escape clears it.
import * as THREE from "three";
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
    const hits = raycaster
      .intersectObjects(ctx.scene.children, true)
      .filter((h) => shown(h.object) && !isInside(h.object, group) && !(h.object instanceof THREE.Line));
    return hits[0]?.point.clone() ?? null;
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
  ctx.onFrameEnd(() => {
    if (points.length !== 2) return;
    mid.copy(points[0]).add(points[1]).multiplyScalar(0.5).project(ctx.camera);
    const r = canvas.getBoundingClientRect();
    const behind = mid.z > 1;
    label.style.display = behind ? "none" : "block";
    label.style.left = `${r.left + ((mid.x + 1) / 2) * r.width}px`;
    label.style.top = `${r.top + ((1 - mid.y) / 2) * r.height}px`;
  });
}

/** three.js raycasts hidden objects too; skip anything under a hidden parent (gap overlay,
 * splats hidden by ?objects=meshonly) and translucent helpers. */
function shown(o: THREE.Object3D): boolean {
  for (let p: THREE.Object3D | null = o; p; p = p.parent) if (!p.visible) return false;
  const m = (o as THREE.Mesh).material as THREE.Material | undefined;
  return !(m && !Array.isArray(m) && m.transparent && m.opacity < 0.9 && o.type === "Mesh");
}

function isInside(o: THREE.Object3D, root: THREE.Object3D): boolean {
  for (let p: THREE.Object3D | null = o; p; p = p.parent) if (p === root) return true;
  return false;
}
