// Lane ui, U1: desktop editing with mouse and keyboard. Click an object to select it (ray
// against the manifest boxes with lane lod's rayBoxDistance), drag it across a horizontal plane through
// the grab point (height unchanged), Q/E rotate by 15 degrees, Delete or Backspace hides,
// R restores the newest hidden object, Ctrl/Cmd+Z undoes, Escape clears the selection.
// Drags on empty space reach SparkControls as before. Nothing here runs during an XR session.
// Touch (lane devices): a tap selects the object under the finger (or clears the selection
// on empty space), and only the selected object can be dragged, so a one-finger look that
// starts over an object turns the view instead of moving the object (devices/touch.ts).
import * as THREE from "three";
import type { ViewerContext } from "../context";
import type { SceneObject } from "../scene";
import { halfSize, rayBoxDistance, type Pickable } from "../lod/pick";
import { isTap, TAP_SLOP_PX } from "../devices/gestures";
import { lastHidden, type EditRequest } from "./edits";

export const ROTATE_STEP_DEG = 15;
const CLICK_SLOP_PX = 4;

export interface EditorDeps {
  /** Sends an edit (server or local mirror); resolves once the new manifest is applied. */
  edit(req: EditRequest): Promise<void>;
  /** Short message for the user, e.g. "nothing to undo". */
  notify(text: string): void;
}

interface Drag {
  id: string;
  pointerId: number;
  plane: THREE.Plane;
  grab: THREE.Vector3; // grab point on the plane at pointerdown
  start: THREE.Vector3; // pivot position (bbox centre in world) at pointerdown
  x0: number;
  y0: number;
  moved: boolean;
}

export const isTyping = (t: EventTarget | null): boolean =>
  t instanceof HTMLElement && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName) || t.isContentEditable);

export class Editor {
  selected: string | null = null;
  private drag: Drag | null = null;
  /** A press that may become a click: on empty space (id null) or, by touch, on an object. */
  private emptyDown: {
    pointerId: number;
    x: number;
    y: number;
    id: string | null;
    slop: number;
  } | null = null;
  private readonly raycaster = new THREE.Raycaster();
  private readonly ndc = new THREE.Vector2();
  private readonly outline: THREE.LineSegments;
  private readonly label: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;

  constructor(
    private readonly ctx: ViewerContext,
    private readonly deps: EditorDeps,
  ) {
    this.canvas = ctx.renderer.domElement;
    const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    const mat = new THREE.LineBasicMaterial({ color: 0xffc83d, depthTest: false, transparent: true });
    this.outline = new THREE.LineSegments(edges, mat);
    this.outline.name = "ui-selection";
    this.outline.renderOrder = 10;
    this.label = document.createElement("div");
    this.label.className = "mr-ui-label";
    this.label.hidden = true;
    document.body.appendChild(this.label);

    // Capture phase on window runs before SparkControls' listeners on the canvas and the
    // document, so a press on an object can be kept from orbiting the camera.
    window.addEventListener("pointerdown", (e) => this.onPointerDown(e), { capture: true });
    window.addEventListener("pointermove", (e) => this.onPointerMove(e));
    window.addEventListener("pointerup", (e) => this.onPointerUp(e));
    window.addEventListener("pointercancel", (e) => this.onPointerUp(e, true));
    window.addEventListener("keydown", (e) => this.onKey(e), { capture: true });

    ctx.onManifest((m) => {
      const o = this.selected ? m.objects.find((x) => x.id === this.selected) : undefined;
      if (this.selected && (!o || !o.visible)) this.select(null);
      else if (o) this.attachOutline(o);
    });
    ctx.onFrame(() => this.updateLabel());
  }

  private get xr(): boolean {
    return this.ctx.renderer.xr.isPresenting;
  }

  private objects(): SceneObject[] {
    return this.ctx.manifest?.objects ?? [];
  }

  private rayAt(clientX: number, clientY: number): THREE.Ray {
    const r = this.canvas.getBoundingClientRect();
    this.ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.ctx.rig.updateMatrixWorld(true);
    this.raycaster.setFromCamera(this.ndc, this.ctx.camera);
    return this.raycaster.ray;
  }

  private pickables(): Pickable[] {
    const view = this.ctx.view;
    if (!view) return [];
    view.root.updateMatrixWorld(true);
    const out: Pickable[] = [];
    for (const o of this.objects()) {
      const node = view.objectNode(o.id);
      if (!o.visible || !node) continue;
      out.push({ id: o.id, pivotWorld: node.pivot.matrixWorld, half: halfSize(o) });
    }
    return out;
  }

  /**
   * Nearest box the ray enters. Boxes that contain the camera are skipped: picking them
   * would grab that object on every click and leave no way to orbit.
   */
  private pick(ray: THREE.Ray): { id: string; point: THREE.Vector3 } | null {
    let best: { id: string; distance: number } | null = null;
    for (const p of this.pickables()) {
      const d = rayBoxDistance(ray, p.pivotWorld, p.half);
      if (d !== null && d > 0 && (best === null || d < best.distance)) best = { id: p.id, distance: d };
    }
    return best ? { id: best.id, point: ray.at(best.distance, new THREE.Vector3()) } : null;
  }

  select(id: string | null): void {
    this.selected = id;
    const o = id ? this.objects().find((x) => x.id === id) : undefined;
    if (o) this.attachOutline(o);
    else {
      this.outline.removeFromParent();
      this.label.hidden = true;
    }
  }

  private attachOutline(o: SceneObject): void {
    const node = this.ctx.view?.objectNode(o.id);
    if (!node) return;
    const h = halfSize(o);
    this.outline.scale.set(Math.max(2 * h[0], 1e-3), Math.max(2 * h[1], 1e-3), Math.max(2 * h[2], 1e-3));
    if (this.outline.parent !== node.pivot) node.pivot.add(this.outline);
    this.label.textContent = `${o.label}  ${o.id}`;
    this.label.title = "drag to move · Q/E rotate · Delete hide · Ctrl+Z undo";
  }

  private updateLabel(): void {
    const id = this.selected;
    const node = id ? this.ctx.view?.objectNode(id) : undefined;
    const o = id ? this.objects().find((x) => x.id === id) : undefined;
    if (!node || !o || this.xr) {
      this.label.hidden = true;
      return;
    }
    const top = new THREE.Vector3(0, halfSize(o)[1], 0).applyMatrix4(node.pivot.matrixWorld);
    top.project(this.ctx.camera);
    if (top.z > 1 || top.z < -1) {
      this.label.hidden = true;
      return;
    }
    const r = this.canvas.getBoundingClientRect();
    this.label.hidden = false;
    this.label.style.left = `${r.left + ((top.x + 1) / 2) * r.width}px`;
    this.label.style.top = `${r.top + ((1 - top.y) / 2) * r.height}px`;
  }

  // ------------------------------------------------------------ pointer

  private onPointerDown(e: PointerEvent): void {
    if (this.xr || e.target !== this.canvas || e.button !== 0 || this.drag) return;
    const touch = e.pointerType === "touch";
    if (touch && !e.isPrimary) {
      // A second finger makes this a pan or pinch (devices/touch.ts), never a tap.
      this.emptyDown = null;
      return;
    }
    // preventDefault below would keep focus in the command bar, where Q, E, Delete and
    // Space would then type instead of editing.
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    const ray = this.rayAt(e.clientX, e.clientY);
    const hit = this.pick(ray);
    if (!hit || (touch && hit.id !== this.selected)) {
      const slop = touch ? TAP_SLOP_PX : CLICK_SLOP_PX;
      this.emptyDown = { pointerId: e.pointerId, x: e.clientX, y: e.clientY, id: hit?.id ?? null, slop };
      return;
    }
    e.stopPropagation(); // keep SparkControls from orbiting
    e.preventDefault();
    const node = this.ctx.view?.objectNode(hit.id);
    if (!node) return;
    this.select(hit.id);
    this.canvas.setPointerCapture(e.pointerId);
    this.drag = {
      id: hit.id,
      pointerId: e.pointerId,
      plane: new THREE.Plane(new THREE.Vector3(0, 1, 0), -hit.point.y),
      grab: hit.point.clone(),
      start: node.pivot.position.clone(),
      x0: e.clientX,
      y0: e.clientY,
      moved: false,
    };
    // ctx.controlsEnabled stays on: SparkControls never saw this press, so it does not
    // orbit, and pausing it would leave a stale timestamp that jumps the rig afterwards.
    this.canvas.style.cursor = "grabbing";
  }

  private onPointerMove(e: PointerEvent): void {
    const d = this.drag;
    if (!d) {
      if (!this.xr && e.target === this.canvas && e.buttons === 0) {
        this.canvas.style.cursor = this.pick(this.rayAt(e.clientX, e.clientY)) ? "grab" : "";
      }
      return;
    }
    if (e.pointerId !== d.pointerId) return;
    if (!d.moved && Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < CLICK_SLOP_PX) return;
    d.moved = true;
    const p = this.rayAt(e.clientX, e.clientY).intersectPlane(d.plane, new THREE.Vector3());
    const node = this.ctx.view?.objectNode(d.id);
    if (!p || !node) return; // ray parallel to or pointing away from the plane
    node.pivot.position.set(d.start.x + p.x - d.grab.x, d.start.y, d.start.z + p.z - d.grab.z);
  }

  private onPointerUp(e: PointerEvent, cancelled = false): void {
    const empty = this.emptyDown?.pointerId === e.pointerId ? this.emptyDown : null;
    if (empty) this.emptyDown = null;
    const d = this.drag;
    if (!d) {
      // A click (no drag) on empty space clears the selection; an orbit drag keeps it. A
      // touch tap on an object selects it.
      if (empty && !cancelled && e.target === this.canvas) {
        if (isTap(empty, { x: e.clientX, y: e.clientY }, empty.slop)) this.select(empty.id);
      }
      return;
    }
    if (e.pointerId !== d.pointerId) return;
    this.drag = null;
    this.canvas.style.cursor = "grab";
    if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    const node = this.ctx.view?.objectNode(d.id);
    if (!node) return;
    if (cancelled || !d.moved) {
      node.pivot.position.copy(d.start);
      return;
    }
    const x = round(node.pivot.position.x);
    const z = round(node.pivot.position.z);
    // On failure the edit path re-applies the last good manifest, which snaps the object back.
    void this.deps.edit({ op: "place", object_id: d.id, args: { x, z } });
  }

  // ------------------------------------------------------------ keys

  private onKey(e: KeyboardEvent): void {
    if (this.xr || isTyping(e.target)) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.code === "KeyZ" && !e.shiftKey && !e.altKey) {
      e.preventDefault();
      e.stopPropagation();
      if (!e.repeat) void this.deps.edit({ op: "undo", object_id: null });
      return;
    }
    if (mod || e.altKey) return;
    if (e.code === "KeyR") {
      if (e.repeat) return;
      const m = this.ctx.manifest;
      const id = m ? lastHidden(m) : null;
      if (id) void this.deps.edit({ op: "restore", object_id: id });
      else this.deps.notify("nothing hidden to restore");
      return;
    }
    const id = this.selected;
    if (!id) return;
    let req: EditRequest | null = null;
    // Q/E also move SparkControls down/up; with a selection they rotate the object instead.
    if (e.code === "KeyQ") req = { op: "rotate", object_id: id, args: { yaw_deg: ROTATE_STEP_DEG } };
    else if (e.code === "KeyE") req = { op: "rotate", object_id: id, args: { yaw_deg: -ROTATE_STEP_DEG } };
    else if (e.code === "Delete" || e.code === "Backspace") req = { op: "remove", object_id: id };
    else if (e.code === "Escape") {
      this.select(null);
      return;
    }
    if (!req) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.repeat || this.drag) return;
    if (req.op === "remove") this.select(null);
    void this.deps.edit(req);
  }
}

const round = (v: number) => Math.round(v * 1000) / 1000; // millimetres are plenty
