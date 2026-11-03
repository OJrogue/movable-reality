// Lane ui: move and rotate gizmo on the selected object, as in a game engine (three.js
// TransformControls). Move drags along one axis (red X, green Y up, blue Z) or across a
// plane (the squares); Rotate turns the object about world +Y, the only rotation a scene
// edit holds. T or the buttons top centre switch the mode, Shift snaps (10 cm, 15 deg).
// Editor routes the pointer here: TransformControls gets no listeners of its own, so a
// press on a handle never reaches SparkControls or the editor's own object drag.
import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { EditRequest } from "./edits";

export type GizmoMode = "translate" | "rotate";

export const SNAP_M = 0.1;
export const SNAP_DEG = 15;

/** Yaw of a rotation about +Y in degrees (the angle the object's +X axis turned). */
export function yawDeg(q: THREE.Quaternion): number {
  const x = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
  return THREE.MathUtils.radToDeg(Math.atan2(-x.z, x.x));
}

/** The scene edit a finished gizmo drag stands for, or null when it barely moved. */
export function gizmoEdit(
  id: string,
  mode: GizmoMode,
  start: { position: THREE.Vector3; quaternion: THREE.Quaternion },
  end: { position: THREE.Vector3; quaternion: THREE.Quaternion },
): EditRequest | null {
  const round = (v: number, step: number) => Math.round(v / step) * step;
  if (mode === "rotate") {
    let d = yawDeg(end.quaternion) - yawDeg(start.quaternion);
    d = ((((d + 180) % 360) + 360) % 360) - 180; // (-180, 180]
    d = round(d, 0.1);
    return Math.abs(d) < 0.05 ? null : { op: "rotate", object_id: id, args: { yaw_deg: d } };
  }
  const v = end.position.clone().sub(start.position);
  const [dx, dy, dz] = [v.x, v.y, v.z].map((c) => round(c, 0.001));
  if (Math.hypot(dx, dy, dz) < 0.001) return null;
  return { op: "move", object_id: id, args: { dx, dy, dz } };
}

// TransformControls' pointer methods take {x, y, button} in NDC (the typings say PointerEvent);
// a move carries button -1, as a browser pointermove does, or it is ignored.
const ptr = (ndc: THREE.Vector2, button = 0) => ({ x: ndc.x, y: ndc.y, button }) as unknown as PointerEvent;

export class Gizmo {
  readonly controls: TransformControls;
  private id: string | null = null;
  private start: { position: THREE.Vector3; quaternion: THREE.Quaternion } | null = null;
  private readonly bar: HTMLDivElement;
  private readonly buttons: Record<GizmoMode, HTMLButtonElement>;

  constructor(
    scene: THREE.Scene,
    camera: THREE.Camera,
    private readonly onEdit: (req: EditRequest) => void,
  ) {
    this.controls = new TransformControls(camera);
    this.controls.setSpace("world");
    this.controls.setSize(0.9);
    const helper = this.controls.getHelper();
    helper.name = "ui-gizmo";
    helper.renderOrder = 11;
    scene.add(helper);
    this.bar = document.createElement("div");
    this.bar.className = "mr-ui-gizmo";
    this.bar.hidden = true;
    const button = (mode: GizmoMode, text: string, title: string) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = text;
      b.title = title;
      b.addEventListener("click", () => this.setMode(mode));
      this.bar.append(b);
      return b;
    };
    this.buttons = {
      translate: button("translate", "Move", "Move along an axis or a plane (T switches, Shift snaps 10 cm)"),
      rotate: button("rotate", "Rotate", "Turn about the vertical axis (T switches, Shift snaps 15°)"),
    };
    document.body.appendChild(this.bar);
    this.setMode("translate");
  }

  get dragging(): boolean {
    return this.controls.dragging;
  }

  get mode(): GizmoMode {
    return this.controls.mode as GizmoMode;
  }

  setMode(mode: GizmoMode): void {
    this.controls.setMode(mode);
    this.controls.showX = mode === "translate";
    this.controls.showZ = mode === "translate";
    this.controls.showY = true;
    for (const [m, b] of Object.entries(this.buttons)) b.classList.toggle("on", m === mode);
  }

  toggleMode(): void {
    this.setMode(this.mode === "translate" ? "rotate" : "translate");
  }

  /** Show the gizmo on an object's pivot (the group that carries its edit), or hide it. */
  attach(id: string | null, pivot: THREE.Object3D | null): void {
    if (this.controls.dragging) return;
    this.id = id;
    if (id && pivot) {
      if (this.controls.object !== pivot) this.controls.attach(pivot);
      this.bar.hidden = false;
    } else {
      this.controls.detach();
      this.bar.hidden = true;
    }
  }

  /** NDC pointer: hover highlights a handle; returns whether one is under the pointer. */
  hover(ndc: THREE.Vector2): boolean {
    if (!this.controls.object) return false;
    this.controls.pointerHover(ptr(ndc));
    return this.controls.axis !== null;
  }

  /** Starts a drag when a handle is under the pointer. */
  down(ndc: THREE.Vector2, snap: boolean): boolean {
    if (!this.hover(ndc)) return false;
    this.setSnap(snap);
    const o = this.controls.object!;
    this.start = { position: o.position.clone(), quaternion: o.quaternion.clone() };
    this.controls.pointerDown(ptr(ndc));
    return this.controls.dragging;
  }

  move(ndc: THREE.Vector2, snap: boolean): void {
    this.setSnap(snap);
    this.controls.pointerMove(ptr(ndc, -1));
  }

  /** Ends a drag: sends the edit, or puts the object back when cancelled. */
  up(cancelled: boolean): void {
    const o = this.controls.object;
    const start = this.start;
    this.controls.pointerUp(ptr(new THREE.Vector2()));
    this.start = null;
    if (!o || !start || !this.id) return;
    const req = gizmoEdit(this.id, this.mode, start, { position: o.position, quaternion: o.quaternion });
    if (cancelled || !req) {
      o.position.copy(start.position);
      o.quaternion.copy(start.quaternion);
      return;
    }
    // On failure the edit path re-applies the last good manifest, which snaps the object back.
    this.onEdit(req);
  }

  private setSnap(snap: boolean): void {
    this.controls.setTranslationSnap(snap ? SNAP_M : null);
    this.controls.setRotationSnap(snap ? THREE.MathUtils.degToRad(SNAP_DEG) : null);
  }
}
