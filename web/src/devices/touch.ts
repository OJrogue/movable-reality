// Lane devices: touch controls for phones, tablets and touch screens. One finger looks
// around, two fingers pan and pinch (gestures.ts). Works with any profile, because a
// touch laptop on the desktop profile gets the same gestures; never during an XR session.
//
// Event order: the ui lane's Editor and this module both listen for pointerdown on window
// in the capture phase. installTouch runs after installUi, so the Editor sees every touch
// first (listeners on one target run in registration order, and stopPropagation does not
// skip them). When the Editor takes a touch (dragging the selected object) it calls
// preventDefault, and this module leaves that finger alone. Every other touch is stopped
// here before it reaches SparkControls' listener on the canvas, whose own two-finger
// handling only works when both fingers land within a few milliseconds and rolls the
// camera on a pinch.
import * as THREE from "three";
import { DEFAULT_TOUCH, look, twoFinger, type Point, type TouchTuning } from "./gestures";

export interface TouchDeps {
  canvas: HTMLCanvasElement;
  rig: THREE.Object3D;
  camera: THREE.Camera;
  /** False during XR sessions and while bench mode drives the rig. */
  enabled(): boolean;
}

const _euler = new THREE.Euler(0, 0, 0, "YXZ");
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();

export class TouchControls {
  /** Fingers this module tracks, in landing order. */
  private readonly touches = new Map<number, Point>();
  /** Fingers the Editor took (object drag); further fingers are ignored meanwhile. */
  private readonly editorTouches = new Set<number>();
  readonly tuning: TouchTuning;
  /** Gestures seen since load, for tests: window.mrDevice.touch. */
  readonly counts = { look: 0, pan: 0, pinch: 0 };

  constructor(
    private readonly deps: TouchDeps,
    tuning: Partial<TouchTuning> = {},
  ) {
    this.tuning = { ...DEFAULT_TOUCH, ...tuning };
    // The browser must not scroll or zoom the page on a touch over the canvas.
    deps.canvas.style.touchAction = "none";
    window.addEventListener("pointerdown", (e) => this.onDown(e), { capture: true });
    window.addEventListener("pointermove", (e) => this.onMove(e));
    window.addEventListener("pointerup", (e) => this.onUp(e));
    window.addEventListener("pointercancel", (e) => this.onUp(e));
  }

  get active(): number {
    return this.touches.size;
  }

  private onDown(e: PointerEvent): void {
    if (e.pointerType !== "touch" || e.target !== this.deps.canvas) return;
    if (!this.deps.enabled()) return;
    if (e.defaultPrevented) {
      this.editorTouches.add(e.pointerId);
      return;
    }
    e.stopPropagation(); // SparkControls never sees it
    if (this.editorTouches.size > 0) return; // a second finger while an object is dragged
    this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
    try {
      this.deps.canvas.setPointerCapture(e.pointerId);
    } catch {
      // synthetic events (tests) have no active pointer to capture
    }
  }

  private onMove(e: PointerEvent): void {
    const prev = this.touches.get(e.pointerId);
    if (!prev) return;
    const cur = { x: e.clientX, y: e.clientY };
    if (!this.deps.enabled()) {
      this.touches.set(e.pointerId, cur);
      return;
    }
    const ids = [...this.touches.keys()];
    if (ids.length === 1) {
      this.applyLook(cur.x - prev.x, cur.y - prev.y);
    } else {
      // The first two fingers drive the gesture; a third is tracked but ignored.
      const [a, b] = ids;
      if (e.pointerId === a || e.pointerId === b) {
        const other = this.touches.get(e.pointerId === a ? b : a)!;
        const step =
          e.pointerId === a
            ? twoFinger(prev, other, cur, other, this.tuning)
            : twoFinger(other, prev, other, cur, this.tuning);
        this.applyMove(step.right, step.up, step.forward);
      }
    }
    this.touches.set(e.pointerId, cur);
  }

  private onUp(e: PointerEvent): void {
    this.editorTouches.delete(e.pointerId);
    if (!this.touches.delete(e.pointerId)) return;
    if (this.deps.canvas.hasPointerCapture?.(e.pointerId)) {
      this.deps.canvas.releasePointerCapture(e.pointerId);
    }
  }

  private applyLook(dx: number, dy: number): void {
    if (dx === 0 && dy === 0) return;
    const rig = this.deps.rig;
    _euler.setFromQuaternion(rig.quaternion, "YXZ");
    const next = look(_euler.y, _euler.x, dx, dy, this.tuning);
    _euler.set(next.pitch, next.yaw, 0, "YXZ");
    rig.quaternion.setFromEuler(_euler);
    this.counts.look++;
  }

  private applyMove(right: number, up: number, forward: number): void {
    const rig = this.deps.rig;
    const cam = this.deps.camera;
    cam.updateMatrixWorld(true);
    cam.getWorldQuaternion(_q);
    if (right !== 0 || up !== 0) {
      rig.position.add(_v.set(right, up, 0).applyQuaternion(_q));
      this.counts.pan++;
    }
    if (forward !== 0) {
      rig.position.add(_v.set(0, 0, -forward).applyQuaternion(_q));
      this.counts.pinch++;
    }
  }
}
