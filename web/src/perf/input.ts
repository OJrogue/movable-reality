// Keys and XR controller buttons for the perf tools. Edge-triggered: one action per press.
//
// Desktop keys (Spark's controls use WASD, QE and the arrows, so these are free):
//   M  naive / LOD toggle      H  HUD on / off
//   B  budget controller on / off      L  save the frame log (POST in dev, else download)
// XR, either controller (xr-standard gamepad mapping; Spark's default XR controls are not
// enabled in this viewer, so the buttons are free):
//   A or X (button 4)          naive / LOD toggle
//   B or Y (button 5)          HUD on / off
//   thumbstick press (button 3) save the frame log
import type * as THREE from "three";

export type PerfAction = "mode" | "hud" | "budget" | "save";

const KEYS: Record<string, PerfAction> = {
  KeyM: "mode",
  KeyH: "hud",
  KeyB: "budget",
  KeyL: "save",
};

const XR_BUTTONS: Record<number, PerfAction> = { 4: "mode", 5: "hud", 3: "save" };

export class PerfInput {
  private readonly queue: PerfAction[] = [];
  private readonly held = new WeakMap<XRInputSource, boolean[]>();

  constructor(private readonly renderer: THREE.WebGLRenderer) {
    document.addEventListener("keydown", (e) => {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      const action = KEYS[e.code];
      if (action) this.queue.push(action);
    });
  }

  /** Actions since the last call: queued key presses plus new XR button presses. */
  poll(): PerfAction[] {
    const session = this.renderer.xr.getSession();
    for (const source of session?.inputSources ?? []) {
      const pad = source.gamepad;
      if (!pad || pad.mapping !== "xr-standard") continue;
      const prev = this.held.get(source) ?? [];
      const now = pad.buttons.map((b) => b.pressed);
      for (const [i, action] of Object.entries(XR_BUTTONS)) {
        const k = Number(i);
        if (now[k] && !prev[k]) this.queue.push(action);
      }
      this.held.set(source, now);
    }
    return this.queue.splice(0);
  }
}
