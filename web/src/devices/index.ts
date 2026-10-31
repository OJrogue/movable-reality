// Lane devices: one viewer for laptops, phones, tablets and standalone headsets, each at the
// best quality it can hold. main.ts calls installDevices(ctx) before the other lanes, so
// they read ctx.device at install time, and installTouch(ctx) after the ui lane (event
// order, see touch.ts).
//
// URL parameter: ?device=desktop|mobile|xr picks the profile outside an XR session
// (default: detected, see profile.ts). An XR session always uses xr. Any other value is
// left to the renderer lane, which uses ?device=<label> to name benchmark logs.
//
// window.mrDevice: the active profile name, its settings, where the choice came from and
// the touch gesture counts, for console checks and tests.
import type { ViewerContext } from "../context";
import type { DeviceSettings } from "./profile";
import {
  parseDeviceParam,
  pixelRatioFor,
  selectDevice,
  settingsFor,
  type DeviceEnv,
  type DeviceName,
} from "./profile";
import { TouchControls } from "./touch";

export type DevicesContext = ViewerContext & { runDeviceChange(s: DeviceSettings): void };

export interface DeviceHandle {
  readonly name: DeviceName;
  readonly settings: DeviceSettings;
  /** "xr" (session running), "url" (?device=) or "auto" (detected). */
  readonly source: "xr" | "url" | "auto";
  readonly env: DeviceEnv;
  readonly pixelRatio: number;
  readonly touch: { active: number; look: number; pan: number; pinch: number } | null;
}

function readEnv(xrPresenting: boolean): DeviceEnv {
  const coarse = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  return {
    xrPresenting,
    coarsePointer: coarse,
    // Some embedded browsers report a 0 x 0 screen; the window size still counts then.
    screenShortSide: Math.min(
      ...[screen.width, screen.height, window.innerWidth, window.innerHeight].filter((v) => v > 0),
    ),
    userAgent: navigator.userAgent,
  };
}

let touchControls: TouchControls | null = null;

/** The HUD (index.html, #hud) in a smaller font so it fits a portrait phone. */
const DEVICE_CSS = `
body[data-device="mobile"] #hud {
  font-size: 10px; max-width: calc(100vw - 24px); overflow: hidden; padding: 4px 8px;
  top: calc(8px + env(safe-area-inset-top, 0px)); left: 8px;
}
`;

export function installDevices(ctx: DevicesContext): DeviceHandle {
  const override = parseDeviceParam(ctx.params.get("device"));
  const renderer = ctx.renderer;
  let env = readEnv(renderer.xr.isPresenting);

  const apply = (name: DeviceName) => {
    const s = settingsFor(name);
    // three.js refuses size changes while presenting; the XR framebuffer has its own scale.
    if (!renderer.xr.isPresenting) {
      const ratio = pixelRatioFor(s, window.devicePixelRatio);
      if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
    }
    ctx.spark.sortRadial = s.sortRadial;
    ctx.spark.minSortIntervalMs = s.minSortIntervalMs;
    document.body.dataset.device = name;
    ctx.runDeviceChange(s);
    console.info(`[devices] profile ${name} (${handle.source})`);
  };

  const update = () => {
    const name = selectDevice(env, override);
    if (name !== ctx.device?.name) apply(name);
    else if (!renderer.xr.isPresenting) {
      // Same profile, but the window may have moved to a screen with another ratio.
      const ratio = pixelRatioFor(ctx.device, window.devicePixelRatio);
      if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
    }
  };

  const handle: DeviceHandle = {
    get name() {
      return ctx.device!.name;
    },
    get settings() {
      return ctx.device!;
    },
    get source() {
      return env.xrPresenting ? "xr" : override ? "url" : "auto";
    },
    get env() {
      return { ...env };
    },
    get pixelRatio() {
      return renderer.getPixelRatio();
    },
    get touch() {
      return touchControls ? { active: touchControls.active, ...touchControls.counts } : null;
    },
  };

  // Listeners registered here run before the renderer lane's (installPerf comes later), so
  // perf sees the xr profile when it handles the same sessionstart.
  renderer.xr.addEventListener("sessionstart", () => {
    env = { ...env, xrPresenting: true };
    update();
  });
  renderer.xr.addEventListener("sessionend", () => {
    env = readEnv(false);
    update();
  });
  window.addEventListener("resize", () => {
    if (renderer.xr.isPresenting) return;
    env = readEnv(false);
    update();
  });
  // A touch keyboard opening or a pointer being plugged in can change the coarse query.
  if (typeof matchMedia === "function") {
    matchMedia("(pointer: coarse)").addEventListener?.("change", () => {
      if (renderer.xr.isPresenting) return;
      env = readEnv(false);
      update();
    });
  }

  const style = document.createElement("style");
  style.textContent = DEVICE_CSS;
  document.head.appendChild(style);
  apply(selectDevice(env, override));
  (window as unknown as { mrDevice: DeviceHandle }).mrDevice = handle;
  return handle;
}

/** Touch gestures (one finger look, two fingers pan and pinch). Call after installUi. */
export function installTouch(ctx: ViewerContext): TouchControls {
  touchControls = new TouchControls({
    canvas: ctx.renderer.domElement,
    rig: ctx.rig,
    camera: ctx.camera,
    enabled: () => ctx.controlsEnabled && !ctx.renderer.xr.isPresenting,
  });
  return touchControls;
}

export { DEVICE_SETTINGS, settingsFor, selectDevice, detectDevice } from "./profile";
