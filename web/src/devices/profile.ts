// Lane devices: device profiles. Pure logic, no three.js or Spark imports, so it is tested
// on the CPU (profile.test.ts).
//
// Three profiles set the quality knobs the viewer has:
//   desktop  laptop or desktop browser: full device pixel ratio, a high splat budget that
//            starts at the top and only steps down when frames are missed, a frame target
//            from the display refresh (60 Hz until it is measured, never above 60 Hz);
//   mobile   phone or tablet browser: pixel ratio capped at 1.5, a smaller splat budget,
//            a 30 to 60 Hz frame target;
//   xr       WebXR session on a standalone headset: exactly the 72 Hz settings the
//            renderer lane tuned (DEFAULT_BUDGET in perf/budget.ts, 13.9 ms, pixel ratio 2).
// Selection: a running WebXR session always means xr. Otherwise ?device=desktop|mobile|xr
// wins, and without it a coarse pointer on a small screen, or a mobile user agent, means
// mobile; everything else is desktop.
import type { BudgetConfig } from "../perf/budget";

export type DeviceName = "desktop" | "mobile" | "xr";
export const DEVICE_NAMES: readonly DeviceName[] = ["desktop", "mobile", "xr"];

export interface DeviceSettings {
  name: DeviceName;
  /** renderer.setPixelRatio(min(devicePixelRatio, pixelRatioCap)). Infinity = full ratio. */
  pixelRatioCap: number;
  /** LOD on (Spark LOD trees, budget controller) unless ?mode=naive asks for full splats. */
  lod: boolean;
  /**
   * Budget controller overrides on top of DEFAULT_BUDGET: the splat range and the starting
   * level. Empty for xr, so the headset keeps the tuned controller unchanged.
   */
  budget: Partial<Pick<BudgetConfig, "splatMin" | "splatMax" | "startLevel">>;
  /** Frame-time target before the display rate is known (ms). */
  defaultTargetMs: number;
  /**
   * Range for a target taken from the measured display rate. A 120 Hz laptop keeps a
   * 60 Hz target, so its splat budget is not cut to reach 120 fps. Null: no clamp (xr).
   */
  targetHz: { min: number; max: number } | null;
  /** SparkRenderer sort options (Spark's defaults; listed so the HUD and tests see them). */
  sortRadial: boolean;
  minSortIntervalMs: number;
}

const LEVELS = 20; // DEFAULT_BUDGET.levels; startLevel below is on this scale

export const DEVICE_SETTINGS: Readonly<Record<DeviceName, Readonly<DeviceSettings>>> = {
  desktop: {
    name: "desktop",
    pixelRatioCap: Infinity,
    lod: true,
    // 400k to 6M splats; start at the top level (capped at the scene's own size once the
    // PLY headers are read), so a laptop that holds the frame rate shows every splat.
    budget: { splatMin: 400_000, splatMax: 6_000_000, startLevel: LEVELS },
    defaultTargetMs: 16.7,
    targetHz: { min: 30, max: 60 },
    sortRadial: true,
    minSortIntervalMs: 0,
  },
  mobile: {
    name: "mobile",
    pixelRatioCap: 1.5,
    lod: true,
    // 100k to 1.5M (Spark's own defaults are 1M on Android, 1.5M on iOS). Start near 600k.
    budget: { splatMin: 100_000, splatMax: 1_500_000, startLevel: 13 },
    defaultTargetMs: 16.7,
    targetHz: { min: 30, max: 60 },
    sortRadial: true,
    minSortIntervalMs: 0,
  },
  xr: {
    name: "xr",
    pixelRatioCap: 2,
    lod: true,
    budget: {},
    defaultTargetMs: 13.9,
    targetHz: null,
    sortRadial: true,
    minSortIntervalMs: 0,
  },
};

/** What detection looks at; main.ts reads it from the browser, tests pass it in. */
export interface DeviceEnv {
  /** A WebXR session is presenting. */
  xrPresenting: boolean;
  /** matchMedia("(pointer: coarse)") for the primary pointer. */
  coarsePointer: boolean;
  /** Shorter side of the screen in CSS pixels. */
  screenShortSide: number;
  userAgent: string;
}

/** Phones and tablets: tablets reach 1024 CSS px on the short side (iPad Pro 12.9"). */
export const SMALL_SCREEN_PX = 1024;
const MOBILE_UA = /Mobi|Android|iPhone|iPad|iPod|Opera Mini|IEMobile/;

export function parseDeviceParam(value: string | null): DeviceName | null {
  const v = value?.trim().toLowerCase() ?? "";
  return (DEVICE_NAMES as readonly string[]).includes(v) ? (v as DeviceName) : null;
}

/** The profile outside an XR session, from the browser alone. */
export function detectDevice(env: Omit<DeviceEnv, "xrPresenting">): DeviceName {
  if (env.coarsePointer && env.screenShortSide > 0 && env.screenShortSide <= SMALL_SCREEN_PX) {
    return "mobile";
  }
  return MOBILE_UA.test(env.userAgent) ? "mobile" : "desktop";
}

/**
 * Profile in use. A WebXR session always runs the xr profile (a headset frame budget does
 * not depend on what the page was opened with). Outside a session the URL override wins.
 */
export function selectDevice(env: DeviceEnv, override: DeviceName | null): DeviceName {
  if (env.xrPresenting) return "xr";
  return override ?? detectDevice(env);
}

export function settingsFor(name: DeviceName): DeviceSettings {
  const s = DEVICE_SETTINGS[name];
  return { ...s, budget: { ...s.budget }, targetHz: s.targetHz ? { ...s.targetHz } : null };
}

export function pixelRatioFor(s: DeviceSettings, devicePixelRatio: number): number {
  const dpr = devicePixelRatio > 0 ? devicePixelRatio : 1;
  return Math.min(dpr, s.pixelRatioCap);
}

/** Display rate to aim for: the measured rate clamped to the profile's range. */
export function targetHzFor(s: DeviceSettings, displayHz: number): number {
  if (!s.targetHz) return displayHz;
  return Math.min(s.targetHz.max, Math.max(s.targetHz.min, displayHz));
}

/** Time at the lowest budget level, with most frames missed, before the target drops. */
export const FLOOR_HOLD_MS = 3000;
export const FLOOR_MISSED_SHARE = 0.5;

/**
 * Lower rate for a device that misses most frames even at the lowest budget level (a slow
 * phone that cannot hold 60 Hz). Returns the profile's minimum rate when the target should
 * drop to it, else null. Never in XR (targetHz null): the headset keeps its display rate.
 */
export function floorHz(
  s: DeviceSettings,
  currentTargetMs: number,
  level: number,
  missedShare: number,
  heldMs: number,
): number | null {
  if (!s.targetHz || currentTargetMs >= 1000 / s.targetHz.min - 0.1) return null;
  if (level > 0 || missedShare < FLOOR_MISSED_SHARE || heldMs < FLOOR_HOLD_MS) return null;
  return s.targetHz.min;
}
