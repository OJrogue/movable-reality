import { describe, expect, it } from "vitest";
import { DEFAULT_BUDGET, settingsForLevel, type BudgetConfig } from "../perf/budget";
import { logLabelParam } from "../perf/frameLog";
import { levelAt, overShare, simulate } from "../perf/simulate";
import {
  DEVICE_SETTINGS,
  detectDevice,
  FLOOR_HOLD_MS,
  floorHz,
  parseDeviceParam,
  pixelRatioFor,
  selectDevice,
  settingsFor,
  targetHzFor,
  type DeviceEnv,
} from "./profile";

const UA = {
  mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36",
  windows: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36",
  iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
  android: "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Mobile Safari/537.36",
  // iPadOS Safari asks for desktop sites and reports a Mac user agent.
  ipad: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
  steamFrame: "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36",
};

const env = (o: Partial<DeviceEnv>): DeviceEnv => ({
  xrPresenting: false,
  coarsePointer: false,
  screenShortSide: 900,
  userAgent: UA.mac,
  ...o,
});

describe("parseDeviceParam", () => {
  it("accepts the three profile names, any case", () => {
    expect(parseDeviceParam("desktop")).toBe("desktop");
    expect(parseDeviceParam(" Mobile ")).toBe("mobile");
    expect(parseDeviceParam("XR")).toBe("xr");
  });
  it("leaves other values (benchmark log labels) alone", () => {
    expect(parseDeviceParam(null)).toBeNull();
    expect(parseDeviceParam("")).toBeNull();
    expect(parseDeviceParam("steam-frame")).toBeNull();
    expect(logLabelParam("steam-frame")).toBe("steam-frame");
    expect(logLabelParam("mobile")).toBeNull();
    expect(logLabelParam(null)).toBeNull();
  });
});

describe("detectDevice and selectDevice", () => {
  it("laptop with mouse or trackpad: desktop", () => {
    expect(detectDevice(env({}))).toBe("desktop");
    expect(detectDevice(env({ userAgent: UA.windows, screenShortSide: 1440 }))).toBe("desktop");
  });
  it("phone: mobile by user agent and by coarse pointer on a small screen", () => {
    expect(detectDevice(env({ userAgent: UA.iphone, coarsePointer: true, screenShortSide: 393 }))).toBe("mobile");
    expect(detectDevice(env({ userAgent: UA.android, coarsePointer: false, screenShortSide: 412 }))).toBe("mobile");
  });
  it("iPad with a desktop user agent: mobile through the coarse pointer", () => {
    expect(detectDevice(env({ userAgent: UA.ipad, coarsePointer: true, screenShortSide: 1024 }))).toBe("mobile");
  });
  it("touch monitor larger than a tablet with a desktop browser: desktop", () => {
    expect(detectDevice(env({ userAgent: UA.windows, coarsePointer: true, screenShortSide: 1440 }))).toBe("desktop");
  });
  it("an XR session always means xr, even against the URL", () => {
    expect(selectDevice(env({ xrPresenting: true, userAgent: UA.steamFrame }), null)).toBe("xr");
    expect(selectDevice(env({ xrPresenting: true }), "desktop")).toBe("xr");
    expect(selectDevice(env({ xrPresenting: true, userAgent: UA.android }), "mobile")).toBe("xr");
  });
  it("outside XR, ?device= wins over detection", () => {
    expect(selectDevice(env({}), "mobile")).toBe("mobile");
    expect(selectDevice(env({ userAgent: UA.iphone, coarsePointer: true, screenShortSide: 390 }), "desktop")).toBe("desktop");
    expect(selectDevice(env({}), "xr")).toBe("xr");
    expect(selectDevice(env({}), null)).toBe("desktop");
  });
});

describe("settings per profile", () => {
  const cfgFor = (name: "desktop" | "mobile" | "xr"): BudgetConfig => ({
    ...DEFAULT_BUDGET,
    ...settingsFor(name).budget,
  });

  it("xr keeps the renderer lane's 72 Hz settings exactly", () => {
    const xr = settingsFor("xr");
    expect(xr.budget).toEqual({});
    expect(cfgFor("xr")).toEqual(DEFAULT_BUDGET);
    expect(xr.defaultTargetMs).toBe(13.9);
    expect(xr.targetHz).toBeNull();
    expect(xr.pixelRatioCap).toBe(2); // main.ts used min(devicePixelRatio, 2) before
    expect(xr.lod).toBe(true);
    expect(targetHzFor(xr, 90)).toBe(90);
  });

  it("desktop: full pixel ratio, a higher budget than xr, starting at the top", () => {
    const d = settingsFor("desktop");
    expect(pixelRatioFor(d, 3)).toBe(3);
    expect(pixelRatioFor(d, 2)).toBe(2);
    const c = cfgFor("desktop");
    expect(c.startLevel).toBe(c.levels);
    expect(c.splatMax).toBeGreaterThan(DEFAULT_BUDGET.splatMax);
    expect(c.splatMin).toBeGreaterThan(DEFAULT_BUDGET.splatMin);
    expect(settingsForLevel(c.startLevel, c).splatCount).toBe(6_000_000);
    expect(d.defaultTargetMs).toBe(16.7);
  });

  it("desktop and mobile follow the display rate but never aim above 60 Hz", () => {
    for (const name of ["desktop", "mobile"] as const) {
      const s = settingsFor(name);
      expect(targetHzFor(s, 60)).toBe(60);
      expect(targetHzFor(s, 120)).toBe(60);
      expect(targetHzFor(s, 144)).toBe(60);
      expect(targetHzFor(s, 24)).toBe(30);
    }
  });

  it("mobile: pixel ratio capped at 1.5 and a budget between xr and desktop", () => {
    const m = settingsFor("mobile");
    expect(pixelRatioFor(m, 3)).toBe(1.5);
    expect(pixelRatioFor(m, 1)).toBe(1);
    expect(pixelRatioFor(m, 0)).toBe(1); // a broken devicePixelRatio
    const c = cfgFor("mobile");
    const start = settingsForLevel(c.startLevel, c).splatCount;
    expect(start).toBeGreaterThan(450_000);
    expect(start).toBeLessThan(700_000);
    expect(c.splatMax).toBeLessThan(cfgFor("desktop").splatMax);
  });

  it("settingsFor returns copies, so a caller cannot change the table", () => {
    const s = settingsFor("desktop");
    s.budget.splatMax = 1;
    s.targetHz!.max = 1;
    expect(DEVICE_SETTINGS.desktop.budget.splatMax).toBe(6_000_000);
    expect(DEVICE_SETTINGS.desktop.targetHz!.max).toBe(60);
  });
});

describe("floorHz (30 Hz fallback)", () => {
  const mobile = settingsFor("mobile");
  it("drops a struggling phone to 30 Hz only at level 0, mostly missed, held 3 s", () => {
    expect(floorHz(mobile, 16.7, 0, 0.6, FLOOR_HOLD_MS)).toBe(30);
    expect(floorHz(mobile, 16.7, 1, 0.9, 10_000)).toBeNull(); // the budget can still go down
    expect(floorHz(mobile, 16.7, 0, 0.3, 10_000)).toBeNull(); // most frames arrive
    expect(floorHz(mobile, 16.7, 0, 0.9, FLOOR_HOLD_MS - 1)).toBeNull();
  });
  it("does nothing once at 30 Hz, and never in XR", () => {
    expect(floorHz(mobile, 33.3, 0, 0.9, 10_000)).toBeNull();
    expect(floorHz(settingsFor("xr"), 13.9, 0, 0.9, 10_000)).toBeNull();
  });
  it("never lowers a desktop target: a laptop steps its budget down instead", () => {
    expect(floorHz(settingsFor("desktop"), 16.7, 0, 0.9, 10_000)).toBeNull();
  });
});

describe("desktop budget on the simulated display", () => {
  const period = 16.7;
  // simulate() takes the target from periodMs; DEFAULT_BUDGET's 13.9 ms must not override it.
  const desktop = { ...DEFAULT_BUDGET, ...settingsFor("desktop").budget, targetMs: period };

  it("a laptop that holds 60 Hz at full detail never steps down", () => {
    const sim = simulate({ seconds: 60, capacity: () => 8_000_000, periodMs: period, cfg: desktop });
    expect(sim.changes.filter((c) => c.reason !== "probe")).toEqual([]);
    expect(levelAt(sim, 59)).toBe(desktop.levels);
  });

  it("a laptop that cannot steps down only after missing frames, then holds the rate", () => {
    const sim = simulate({ seconds: 90, capacity: () => 1_500_000, periodMs: period, cfg: desktop });
    expect(sim.changes.length).toBeGreaterThan(0);
    const first = sim.changes[0];
    expect(["overload", "severe", "marginal"]).toContain(first.reason);
    // The first step down comes after at least one frame over 1.2x the period.
    const firstMiss = sim.dts.findIndex((dt) => dt > period * 1.2);
    expect(firstMiss).toBeGreaterThanOrEqual(0);
    expect(sim.t[firstMiss]).toBeLessThanOrEqual(first.t);
    expect(overShare(sim, 60, Infinity, period)).toBeLessThan(0.02);
    expect(settingsForLevel(levelAt(sim, 89), desktop).splatCount).toBeLessThanOrEqual(1_500_000);
  });
});
