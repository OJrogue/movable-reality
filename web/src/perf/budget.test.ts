import { describe, expect, it } from "vitest";
import {
  BudgetController,
  DEFAULT_BUDGET,
  levelForSplatCount,
  settingsForLevel,
} from "./budget";
import { levelAt, overShare, simulate, type SimResult } from "./simulate";

const cfg = { ...DEFAULT_BUDGET };
const splatsAt = (level: number) => settingsForLevel(level, cfg).splatCount;
/** Highest level whose splat count fits a capacity with the given headroom. */
const bestLevel = (capacity: number, headroom = 1) => {
  let best = 0;
  for (let l = 0; l <= cfg.levels; l++) if (splatsAt(l) * headroom <= capacity) best = l;
  return best;
};
/** Share of time spent at each level from `fromS` on. */
const timeShare = (sim: SimResult, fromS: number) => {
  const at = new Map<number, number>();
  let total = 0;
  for (let i = 0; i < sim.t.length; i++) {
    if (sim.t[i] < fromS * 1000) continue;
    at.set(sim.levels[i], (at.get(sim.levels[i]) ?? 0) + sim.dts[i]);
    total += sim.dts[i];
  }
  return new Map([...at].map(([l, v]) => [l, v / total]));
};

// ---- level mapping -------------------------------------------------------------------

describe("settingsForLevel", () => {
  it("spans the configured bounds and is monotonic", () => {
    const lo = settingsForLevel(0, cfg);
    const hi = settingsForLevel(cfg.levels, cfg);
    expect(lo.splatCount).toBe(cfg.splatMin);
    expect(hi.splatCount).toBe(cfg.splatMax);
    expect(lo.renderScale).toBe(cfg.renderScaleMax);
    expect(hi.renderScale).toBe(cfg.renderScaleMin);
    for (let l = 1; l <= cfg.levels; l++) {
      const a = settingsForLevel(l - 1, cfg);
      const b = settingsForLevel(l, cfg);
      expect(b.splatCount).toBeGreaterThan(a.splatCount);
      expect(b.renderScale).toBeLessThanOrEqual(a.renderScale);
    }
  });

  it("keeps full splat resolution above the knee", () => {
    for (let l = cfg.levels / 2; l <= cfg.levels; l++) {
      expect(settingsForLevel(l, cfg).renderScale).toBe(1);
    }
  });

  it("clamps out-of-range levels", () => {
    expect(settingsForLevel(-5, cfg).level).toBe(0);
    expect(settingsForLevel(99, cfg).level).toBe(cfg.levels);
  });

  it("inverts through levelForSplatCount", () => {
    for (let l = 0; l <= cfg.levels; l++) expect(levelForSplatCount(splatsAt(l), cfg)).toBe(l);
    expect(levelForSplatCount(1e9, cfg)).toBe(cfg.levels);
    expect(levelForSplatCount(1, cfg)).toBe(0);
    expect(levelForSplatCount(Number.NaN, cfg)).toBe(cfg.startLevel);
  });
});

describe("BudgetController config", () => {
  it("rejects bad bounds", () => {
    expect(() => new BudgetController({ splatMin: 0 })).toThrow();
    expect(() => new BudgetController({ splatMin: 2e6, splatMax: 1e6 })).toThrow();
    expect(() => new BudgetController({ renderScaleMax: 0.5 })).toThrow();
    expect(() => new BudgetController({ levels: 0 })).toThrow();
    expect(() => new BudgetController({ targetMs: 0 })).toThrow();
  });
});

// ---- simulated traces (see simulate.ts: vsynced display, 200 ms settings lag) ---------

describe("BudgetController on simulated traces", () => {
  it("overload: sheds detail within 2 s and then stays inside the budget", () => {
    const capacity = 400_000; // the full 1.5M would take 4 periods per frame
    const sim = simulate({
      seconds: 120,
      capacity: () => capacity,
      noise: 0.03,
      cfg: { startLevel: cfg.levels },
    });
    expect(sim.changes[0].reason).toBe("severe");
    expect(sim.changes[0].t).toBeLessThan(1000);
    expect(splatsAt(levelAt(sim, 2.5))).toBeLessThanOrEqual(capacity);
    // Settles within one level (12% in splats) of the best level that fits.
    const settled = levelAt(sim, 119);
    expect(settled).toBeGreaterThanOrEqual(bestLevel(capacity, 1.1) - 1);
    expect(splatsAt(settled)).toBeLessThanOrEqual(capacity);
    expect(overShare(sim, 10)).toBeLessThan(0.01);
  });

  it("recovery: climbs back after the load goes away", () => {
    // Heavy view for 20 s (capacity 250k), then a light one (capacity 1.2M).
    const sim = simulate({
      seconds: 90,
      capacity: (t) => (t < 20_000 ? 250_000 : 1_200_000),
      noise: 0.03,
    });
    expect(splatsAt(levelAt(sim, 19))).toBeLessThanOrEqual(250_000);
    const target = bestLevel(1_200_000, 1.1) - 1;
    const i = sim.levels.findIndex((l, j) => sim.t[j] > 20_000 && l >= target);
    expect(i).toBeGreaterThan(-1);
    expect(sim.t[i] - 20_000).toBeLessThan(30_000);
    expect(overShare(sim, 30)).toBeLessThan(0.01);
  });

  it("noise: jitter and rare hitches do not make it oscillate", () => {
    const capacity = 700_000;
    for (const seed of [1, 2, 3, 7]) {
      const sim = simulate({
        seconds: 600,
        capacity: () => capacity,
        noise: 0.06,
        spikeRate: 0.002,
        seed,
      });
      // Nine minutes after warm-up: one change every 25 s at most, no sawtooth.
      expect(sim.changes.filter((c) => c.t > 60_000).length).toBeLessThanOrEqual(22);
      const shares = [...timeShare(sim, 60).values()];
      expect(Math.max(...shares)).toBeGreaterThan(0.85);
      // Hitches are 0.2% of frames; load adds little on top.
      expect(overShare(sim, 60)).toBeLessThan(0.006);
      // Never sheds far below a level that fits with 25% headroom.
      const from = sim.t.findIndex((t) => t > 60_000);
      expect(Math.min(...sim.levels.slice(from))).toBeGreaterThanOrEqual(
        bestLevel(capacity, 1.25) - 2,
      );
    }
  });

  it("step changes: reacts to a sudden heavy view within 1 s and returns afterwards", () => {
    // 30 s light, 15 s heavy (an object fills the view), then light again.
    const heavy = (t: number) => t >= 30_000 && t < 45_000;
    const sim = simulate({
      seconds: 120,
      capacity: (t) => (heavy(t) ? 200_000 : 900_000),
      noise: 0.03,
    });
    const reaction = sim.changes.find((c) => c.t >= 30_000 && c.reason !== "probe");
    expect(reaction).toBeDefined();
    expect(reaction!.t - 30_000).toBeLessThan(1200);
    expect(splatsAt(levelAt(sim, 33))).toBeLessThanOrEqual(200_000);
    expect(overShare(sim, 34, 45)).toBeLessThan(0.01);
    expect(levelAt(sim, 119)).toBeGreaterThanOrEqual(bestLevel(900_000, 1.1) - 2);
  });

  it("never leaves its bounds", () => {
    const tiny = simulate({ seconds: 30, capacity: () => 50_000 });
    expect(Math.min(...tiny.levels)).toBe(0);
    expect(tiny.ctl.settings().splatCount).toBe(cfg.splatMin);
    expect(tiny.ctl.settings().renderScale).toBe(cfg.renderScaleMax);
    const huge = simulate({ seconds: 120, capacity: () => 1e8 });
    expect(Math.max(...huge.levels)).toBe(cfg.levels);
    expect(huge.ctl.settings().splatCount).toBe(cfg.splatMax);
    expect(huge.changes.every((c) => c.reason === "probe")).toBe(true);
  });

  it("respects a lower ceiling (scene smaller than splatMax)", () => {
    const sim = simulate({ seconds: 60, capacity: () => 1e8, cfg: { startLevel: 2 } });
    expect(levelAt(sim, 59)).toBe(cfg.levels);
    const ctl = new BudgetController();
    ctl.setMaxLevel(levelForSplatCount(300_000, cfg));
    for (let i = 0; i < 10_000; i++) ctl.update(13.9);
    expect(ctl.settings().level).toBe(ctl.maxLevel);
    expect(ctl.settings().splatCount).toBeLessThanOrEqual(340_000);
  });

  it("uses an 11.1 ms target for 90 Hz", () => {
    const capacity = 500_000;
    const sim = simulate({
      seconds: 60,
      periodMs: 11.1,
      capacity: () => capacity,
      cfg: { targetMs: 11.1, startLevel: cfg.levels },
    });
    expect(splatsAt(levelAt(sim, 59))).toBeLessThanOrEqual(capacity);
    expect(overShare(sim, 10, Infinity, 11.1)).toBeLessThan(0.01);
  });
});

describe("BudgetController edge cases", () => {
  it("treats long intervals as pauses, not load", () => {
    const ctl = new BudgetController();
    const start = ctl.settings().level;
    for (let i = 0; i < 20; i++) ctl.update(2000); // tab hidden, headset off
    expect(ctl.settings().level).toBe(start);
  });

  it("ignores a single hitch", () => {
    const ctl = new BudgetController();
    const start = ctl.settings().level;
    for (let i = 0; i < 100; i++) ctl.update(13.9);
    ctl.update(200);
    ctl.update(41.7);
    for (let i = 0; i < 50; i++) ctl.update(13.9);
    expect(ctl.settings().level).toBe(start);
  });

  it("holds still during the cooldown after a step down", () => {
    const ctl = new BudgetController({ startLevel: 20 });
    const steps: number[] = [];
    let t = 0;
    while (t < 5000 && steps.length < 2) {
      t += 27.8;
      if (ctl.update(27.8).reason) steps.push(t);
    }
    expect(steps.length).toBe(2);
    expect(steps[1] - steps[0]).toBeGreaterThanOrEqual(DEFAULT_BUDGET.cooldownMs);
  });

  it("ignores bad input", () => {
    const ctl = new BudgetController();
    const before = ctl.state();
    for (const v of [0, -5, Number.NaN, Infinity]) expect(ctl.update(v)).toEqual(before);
  });

  it("restart and setTarget clear the history", () => {
    const ctl = new BudgetController({ startLevel: 20 });
    for (let i = 0; i < 200; i++) ctl.update(41.7);
    expect(ctl.settings().level).toBeLessThan(20);
    ctl.restart(15);
    expect(ctl.state()).toMatchObject({ level: 15, phase: "hold" });
    ctl.setTarget(11.1);
    expect(ctl.targetMs).toBe(11.1);
    // At 90 Hz a 13.9 ms frame is a missed frame.
    for (let i = 0; i < 100; i++) ctl.update(13.9);
    expect(ctl.settings().level).toBeLessThan(15);
  });

  it("reset during a probe returns to the level the probe came from", () => {
    const ctl = new BudgetController({ startLevel: 5 });
    let probed = false;
    for (let i = 0; i < 400 && !probed; i++) probed = ctl.update(13.9).reason === "probe";
    expect(probed).toBe(true);
    expect(ctl.state()).toMatchObject({ level: 6, phase: "probing" });
    ctl.reset();
    expect(ctl.state()).toMatchObject({ level: 5, phase: "hold" });
  });
});
