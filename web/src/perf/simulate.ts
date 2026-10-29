// Simulated headset for budget-controller tests: render cost grows linearly with the
// splats Spark draws; `capacity(t)` is the splat count that just fits one display period.
// The display is vsynced, so an interval is a whole number of periods. New settings reach
// the GPU `lagMs` after the controller emits them (Spark's LOD traversal and sort run in
// workers). Test support only; nothing in the viewer imports it.
import { BudgetController, type BudgetConfig, type BudgetReason } from "./budget";

export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(r: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
}

export interface SimOptions {
  seconds: number;
  /** Splats that fit one period at time t (ms). */
  capacity: (tMs: number) => number;
  periodMs?: number;
  lagMs?: number;
  /** Relative std-dev of the per-frame cost. */
  noise?: number;
  /** Probability of a one-frame hitch (a missed vsync unrelated to load). */
  spikeRate?: number;
  seed?: number;
  cfg?: Partial<BudgetConfig>;
}

export interface SimResult {
  /** End time of every frame, ms. */
  t: number[];
  dts: number[];
  /** Level in force after each frame. */
  levels: number[];
  changes: { t: number; reason: BudgetReason; level: number }[];
  ctl: BudgetController;
}

export function simulate(o: SimOptions): SimResult {
  const period = o.periodMs ?? 13.9;
  const lag = o.lagMs ?? 200;
  const r = mulberry32(o.seed ?? 1);
  const ctl = new BudgetController({ targetMs: period, ...o.cfg });
  const pending: { at: number; splats: number }[] = [];
  let applied = ctl.settings().splatCount;
  let now = 0;
  const out: SimResult = { t: [], dts: [], levels: [], changes: [], ctl };
  while (now < o.seconds * 1000) {
    while (pending.length && pending[0].at <= now) applied = pending.shift()!.splats;
    const cost = (period * applied * (1 + (o.noise ?? 0) * gauss(r))) / o.capacity(now);
    let dt = period * Math.max(1, Math.ceil(cost / period - 1e-9));
    if (o.spikeRate && r() < o.spikeRate) dt += period;
    now += dt;
    const step = ctl.update(dt);
    if (step.reason) {
      out.changes.push({ t: now, reason: step.reason, level: step.level });
      pending.push({ at: now + lag, splats: step.splatCount });
    }
    out.t.push(now);
    out.dts.push(dt);
    out.levels.push(step.level);
  }
  return out;
}

/** Share of frames over 1.2x the target (a missed vsync) with end time in [fromS, toS). */
export function overShare(sim: SimResult, fromS: number, toS = Infinity, targetMs = 13.9): number {
  let n = 0;
  let over = 0;
  for (let i = 0; i < sim.t.length; i++) {
    if (sim.t[i] < fromS * 1000 || sim.t[i] >= toS * 1000) continue;
    n++;
    if (sim.dts[i] > targetMs * 1.2) over++;
  }
  return n ? over / n : 0;
}

/** Level in force at time s (seconds). */
export function levelAt(sim: SimResult, s: number): number {
  const i = sim.t.findIndex((t) => t >= s * 1000);
  return sim.levels[i === -1 ? sim.levels.length - 1 : i];
}
