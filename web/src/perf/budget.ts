// Frame-time budget controller: a pure state machine, no three.js or Spark imports.
//
// Input: one frame interval per frame (ms). Output: Spark LOD settings (lodSplatCount and
// lodRenderScale) for a detail level 0..levels (levels = most detail). Each level is a
// constant factor in splat count, so "n levels" means "n times that factor".
//
// On a headset, frame intervals are quantised to the display period (13.9 ms at 72 Hz,
// 27.8 ms for one missed vsync), so the controller cannot see headroom directly. It:
//   - steps down when frames are missed. Clear overload (share of slow frames over a full
//     short window above `fastShare`): `downStep` levels. Severe overload (share above
//     `severeShare`, or slow frames of 3 or more periods): a frame of k periods cost
//     between k-1 and k periods, so it sheds the geometric middle of that range,
//     sqrt(k(k-1)), in splats, at least `downStep` and at most `maxDownStep` levels. An
//     overshoot is cheap (probes climb back); a shortfall costs another window of missed
//     frames. Marginal overload (share over a long window above `slowShare`): one level;
//   - after `upWindowMs` with (almost) no slow frames, probes up. Frames count from the
//     first frame of a probe, so a failed probe costs a few missed frames, not a cooldown's
//     worth. A probe fails on an overload or on more than `probeMaxSlow` slow frames within
//     `probeHoldMs`, and goes back to the level it came from. The probe step doubles after
//     each probe that holds (1, 2, 4 ... `maxUpStep` levels) and resets after any step down,
//     so it climbs back quickly after a heavy view; a level that has failed before is only
//     ever probed on its own;
//   - remembers failures: every step down from level L blocks probes into L for
//     `probeBackoffMs`, doubling per failure up to `maxBackoffMs`. A level's failure count
//     clears once the controller has held it (or a higher level) for `stableMs`. This is
//     what stops it oscillating around the highest level the device can just sustain;
//   - ignores frames for `cooldownMs` after every step down, because Spark applies new LOD
//     settings asynchronously (worker traversal, then sort) and the frames right after a
//     change still carry the old cost.
// Up and down use different windows and thresholds (hysteresis). Time is the sum of the
// intervals fed in, so a recorded trace replays identically.

export interface BudgetConfig {
  /** Frame-time target in ms: 13.9 for 72 Hz, 11.1 for 90 Hz. */
  targetMs: number;
  /** Number of steps between the least (0) and the most (levels) detail. */
  levels: number;
  startLevel: number;
  /** lodSplatCount bounds; the count is geometric in the level. */
  splatMin: number;
  splatMax: number;
  /**
   * lodRenderScale bounds (Spark accepts 1 to 5; higher = coarser splats). It stays at
   * renderScaleMin from level `renderScaleKnee * levels` upward and rises linearly to
   * renderScaleMax at level 0, so only the lower levels trade splat size for speed.
   */
  renderScaleMin: number;
  renderScaleMax: number;
  renderScaleKnee: number;
  /** A frame is slow when its interval exceeds targetMs * slowFactor (a missed vsync). */
  slowFactor: number;
  /** Clear overload: share of slow frames over the last fastWindowMs (window full). */
  fastWindowMs: number;
  fastShare: number;
  /** Severe overload: a fast-window share at or above this sheds in proportion to k. */
  severeShare: number;
  /** Marginal overload: share of slow frames over the last slowWindowMs (window full). */
  slowWindowMs: number;
  slowShare: number;
  /** A window needs at least this many slow frames before it can trigger a step down. */
  minSlowFrames: number;
  downStep: number;
  maxDownStep: number;
  /** Probe up after this long with a slow-frame share at or under upShare. */
  upWindowMs: number;
  upShare: number;
  maxUpStep: number;
  cooldownMs: number;
  probeHoldMs: number;
  probeMaxSlow: number;
  probeBackoffMs: number;
  maxBackoffMs: number;
  stableMs: number;
  /** Intervals longer than this are pauses (hidden tab, headset off, load), not load. */
  pauseMs: number;
}

export const DEFAULT_BUDGET: Readonly<BudgetConfig> = {
  targetMs: 13.9,
  levels: 20,
  startLevel: 10,
  splatMin: 150_000,
  splatMax: 1_500_000,
  renderScaleMin: 1,
  renderScaleMax: 3,
  renderScaleKnee: 0.5,
  slowFactor: 1.2,
  fastWindowMs: 500,
  fastShare: 0.25,
  severeShare: 0.75,
  slowWindowMs: 10_000,
  slowShare: 0.01,
  minSlowFrames: 4,
  downStep: 2,
  maxDownStep: 12,
  upWindowMs: 3000,
  upShare: 0.005,
  maxUpStep: 4,
  cooldownMs: 500,
  probeHoldMs: 3000,
  probeMaxSlow: 1,
  probeBackoffMs: 8000,
  maxBackoffMs: 120_000,
  stableMs: 30_000,
  pauseMs: 250,
};

export interface BudgetSettings {
  level: number;
  splatCount: number;
  renderScale: number;
}

export type BudgetPhase = "hold" | "cooldown" | "probing";
export type BudgetReason = "overload" | "marginal" | "severe" | "probe" | "probe-failed";

export interface BudgetStep extends BudgetSettings {
  phase: BudgetPhase;
  /** Set on the frame where the level changed. */
  reason?: BudgetReason;
}

/** Spark settings for a level: splat count geometric, render scale piecewise linear. */
export function settingsForLevel(level: number, cfg: BudgetConfig): BudgetSettings {
  const l = clamp(Math.round(level), 0, cfg.levels);
  const q = l / cfg.levels;
  const splatCount = Math.round(cfg.splatMin * Math.pow(cfg.splatMax / cfg.splatMin, q));
  const knee = cfg.renderScaleKnee;
  const below = knee > 0 ? clamp((knee - q) / knee, 0, 1) : 0;
  const renderScale = cfg.renderScaleMin + (cfg.renderScaleMax - cfg.renderScaleMin) * below;
  return { level: l, splatCount, renderScale: Math.round(renderScale * 1000) / 1000 };
}

/** The level whose splat count is closest to `count` (geometrically). */
export function levelForSplatCount(count: number, cfg: BudgetConfig): number {
  if (!(count > 0)) return clamp(Math.round(cfg.startLevel), 0, cfg.levels);
  if (cfg.splatMax === cfg.splatMin) return cfg.levels;
  const q = Math.log(count / cfg.splatMin) / Math.log(cfg.splatMax / cfg.splatMin);
  return clamp(Math.round(q * cfg.levels), 0, cfg.levels);
}

/** Time-based window of frame intervals with a running count of slow frames. */
class SlowWindow {
  private dts: number[] = [];
  private slows: boolean[] = [];
  private head = 0;
  private sum = 0;
  slowCount = 0;

  constructor(readonly spanMs: number) {}

  push(dt: number, slow: boolean): void {
    this.dts.push(dt);
    this.slows.push(slow);
    this.sum += dt;
    if (slow) this.slowCount++;
    while (this.dts.length - this.head > 1 && this.sum - this.dts[this.head] >= this.spanMs) {
      this.sum -= this.dts[this.head];
      if (this.slows[this.head]) this.slowCount--;
      this.head++;
    }
    if (this.head > 1024) {
      this.dts = this.dts.slice(this.head);
      this.slows = this.slows.slice(this.head);
      this.head = 0;
    }
  }

  clear(): void {
    this.dts = [];
    this.slows = [];
    this.head = 0;
    this.sum = 0;
    this.slowCount = 0;
  }

  /** True once the window covers at least `fraction` of its span. */
  filled(fraction: number): boolean {
    return this.sum >= this.spanMs * fraction;
  }

  get share(): number {
    const n = this.dts.length - this.head;
    return n > 0 ? this.slowCount / n : 0;
  }

  /** Median interval of the slow frames in the window (0 if none). */
  medianSlow(): number {
    const s: number[] = [];
    for (let i = this.head; i < this.dts.length; i++) if (this.slows[i]) s.push(this.dts[i]);
    if (s.length === 0) return 0;
    s.sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
  }
}

export class BudgetController {
  readonly cfg: BudgetConfig;
  private level: number;
  private ceiling: number;
  private nowMs = 0;
  private phase: BudgetPhase = "hold";
  private cooldownUntil = 0;
  private levelSince = 0;
  private probeFrom = -1;
  private probeStart = 0;
  private probeSlow = 0;
  private upStep = 1;
  private readonly fast: SlowWindow;
  private readonly slow: SlowWindow;
  private readonly up: SlowWindow;
  private readonly failures: number[];
  private readonly backoffUntil: number[];

  constructor(cfg: Partial<BudgetConfig> = {}) {
    this.cfg = { ...DEFAULT_BUDGET, ...cfg };
    const c = this.cfg;
    if (!(c.levels >= 1 && Number.isInteger(c.levels))) throw new Error("levels must be >= 1");
    if (!(c.splatMin > 0 && c.splatMax >= c.splatMin)) throw new Error("bad splat bounds");
    if (!(c.renderScaleMin > 0 && c.renderScaleMax >= c.renderScaleMin)) {
      throw new Error("bad render-scale bounds");
    }
    if (!(c.targetMs > 0)) throw new Error("targetMs must be > 0");
    this.ceiling = c.levels;
    this.level = clamp(Math.round(c.startLevel), 0, c.levels);
    this.fast = new SlowWindow(c.fastWindowMs);
    this.slow = new SlowWindow(c.slowWindowMs);
    this.up = new SlowWindow(c.upWindowMs);
    this.failures = new Array(c.levels + 1).fill(0);
    this.backoffUntil = new Array(c.levels + 1).fill(0);
  }

  get targetMs(): number {
    return this.cfg.targetMs;
  }

  /** Controller time: the sum of the intervals fed in, pauses included. */
  get timeMs(): number {
    return this.nowMs;
  }

  /** Highest level the controller may use (e.g. the scene has fewer splats than splatMax). */
  get maxLevel(): number {
    return this.ceiling;
  }

  setMaxLevel(level: number): void {
    this.ceiling = clamp(Math.round(level), 0, this.cfg.levels);
    if (this.level > this.ceiling) {
      this.level = this.ceiling;
      if (this.phase === "probing") this.phase = "hold";
    }
  }

  /** Change the target (e.g. once the XR frame rate is known); forgets recent frames. */
  setTarget(ms: number): void {
    if (!(ms > 0) || ms === this.cfg.targetMs) return;
    this.cfg.targetMs = ms;
    this.reset();
  }

  settings(): BudgetSettings {
    return settingsForLevel(this.level, this.cfg);
  }

  state(): BudgetStep {
    return { ...this.settings(), phase: this.phase };
  }

  /**
   * Forget recent frames (mode switch, XR enter or exit). Keeps the level and the failure
   * history; an unfinished probe goes back to the level it came from.
   */
  reset(): void {
    this.clearWindows();
    if (this.phase === "probing") this.level = this.probeFrom;
    this.phase = "hold";
    this.probeFrom = -1;
    this.upStep = 1;
    this.levelSince = this.nowMs;
  }

  /** Jump to a level with no history (start of a bench run, a new starting count). */
  restart(level = this.cfg.startLevel): void {
    this.clearWindows();
    this.phase = "hold";
    this.probeFrom = -1;
    this.upStep = 1;
    this.level = clamp(Math.round(level), 0, this.ceiling);
    this.failures.fill(0);
    this.backoffUntil.fill(0);
    this.cooldownUntil = 0;
    this.levelSince = this.nowMs;
  }

  /** Feed one frame interval; returns the settings to use from the next frame on. */
  update(dtMs: number): BudgetStep {
    const c = this.cfg;
    if (!(dtMs > 0) || !Number.isFinite(dtMs)) return this.state();
    this.nowMs += dtMs;
    if (dtMs > c.pauseMs) return this.state();
    if (this.phase === "cooldown") {
      if (this.nowMs < this.cooldownUntil) return this.state();
      this.phase = "hold";
    }

    const isSlow = dtMs > c.targetMs * c.slowFactor;
    this.fast.push(dtMs, isSlow);
    this.slow.push(dtMs, isSlow);
    this.up.push(dtMs, isSlow);

    const fastOver =
      this.fast.filled(0.9) &&
      this.fast.slowCount >= c.minSlowFrames &&
      this.fast.share > c.fastShare;
    const slowOver =
      this.slow.filled(1) &&
      this.slow.slowCount >= c.minSlowFrames &&
      this.slow.share > c.slowShare;

    if (this.phase === "probing") {
      if (isSlow) this.probeSlow++;
      if (fastOver || slowOver || this.probeSlow > c.probeMaxSlow) {
        return this.stepDown(this.probeFrom, "probe-failed");
      }
      if (this.nowMs - this.probeStart < c.probeHoldMs) return this.state();
      this.phase = "hold";
      this.upStep = Math.min(c.maxUpStep, this.upStep * 2);
    }

    if (fastOver || slowOver) {
      if (this.level > 0) {
        const k = Math.round(this.fast.medianSlow() / c.targetMs);
        if (fastOver && (this.fast.share >= c.severeShare || k >= 3)) {
          return this.stepDown(this.level - this.severeSteps(k), "severe");
        }
        if (fastOver) return this.stepDown(this.level - c.downStep, "overload");
        return this.stepDown(this.level - 1, "marginal");
      }
      // Already at the floor: nothing left to shed. Count afresh.
      this.clearWindows();
      return this.state();
    }

    if (this.nowMs - this.levelSince >= c.stableMs) {
      for (let l = 0; l <= this.level; l++) this.failures[l] = 0;
    }

    if (this.level < this.ceiling && this.up.filled(1) && this.up.share <= c.upShare) {
      // Climb at most upStep levels, never into or past a level still in backoff, and stop
      // at a level that has failed before so that it is tested on its own.
      const most = Math.min(this.upStep, this.ceiling - this.level);
      let step = 0;
      while (step < most) {
        const next = this.level + step + 1;
        if (this.nowMs < this.backoffUntil[next]) break;
        step++;
        if (this.failures[next] > 0) break;
      }
      if (step > 0) {
        this.probeFrom = this.level;
        this.level += step;
        this.phase = "probing";
        this.probeStart = this.nowMs;
        this.probeSlow = 0;
        this.levelSince = this.nowMs;
        // Keep the long window: a probe that adds slow frames slowly still fails on it.
        this.fast.clear();
        this.up.clear();
        return { ...this.settings(), phase: this.phase, reason: "probe" };
      }
    }
    return this.state();
  }

  /** Levels to shed on a severe overload whose slow frames took about k periods. */
  private severeSteps(periods: number): number {
    const c = this.cfg;
    const k = Math.max(2, periods);
    const perLevel = Math.log(c.splatMax / c.splatMin) / c.levels;
    if (!(perLevel > 0)) return c.maxDownStep;
    const steps = Math.round(Math.log(Math.sqrt(k * (k - 1))) / perLevel);
    return clamp(steps, c.downStep, c.maxDownStep);
  }

  private stepDown(level: number, reason: BudgetReason): BudgetStep {
    const c = this.cfg;
    const from = this.level;
    this.failures[from]++;
    const backoff = c.probeBackoffMs * Math.pow(2, this.failures[from] - 1);
    this.backoffUntil[from] = this.nowMs + Math.min(c.maxBackoffMs, backoff);
    this.level = clamp(level, 0, this.ceiling);
    this.upStep = 1;
    this.clearWindows();
    this.phase = "cooldown";
    this.cooldownUntil = this.nowMs + c.cooldownMs;
    this.levelSince = this.nowMs;
    return { ...this.settings(), phase: this.phase, reason };
  }

  private clearWindows(): void {
    this.fast.clear();
    this.slow.clear();
    this.up.clear();
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
