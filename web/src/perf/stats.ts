// Frame-time statistics shared by the HUD, the frame log and the tests. Pure functions only.
// Percentiles use linear interpolation between closest ranks, the same definition as
// numpy.percentile's default, so scripts/bench_report.py reports the same numbers.

/** Percentile p in [0, 100] of an ascending-sorted array; NaN for an empty array. */
export function percentileSorted(sorted: ArrayLike<number>, p: number): number {
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  const rank = (Math.min(100, Math.max(0, p)) / 100) * (n - 1);
  const lo = Math.floor(rank);
  const hi = Math.min(n - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

export function percentile(values: ArrayLike<number>, p: number): number {
  return percentileSorted(Float64Array.from(values).sort(), p);
}

/** A frame counts as missed (over budget) when its interval exceeds this multiple of the
 * target. The budget controller uses the same factor. Raw intervals jitter around the
 * display period (13.8 to 14.1 ms at 72 Hz), so a strict comparison would count jitter. */
export const MISS_FACTOR = 1.2;

export interface FrameSummary {
  frames: number;
  p50: number;
  p99: number;
  /** Frames per second over the summed intervals. */
  fps: number;
  /** Share of frames whose interval exceeds missFactor * targetMs (missed vsync). */
  missedShare: number;
}

export function summarize(
  dts: ArrayLike<number>,
  targetMs: number,
  missFactor = MISS_FACTOR,
): FrameSummary {
  const sorted = Float64Array.from(dts).sort();
  let sum = 0;
  let missed = 0;
  for (const dt of sorted) {
    sum += dt;
    if (dt > targetMs * missFactor) missed++;
  }
  const n = sorted.length;
  return {
    frames: n,
    p50: percentileSorted(sorted, 50),
    p99: percentileSorted(sorted, 99),
    fps: sum > 0 ? (n * 1000) / sum : 0,
    missedShare: n > 0 ? missed / n : 0,
  };
}

/** Frame-time target for a display rate: 72 Hz -> 13.9 ms, 90 Hz -> 11.1 ms. */
export function targetForHz(hz: number): number {
  return Math.round((1000 / hz) * 10) / 10;
}

/** Display rate from the median of steady frame intervals, snapped to common rates. */
export function estimateHz(medianDtMs: number): number | null {
  if (!(medianDtMs > 0)) return null;
  const hz = 1000 / medianDtMs;
  let best: number | null = null;
  for (const r of [60, 72, 80, 90, 120, 144]) {
    if (Math.abs(hz - r) / r < 0.04 && (best === null || Math.abs(hz - r) < Math.abs(hz - best))) {
      best = r;
    }
  }
  return best;
}

/**
 * Display-rate estimate from recent frame intervals: the last `size` intervals under
 * `maxDtMs` (stalls and loading pauses are ignored), re-estimated every 30 frames once
 * `min` are in. Uses p10: overload only lengthens intervals, so the short ones show the
 * display period. Returns null while it cannot tell (too few frames, no common rate).
 */
export class RateEstimator {
  private ring: number[] = [];
  private pushes = 0;
  private last: number | null = null;

  constructor(
    private readonly size = 120,
    private readonly min = 60,
    private readonly maxDtMs = 100,
  ) {}

  /** Feed one interval; returns a new estimate (Hz) when one is due, else undefined. */
  push(dtMs: number): number | null | undefined {
    if (!(dtMs > 0) || dtMs >= this.maxDtMs) return undefined;
    this.ring.push(dtMs);
    if (this.ring.length > this.size) this.ring.shift();
    this.pushes++;
    if (this.ring.length < this.min || this.pushes % 30 !== 0) return undefined;
    this.last = estimateHz(percentile(this.ring, 10));
    return this.last;
  }

  clear(): void {
    this.ring = [];
    this.pushes = 0;
    this.last = null;
  }
}

/** Keeps the frame intervals of the last `spanMs` milliseconds (the HUD's ~2 s window). */
export class RollingWindow {
  private readonly values: number[] = [];
  private sum = 0;

  constructor(readonly spanMs = 2000) {}

  push(dtMs: number): void {
    this.values.push(dtMs);
    this.sum += dtMs;
    while (this.values.length > 1 && this.sum - this.values[0] >= this.spanMs) {
      this.sum -= this.values.shift()!;
    }
  }

  clear(): void {
    this.values.length = 0;
    this.sum = 0;
  }

  get length(): number {
    return this.values.length;
  }

  summary(targetMs: number): FrameSummary {
    return summarize(this.values, targetMs);
  }
}
