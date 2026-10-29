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

/** A frame counts as missed when its interval exceeds this multiple of the budget. */
export const MISS_FACTOR = 1.5;

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
