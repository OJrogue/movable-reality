import { describe, expect, it } from "vitest";
import { percentile, RollingWindow, summarize } from "./stats";

describe("percentile", () => {
  it("matches numpy's linear interpolation", () => {
    // numpy.percentile([1, 2, 3, 4, 10], [0, 50, 90, 99, 100])
    const v = [10, 1, 4, 2, 3];
    expect(percentile(v, 0)).toBe(1);
    expect(percentile(v, 50)).toBe(3);
    expect(percentile(v, 90)).toBeCloseTo(7.6, 10);
    expect(percentile(v, 99)).toBeCloseTo(9.76, 10);
    expect(percentile(v, 100)).toBe(10);
    expect(percentile([], 50)).toBeNaN();
  });
});

describe("summarize", () => {
  it("reports fps over the summed intervals and the missed-frame share", () => {
    const dts = [...new Array(98).fill(13.9), 27.8, 27.8];
    const s = summarize(dts, 13.9);
    expect(s.frames).toBe(100);
    expect(s.p50).toBeCloseTo(13.9, 10);
    expect(s.p99).toBeCloseTo(27.8, 10);
    expect(s.missedShare).toBeCloseTo(0.02, 10);
    expect(s.fps).toBeCloseTo((100 * 1000) / (98 * 13.9 + 2 * 27.8), 6);
  });
});

describe("RollingWindow", () => {
  it("keeps about spanMs of frames", () => {
    const w = new RollingWindow(1000);
    for (let i = 0; i < 500; i++) w.push(10);
    expect(w.length).toBe(100);
    w.push(500);
    expect(w.length).toBe(51);
    w.clear();
    expect(w.length).toBe(0);
  });
});
