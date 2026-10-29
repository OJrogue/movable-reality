import { describe, expect, it } from "vitest";
import { estimateHz, percentile, RollingWindow, summarize, targetForHz } from "./stats";

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
    const dts = [...new Array(97).fill(13.9), 16.6, 27.8, 27.8];
    const s = summarize(dts, 13.9);
    expect(s.frames).toBe(100);
    expect(s.p50).toBeCloseTo(13.9, 10);
    expect(s.p99).toBeCloseTo(27.8, 10);
    expect(s.missedShare).toBeCloseTo(0.02, 10);
    expect(s.fps).toBeCloseTo((100 * 1000) / (97 * 13.9 + 16.6 + 2 * 27.8), 6);
  });
});

describe("refresh rate", () => {
  it("maps rates to targets and snaps intervals to common rates", () => {
    expect(targetForHz(72)).toBe(13.9);
    expect(targetForHz(90)).toBe(11.1);
    expect(estimateHz(13.89)).toBe(72);
    expect(estimateHz(11.2)).toBe(90);
    expect(estimateHz(16.7)).toBe(60);
    expect(estimateHz(25)).toBeNull();
    expect(estimateHz(0)).toBeNull();
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
