import { describe, expect, it } from "vitest";
import {
  DETAIL_MIN,
  SLIDER_STEPS,
  countForSlider,
  formatSplats,
  sliderForCount,
} from "./detail";

describe("detail slider", () => {
  it("runs geometrically from the minimum to every splat in the scene", () => {
    expect(countForSlider(0, 1_000_000)).toBe(DETAIL_MIN);
    expect(countForSlider(SLIDER_STEPS, 1_000_000)).toBe(1_000_000);
    const mid = countForSlider(SLIDER_STEPS / 2, 1_000_000);
    expect(mid).toBe(Math.round(Math.sqrt(DETAIL_MIN * 1_000_000)));
  });

  it("inverts and clamps", () => {
    for (const n of [DETAIL_MIN, 120_000, 500_000, 1_000_000]) {
      expect(
        countForSlider(sliderForCount(n, 1_000_000), 1_000_000) / n,
      ).toBeCloseTo(1, 2);
    }
    expect(sliderForCount(10, 1_000_000)).toBe(0);
    expect(sliderForCount(6_000_000, 1_000_000)).toBe(SLIDER_STEPS);
  });

  it("handles scenes smaller than the minimum", () => {
    expect(countForSlider(0, 20_000)).toBe(20_000);
    expect(sliderForCount(20_000, 20_000)).toBe(SLIDER_STEPS);
  });

  it("formats counts", () => {
    expect(formatSplats(1_249_501)).toBe("1.25M");
    expect(formatSplats(740_400)).toBe("740k");
    expect(formatSplats(900)).toBe("900");
  });
});
