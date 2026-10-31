import { describe, expect, it } from "vitest";
import { DEFAULT_TOUCH, isTap, look, TAP_SLOP_PX, twoFinger } from "./gestures";

const k = DEFAULT_TOUCH.lookRadPerPx;

describe("look (one finger)", () => {
  it("dragging right turns the view left (the scene follows the finger)", () => {
    const r = look(0, 0, 100, 0);
    expect(r.yaw).toBeCloseTo(100 * k); // positive yaw about +Y turns left
    expect(r.pitch).toBe(0);
  });
  it("dragging down tilts the view up", () => {
    expect(look(0, 0, 0, 40).pitch).toBeCloseTo(40 * k);
  });
  it("pitch stops short of straight up or down", () => {
    expect(look(0, 0, 0, 1e6).pitch).toBe(DEFAULT_TOUCH.maxPitch);
    expect(look(0, 0, 0, -1e6).pitch).toBe(-DEFAULT_TOUCH.maxPitch);
  });
});

describe("twoFinger", () => {
  const a = { x: 100, y: 300 };
  const b = { x: 200, y: 300 };

  it("both fingers moving right pan the camera left, with no pinch", () => {
    const s = twoFinger(a, b, { x: 150, y: 300 }, { x: 250, y: 300 });
    expect(s.right).toBeCloseTo(-50 * DEFAULT_TOUCH.panMPerPx);
    expect(s.up).toBeCloseTo(0);
    expect(s.forward).toBeCloseTo(0);
  });
  it("both fingers moving down pan the camera up", () => {
    const s = twoFinger(a, b, { x: 100, y: 340 }, { x: 200, y: 340 });
    expect(s.up).toBeCloseTo(40 * DEFAULT_TOUCH.panMPerPx);
  });
  it("spreading moves forward, pinching moves back by the same amount", () => {
    const out = twoFinger(a, b, { x: 50, y: 300 }, { x: 250, y: 300 });
    expect(out.forward).toBeCloseTo(Math.log(2) * DEFAULT_TOUCH.pinchM);
    expect(out.right).toBeCloseTo(0); // the midpoint did not move
    const back = twoFinger({ x: 50, y: 300 }, { x: 250, y: 300 }, a, b);
    expect(back.forward).toBeCloseTo(-out.forward);
  });
  it("one finger moving while the other rests pans by half and pinches", () => {
    const s = twoFinger(a, b, a, { x: 300, y: 300 });
    expect(s.right).toBeCloseTo(-50 * DEFAULT_TOUCH.panMPerPx);
    expect(s.forward).toBeGreaterThan(0);
  });
  it("coincident fingers give no pinch instead of infinity", () => {
    const s = twoFinger(a, a, a, b);
    expect(Number.isFinite(s.forward)).toBe(true);
    expect(s.forward).toBe(0);
  });
});

describe("isTap", () => {
  it("allows finger wobble under the slop", () => {
    expect(isTap({ x: 0, y: 0 }, { x: 6, y: 6 })).toBe(true);
    expect(isTap({ x: 0, y: 0 }, { x: TAP_SLOP_PX, y: 0 })).toBe(false);
    expect(isTap({ x: 0, y: 0 }, { x: 3, y: 0 }, 4)).toBe(true);
  });
});
