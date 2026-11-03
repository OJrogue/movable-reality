import { describe, expect, it } from "vitest";
import { OBJECT_STYLES } from "../scene";
import { DEBUG_KEY, debugInitial, STYLE_NAMES } from "./chrome";
import type { KeyValue } from "./onboarding";

const store = (v: string | null): KeyValue => ({ getItem: () => v, setItem: () => undefined });
const q = (s: string) => new URLSearchParams(s);

describe("debugInitial", () => {
  it("is off for a first visit", () => {
    expect(debugInitial(q("scene=a"), store(null))).toBe(false);
    expect(debugInitial(q("scene=a"), null)).toBe(false);
  });
  it("remembers the last choice", () => {
    expect(debugInitial(q(""), store("1"))).toBe(true);
    expect(debugInitial(q(""), store("0"))).toBe(false);
  });
  it("turns on for bench runs and an explicit HUD", () => {
    expect(debugInitial(q("bench=1"), store("0"))).toBe(true);
    expect(debugInitial(q("hud=1"), store(null))).toBe(true);
  });
  it("lets ?debug win", () => {
    expect(debugInitial(q("debug=0&bench=1"), store("1"))).toBe(false);
    expect(debugInitial(q("debug=1"), store("0"))).toBe(true);
  });
  it("survives storage that throws", () => {
    const bad: KeyValue = { getItem: () => { throw new Error("blocked"); }, setItem: () => undefined }; // prettier-ignore
    expect(debugInitial(q(""), bad)).toBe(false);
    expect(DEBUG_KEY.startsWith("mr.")).toBe(true);
  });
});

describe("object style names", () => {
  it("names every style", () => {
    for (const s of OBJECT_STYLES) expect(STYLE_NAMES[s]).toBeTruthy();
  });
});
