import { describe, expect, it } from "vitest";
import { DISMISS_KEY, SAMPLE_COMMANDS, sampleCommands, shouldShowOnboarding, type KeyValue } from "./onboarding";

const store = (v: string | null): KeyValue => ({ getItem: () => v, setItem: () => undefined });
const throwing: KeyValue = {
  getItem: () => {
    throw new Error("blocked");
  },
  setItem: () => undefined,
};
const q = (s: string) => new URLSearchParams(s);

describe("shouldShowOnboarding", () => {
  it("shows the panel on a first visit", () => {
    expect(shouldShowOnboarding(q(""), store(null))).toBe(true);
    expect(shouldShowOnboarding(q(""), null)).toBe(true);
  });
  it("respects a remembered dismissal", () => {
    expect(shouldShowOnboarding(q(""), store("1"))).toBe(false);
  });
  it("lets ?onboarding override storage", () => {
    expect(shouldShowOnboarding(q("onboarding=0"), store(null))).toBe(false);
    expect(shouldShowOnboarding(q("onboarding=1"), store("1"))).toBe(true);
  });
  it("shows the panel when storage throws", () => {
    expect(shouldShowOnboarding(q(""), throwing)).toBe(true);
  });
  it("uses a namespaced key", () => {
    expect(DISMISS_KEY.startsWith("mr.")).toBe(true);
  });
});

describe("sampleCommands", () => {
  it("names objects of the scene, skipping surfaces and clutter", () => {
    const out = sampleCommands(["wall", "Sofa", "sofa", "book", "armchair", "coffee table", "lamp"]);
    expect(out[0]).toBe("Move the sofa half a metre to the left");
    expect(out.join(" ")).not.toMatch(/wall|book/);
    expect(out).toContain("Remove the coffee table");
  });
  it("falls back to the generic samples", () => {
    expect(sampleCommands(["wall", "chair"])).toEqual(SAMPLE_COMMANDS);
  });
  it("never pairs an object with itself", () => {
    const out = sampleCommands(["bed", "desk", "chair"]);
    expect(out.join(" ")).not.toMatch(/next to/);
  });
  it("never repeats a command with few labels", () => {
    const out = sampleCommands(["bed", "desk"]);
    expect(new Set(out).size).toBe(out.length);
  });
});
