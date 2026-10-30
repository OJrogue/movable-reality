import { describe, expect, it } from "vitest";
import { DISMISS_KEY, shouldShowOnboarding, type KeyValue } from "./onboarding";

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
