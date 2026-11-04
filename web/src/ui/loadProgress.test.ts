import { describe, expect, it } from "vitest";
import { LoadProgress, loadingText, sizeFromContentRange } from "./loadProgress";

describe("LoadProgress", () => {
  it("weighs files by size", () => {
    const p = new LoadProgress();
    p.add("big");
    p.add("small");
    p.setSize("big", 900);
    p.setSize("small", 100);
    p.update("small", 100, 0);
    expect(p.snapshot().fraction).toBeCloseTo(0.095);
    p.update("big", 450, 0);
    expect(p.snapshot().fraction).toBeCloseTo(0.5225);
    expect(p.snapshot().loadedBytes).toBe(550);
    expect(p.snapshot().totalBytes).toBe(1000);
  });

  it("counts files that have not started, so the bar does not jump when they do", () => {
    const p = new LoadProgress();
    for (const f of ["a", "b", "c", "d"]) {
      p.add(f);
      p.setSize(f, 100);
    }
    p.update("a", 100, 100);
    expect(p.snapshot().fraction).toBeCloseTo(0.95 / 4);
  });

  it("never falls when a size arrives late", () => {
    const p = new LoadProgress();
    p.add("a");
    p.add("b");
    p.setSize("a", 100);
    p.update("a", 100, 0); // b counts as average: half done
    const before = p.snapshot().fraction;
    p.setSize("b", 10_000); // really about 1% done
    expect(p.snapshot().fraction).toBe(before);
  });

  it("takes a size from the response when the probe failed", () => {
    const p = new LoadProgress();
    p.add("a");
    expect(p.snapshot().totalBytes).toBeNull();
    p.update("a", 10, 40);
    expect(p.snapshot().totalBytes).toBe(40);
  });

  it("counts a failed file as settled and stays under 100% until complete()", () => {
    const p = new LoadProgress();
    p.add("a");
    p.add("b");
    p.setSize("a", 100);
    p.setSize("b", 100);
    expect(p.allSettled()).toBe(false);
    p.settle("a");
    p.settle("b");
    expect(p.allSettled()).toBe(true);
    expect(p.snapshot().fraction).toBeCloseTo(0.95);
    p.complete();
    expect(p.snapshot().fraction).toBe(1);
  });

  it("is not settled before any file is known", () => {
    expect(new LoadProgress().allSettled()).toBe(false);
  });
});

describe("loadingText", () => {
  const base = { fraction: 0.42, loadedBytes: 36e6, totalBytes: 85e6, filesDone: 3, files: 34 };

  it("shows megabytes once all sizes are known", () => {
    expect(loadingText(base)).toBe("42% · 36 of 85 MB");
  });

  it("falls back to files without sizes", () => {
    expect(loadingText({ ...base, totalBytes: null })).toBe("42% · 3 of 34 files");
  });

  it("names the steps before and after the download", () => {
    expect(loadingText({ ...base, files: 0, filesDone: 0 })).toBe("Loading the room");
    expect(loadingText({ ...base, filesDone: 34 })).toBe("Setting up the room");
  });
});

describe("sizeFromContentRange", () => {
  it("reads the total", () => {
    expect(sizeFromContentRange("bytes 0-0/8751088")).toBe(8751088);
  });

  it("rejects a missing or unknown total", () => {
    expect(sizeFromContentRange(null)).toBeNull();
    expect(sizeFromContentRange("bytes 0-0/*")).toBeNull();
  });
});
