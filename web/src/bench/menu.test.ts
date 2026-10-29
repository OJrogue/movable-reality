import { describe, expect, it } from "vitest";
import { r0Runs } from "./menu";

describe("R0 run menu", () => {
  it("lists 5 scenes x naive/lod with bench URLs", () => {
    const runs = r0Runs(new URLSearchParams("duration=60&device=Steam Frame"));
    expect(runs).toHaveLength(10);
    expect(runs.map((r) => `${r.scene}:${r.mode}`).slice(0, 3)).toEqual([
      "bench_50000:naive",
      "bench_50000:lod",
      "bench_150000:naive",
    ]);
    const q = new URLSearchParams(runs[1].url.slice(2));
    expect(Object.fromEntries(q)).toEqual({
      bench: "1",
      scene: "bench_50000",
      mode: "lod",
      duration: "60",
      path: "walk",
      hz: "72",
      device: "Steam Frame",
    });
  });

  it("defaults to 2 min runs on the walk path", () => {
    const q = new URLSearchParams(r0Runs(new URLSearchParams())[0].url.slice(2));
    expect(q.get("duration")).toBe("120");
    expect(q.get("path")).toBe("walk");
    expect(q.has("device")).toBe(false);
  });
});
