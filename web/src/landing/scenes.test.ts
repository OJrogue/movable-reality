import { describe, expect, it } from "vitest";
import { availabilityOf, DEMO_SCENES, findDemoScene, sceneAvailability, viewerUrl } from "./scenes";

const json = (body: unknown, ok = true) =>
  ({ ok, headers: new Headers({ "content-type": "application/json" }), json: async () => body }) as Response;
const html = { ok: true, headers: new Headers({ "content-type": "text/html" }) } as Response;

describe("demo scenes", () => {
  it("has three scenes with unique ids, one of them real", () => {
    expect(DEMO_SCENES).toHaveLength(3);
    expect(new Set(DEMO_SCENES.map((s) => s.id)).size).toBe(3);
    expect(DEMO_SCENES.filter((s) => s.kind === "real")).toHaveLength(1);
    expect(findDemoScene("blender06")?.title).toBe("Bedroom");
    expect(findDemoScene(null)).toBeUndefined();
  });
  it("builds viewer links that keep a base override", () => {
    expect(viewerUrl("blender07")).toBe("/?scene=blender07");
    expect(viewerUrl("a", "https://x/s")).toBe("/?scene=a&base=https%3A%2F%2Fx%2Fs");
  });
});

describe("availability", () => {
  it("reads the dev server index", () => {
    expect(availabilityOf({ id: "a", ready: true })).toBe("ready");
    expect(availabilityOf({ id: "a", ready: true, pipeline: { stages: { lod: { state: "running" } } } })).toBe("updating");
    expect(availabilityOf({ id: "a", ready: false, pipeline: { stages: {} } })).toBe("processing");
    expect(availabilityOf({ id: "a", ready: false })).toBe("missing");
  });
  it("uses /__scenes when the dev server answers", async () => {
    const f = (async () => json([{ id: "a", ready: true }, { id: "b", ready: false, pipeline: { stages: {} } }])) as typeof fetch;
    const m = await sceneAvailability(["a", "b", "c"], "/scenes", f);
    expect([...m]).toEqual([["a", "ready"], ["b", "processing"], ["c", "missing"]]);
  });
  it("falls back to scene.json on a static host", async () => {
    const f = (async (url: string) => {
      if (url === "/__scenes") return html; // SPA fallback serves index.html
      return url.includes("/a/") ? json({}) : json({}, false);
    }) as typeof fetch;
    const m = await sceneAvailability(["a", "b"], "/s", f);
    expect(m.get("a")).toBe("ready");
    expect(m.get("b")).toBe("missing");
  });
});
