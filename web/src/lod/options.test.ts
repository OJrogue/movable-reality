import { describe, expect, it } from "vitest";
import type { SceneManifest } from "../scene";
import { lodChoice, radPath } from "./options";

const base: SceneManifest = {
  version: 1,
  scene_id: "t",
  units: "m",
  world_from_recon: [],
  background_ply: "objects/background.ply",
  objects: [
    {
      id: "chair#1",
      label: "chair",
      ply: "objects/chair_1.ply",
      completed_ply: null,
      lod: "lod/chair_1.rad",
      bbox_min: [0, 0, 0],
      bbox_max: [1, 1, 1],
      transform: { translation: [0, 0, 0], yaw_deg: 0, scale: 1 },
      visible: true,
      source: "scan",
      prompt: null,
    },
    {
      id: "table#1",
      label: "table",
      ply: "objects/table_1.ply",
      completed_ply: null,
      bbox_min: [0, 0, 0],
      bbox_max: [1, 1, 1],
      transform: { translation: [0, 0, 0], yaw_deg: 0, scale: 1 },
      visible: true,
      source: "scan",
      prompt: null,
    },
  ],
  history: [],
};

describe("lodChoice", () => {
  it("loads the .RAD tree when the manifest names one, with no lod flag", () => {
    const c = lodChoice("objects/chair_1.ply", "chair#1", base, "/scenes/demo", true, 2);
    expect(c.source).toBe("rad");
    expect(c.options).toEqual({ url: "/scenes/demo/lod/chair_1.rad", lodScale: 2 });
  });

  it("builds the tree at load time when there is no .RAD (missing or older manifest)", () => {
    const c = lodChoice("objects/table_1.ply", "table#1", base, "/scenes/demo", "quality", 1);
    expect(c.source).toBe("load-time");
    expect(c.path).toBe("objects/table_1.ply");
    expect(c.options).toEqual({ lod: "quality", lodScale: 1 });
    expect(c.options.url).toBeUndefined(); // SceneView fills in the PLY url
  });

  it("streams the .RAD tree only when paging is asked for, never a PLY", () => {
    const rad = lodChoice("objects/chair_1.ply", "chair#1", base, "/s", true, 1, true);
    expect(rad.options.paged).toBe(true);
    const ply = lodChoice("objects/table_1.ply", "table#1", base, "/s", true, 1, true);
    expect(ply.options.paged).toBeUndefined();
  });

  it("uses background_lod for the background", () => {
    expect(radPath(base, null)).toBeNull();
    const withBg = { ...base, background_lod: "lod/background.rad" };
    expect(radPath(withBg, null)).toBe("lod/background.rad");
    expect(lodChoice("objects/background.ply", null, withBg, "/s", true, 1).source).toBe("rad");
  });
});

describe("completed objects", () => {
  const chair = base.objects[0];
  const withCompleted: SceneManifest = {
    ...base,
    objects: [
      {
        ...chair,
        completed_ply: "objects/chair_1_completed.ply",
        completed_lod: "lod/chair_1_completed.rad",
      },
      base.objects[1],
    ],
  };

  it("loads completed_lod for the completed PLY and lod for the scan", () => {
    const done = lodChoice("objects/chair_1_completed.ply", "chair#1", withCompleted, "/s", true, 1);
    expect(done.options.url).toBe("/s/lod/chair_1_completed.rad");
    const scan = lodChoice("objects/chair_1.ply", "chair#1", withCompleted, "/s", true, 1);
    expect(scan.options.url).toBe("/s/lod/chair_1.rad");
  });

  it("builds the completed tree at load time when completed_lod is missing", () => {
    const m: SceneManifest = {
      ...withCompleted,
      objects: [{ ...withCompleted.objects[0], completed_lod: null }, base.objects[1]],
    };
    const c = lodChoice("objects/chair_1_completed.ply", "chair#1", m, "/s", true, 1);
    expect(c.source).toBe("load-time");
    expect(c.path).toBe("objects/chair_1_completed.ply");
  });
});
