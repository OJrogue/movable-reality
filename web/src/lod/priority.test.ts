import { describe, expect, it } from "vitest";
import type { SceneManifest, SceneObject } from "../scene";
import {
  DEFAULT_PRIORITY,
  focusReason,
  latestEditedObject,
  parsePriorityConfig,
  RAMP_EPSILON,
  rampScale,
  targetScales,
} from "./priority";

function obj(id: string, visible = true): SceneObject {
  return {
    id,
    label: id,
    ply: `objects/${id}.ply`,
    completed_ply: null,
    bbox_min: [0, 0, 0],
    bbox_max: [1, 1, 1],
    transform: { translation: [0, 0, 0], yaw_deg: 0, scale: 1 },
    visible,
    source: "scan",
    prompt: null,
  };
}

function manifest(objects: SceneObject[], history: unknown[] = []): SceneManifest {
  return {
    version: 1,
    scene_id: "t",
    units: "m",
    world_from_recon: [],
    background_ply: "objects/background.ply",
    objects,
    history,
  };
}

describe("latestEditedObject", () => {
  it("returns the object of the newest history entry that names one", () => {
    const m = manifest([obj("chair#1"), obj("table#1")], [
      { op: "move", object_id: "chair#1" },
      { op: "rotate", object_id: "table#1" },
      { op: "note", object_id: null },
    ]);
    expect(latestEditedObject(m)).toBe("table#1");
  });

  it("gives no priority to a removed object or an unknown id", () => {
    const removed = manifest([obj("chair#1", false)], [{ op: "remove", object_id: "chair#1" }]);
    expect(latestEditedObject(removed)).toBeNull();
    const unknown = manifest([obj("chair#1")], [{ op: "move", object_id: "sofa#9" }]);
    expect(latestEditedObject(unknown)).toBeNull();
    expect(latestEditedObject(manifest([obj("chair#1")]))).toBeNull();
  });
});

describe("targetScales", () => {
  const m = manifest([obj("a"), obj("b"), obj("c")]);
  const cfg = { ...DEFAULT_PRIORITY, priorityScale: 2.5 };

  it("raises hovered and edited objects and leaves the rest at the base scale", () => {
    const t = targetScales(m, { hovered: "a", edited: "c" }, cfg);
    expect(Object.fromEntries(t)).toEqual({ a: 2.5, b: 1, c: 2.5 });
  });

  it("returns every object to the base scale when nothing is focused", () => {
    const t = targetScales(m, { hovered: null, edited: null }, cfg);
    expect([...t.values()]).toEqual([1, 1, 1]);
  });

  it("labels why an object is prioritised", () => {
    expect(focusReason("a", { hovered: "a", edited: "a" })).toBe("a (hover, edit)");
    expect(focusReason("b", { hovered: "a", edited: null })).toBe("b");
  });
});

describe("rampScale", () => {
  it("covers about 95% of the change after rampMs and never overshoots", () => {
    let s = 1;
    const seen: number[] = [];
    for (let t = 0; t < 300; t += 10) {
      s = rampScale(s, 2, 10, 300);
      seen.push(s);
    }
    expect(s).toBeGreaterThan(1.94);
    expect(s).toBeLessThanOrEqual(2);
    expect(seen.every((v, i) => i === 0 || v >= seen[i - 1])).toBe(true);
  });

  it("is frame-rate independent", () => {
    const twoSteps = rampScale(rampScale(1, 2, 8, 300), 2, 8, 300);
    expect(twoSteps).toBeCloseTo(rampScale(1, 2, 16, 300), 10);
  });

  it("ramps down as well as up and snaps to the target near the end", () => {
    expect(rampScale(2, 1, 16, 300)).toBeLessThan(2);
    expect(rampScale(1 + RAMP_EPSILON / 2, 1, 16, 300)).toBe(1);
    let s = 2;
    let frames = 0;
    while (s !== 1 && frames < 1000) {
      s = rampScale(s, 1, 16, 300);
      frames++;
    }
    expect(s).toBe(1);
    expect(frames).toBeLessThan(60); // under a second at 72 Hz
  });

  it("jumps when the ramp is off and ignores negative time", () => {
    expect(rampScale(1, 2, 16, 0)).toBe(2);
    expect(rampScale(1, 2, -5, 300)).toBe(1);
  });
});

describe("parsePriorityConfig", () => {
  it("reads lodPriority and lodRamp from the URL", () => {
    const cfg = parsePriorityConfig(new URLSearchParams("lodPriority=3&lodRamp=0"));
    expect(cfg).toEqual({ priorityScale: 3, baseScale: 1, rampMs: 0 });
  });

  it("falls back to the defaults for missing or invalid values", () => {
    expect(parsePriorityConfig(new URLSearchParams(""))).toEqual(DEFAULT_PRIORITY);
    const bad = parsePriorityConfig(new URLSearchParams("lodPriority=abc&lodRamp=-1"));
    expect(bad).toEqual(DEFAULT_PRIORITY);
    expect(parsePriorityConfig(new URLSearchParams("lodPriority=1000")).priorityScale).toBe(2);
  });
});
