import { describe, expect, it } from "vitest";
import type { SceneManifest, SceneObject } from "../scene";
import { applyEdit, EditError, isoSeconds, lastHidden, type HistoryEntry } from "./edits";

function obj(id: string): SceneObject {
  return {
    id,
    label: id.split("#")[0],
    ply: `objects/${id}.ply`,
    completed_ply: null,
    bbox_min: [0, 0, 0],
    bbox_max: [2, 1, 4],
    transform: { translation: [0, 0, 0], yaw_deg: 0, scale: 1 },
    visible: true,
    source: "scan",
    prompt: null,
  };
}

function manifest(): SceneManifest {
  return {
    version: 1,
    scene_id: "t",
    units: "m",
    world_from_recon: [],
    background_ply: "bg.ply",
    objects: [obj("chair#1"), obj("lamp#1")],
    history: [],
  };
}

const NOW = new Date("2026-10-30T12:34:56.789Z");
const hist = (m: SceneManifest) => m.history as HistoryEntry[];

describe("applyEdit", () => {
  it("does not mutate the input", () => {
    const m = manifest();
    applyEdit(m, { op: "move", object_id: "chair#1", args: { dx: 1 } }, NOW);
    expect(m.objects[0].transform.translation).toEqual([0, 0, 0]);
    expect(m.history).toEqual([]);
  });

  it("moves and logs like scene.py", () => {
    const m = applyEdit(manifest(), { op: "move", object_id: "chair#1", args: { dx: 1, dz: -2 } }, NOW);
    expect(m.objects[0].transform.translation).toEqual([1, 0, -2]);
    expect(hist(m)[0]).toEqual({
      ts: "2026-10-30T12:34:56+00:00",
      op: "move",
      object_id: "chair#1",
      args: { dx: 1, dy: 0, dz: -2 },
      instruction: null,
      verified: null,
    });
  });

  it("places the bbox centre and keeps the height", () => {
    let m = applyEdit(manifest(), { op: "move", object_id: "chair#1", args: { dy: 0.5 } }, NOW);
    m = applyEdit(m, { op: "place", object_id: "chair#1", args: { x: 3, z: 1 } }, NOW);
    // bbox centre is (1, 0.5, 2), so translation = (3 - 1, 0.5, 1 - 2)
    expect(m.objects[0].transform.translation).toEqual([2, 0.5, -1]);
    expect(hist(m)[1].args).toEqual({ x: 3, z: 1, prev: [0, 0.5, 0] });
  });

  it("wraps yaw into [0, 360) like Python's modulo", () => {
    const m = applyEdit(manifest(), { op: "rotate", object_id: "chair#1", args: { yaw_deg: -15 } }, NOW);
    expect(m.objects[0].transform.yaw_deg).toBe(345);
  });

  it("removes and restores", () => {
    let m = applyEdit(manifest(), { op: "remove", object_id: "lamp#1" }, NOW);
    expect(m.objects[1].visible).toBe(false);
    m = applyEdit(m, { op: "restore", object_id: "lamp#1" }, NOW);
    expect(m.objects[1].visible).toBe(true);
  });

  it("rejects unknown objects, missing arguments and missing ids", () => {
    expect(() => applyEdit(manifest(), { op: "remove", object_id: "sofa#9" })).toThrow(EditError);
    expect(() => applyEdit(manifest(), { op: "place", object_id: "chair#1", args: { x: 1 } })).toThrow(
      EditError,
    );
    expect(() => applyEdit(manifest(), { op: "rotate", object_id: null })).toThrow(EditError);
  });
});

describe("undo", () => {
  it("reverts the newest edit not yet undone and keeps history append-only", () => {
    let m = manifest();
    m = applyEdit(m, { op: "move", object_id: "chair#1", args: { dx: 1 } }, NOW);
    m = applyEdit(m, { op: "rotate", object_id: "lamp#1", args: { yaw_deg: 30 } }, NOW);
    m = applyEdit(m, { op: "place", object_id: "chair#1", args: { x: 5, z: 5 } }, NOW);
    m = applyEdit(m, { op: "remove", object_id: "lamp#1" }, NOW);

    m = applyEdit(m, { op: "undo", object_id: null }, NOW);
    expect(m.objects[1].visible).toBe(true);
    m = applyEdit(m, { op: "undo", object_id: null }, NOW);
    expect(m.objects[0].transform.translation).toEqual([1, 0, 0]);
    m = applyEdit(m, { op: "undo", object_id: null }, NOW);
    expect(m.objects[1].transform.yaw_deg).toBe(0);
    m = applyEdit(m, { op: "undo", object_id: null }, NOW);
    expect(m.objects[0].transform.translation).toEqual([0, 0, 0]);

    expect(hist(m).map((e) => e.op)).toEqual([
      "move", "rotate", "place", "remove", "undo", "undo", "undo", "undo",
    ]);
    expect(hist(m).slice(4).map((e) => e.args)).toEqual([
      { index: 3, of: "remove" },
      { index: 2, of: "place" },
      { index: 1, of: "rotate" },
      { index: 0, of: "move" },
    ]);
    // Nothing left: no change, no new entry.
    const again = applyEdit(m, { op: "undo", object_id: null }, NOW);
    expect(again.history.length).toBe(8);
  });

  it("hides an object again when undoing a restore", () => {
    let m = applyEdit(manifest(), { op: "remove", object_id: "chair#1" }, NOW);
    m = applyEdit(m, { op: "restore", object_id: "chair#1" }, NOW);
    m = applyEdit(m, { op: "undo", object_id: null }, NOW);
    expect(m.objects[0].visible).toBe(false);
  });
});

describe("lastHidden", () => {
  it("finds the newest removed object that is still hidden", () => {
    let m = manifest();
    expect(lastHidden(m)).toBeNull();
    m = applyEdit(m, { op: "remove", object_id: "chair#1" }, NOW);
    m = applyEdit(m, { op: "remove", object_id: "lamp#1" }, NOW);
    expect(lastHidden(m)).toBe("lamp#1");
    m = applyEdit(m, { op: "restore", object_id: "lamp#1" }, NOW);
    expect(lastHidden(m)).toBe("chair#1");
  });
});

describe("isoSeconds", () => {
  it("drops milliseconds and writes the UTC offset", () => {
    expect(isoSeconds(NOW)).toBe("2026-10-30T12:34:56+00:00");
  });
});
