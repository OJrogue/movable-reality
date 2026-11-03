import { describe, expect, it } from "vitest";
import { byCategory, dropPoint, type LibraryItem, yawFacing } from "./index";
import { addLibraryObject, applyEdit } from "../ui/edits";
import { roomRows } from "../ui/libraryPanel";
import type { SceneManifest, SceneObject } from "../scene";

const object = (id: string, over: Partial<SceneObject> = {}): SceneObject => ({
  id,
  label: id.split("#")[0],
  ply: `objects/${id}.ply`,
  completed_ply: null,
  bbox_min: [1, 0, 1],
  bbox_max: [2, 0.7, 2],
  transform: { translation: [0, 0, 0], yaw_deg: 0, scale: 1 },
  visible: true,
  source: "scan",
  prompt: null,
  ...over,
});

const manifest = (objects: SceneObject[]): SceneManifest => ({
  version: 1,
  scene_id: "s",
  units: "m",
  world_from_recon: [],
  background_ply: "b.ply",
  objects,
  history: [],
});

describe("dropPoint", () => {
  it("takes where the view hits the floor when that is near", () => {
    const p = dropPoint([0, 1.6, 0], [0, -0.8, -0.6]);
    expect(p.x).toBeCloseTo(0);
    expect(p.z).toBeCloseTo(-1.2);
  });
  it("falls back to 2 m ahead when looking up or at the horizon", () => {
    for (const fy of [0.3, -0.01]) {
      const p = dropPoint([1, 1.6, 1], [1, fy, 0]);
      expect(p.x).toBeCloseTo(3);
      expect(p.z).toBeCloseTo(1);
    }
  });
  it("copes with looking straight up", () => {
    expect(dropPoint([0, 1.6, 0], [0, 1, 0]).z).toBeCloseTo(-2);
  });
});

describe("yawFacing", () => {
  it("turns the +Z front towards the user", () => {
    expect(yawFacing(0, 0, [0, 1.6, 2])).toBe(0);
    expect(yawFacing(0, 0, [2, 1.6, 0])).toBe(90);
    expect(yawFacing(0, 0, [0, 1.6, -2])).toBe(180);
    expect(yawFacing(0, 0, [-2, 1.6, 0])).toBe(270);
  });
  it("stays square to the walls", () => {
    expect(yawFacing(0, 0, [2, 1.6, 1.5])).toBe(90);
    expect(yawFacing(0, 0, [1.5, 1.6, 2])).toBe(0);
  });
});

describe("byCategory", () => {
  it("keeps catalogue order", () => {
    const item = (id: string, category: string) => ({ id, category }) as LibraryItem;
    const groups = byCategory([item("a", "X"), item("b", "Y"), item("c", "X")]);
    expect(groups.map(([c, items]) => [c, items.map((i) => i.id)])).toEqual([
      ["X", ["a", "c"]],
      ["Y", ["b"]],
    ]);
  });
});

describe("addLibraryObject", () => {
  const sofa = { id: "sofa", label: "sofa", size: [2, 0.8, 1] as [number, number, number] };

  it("matches Scene.add_library_object and leaves the input alone", () => {
    const m = manifest([object("table#1")]);
    const out = addLibraryObject(m, sofa, { x: 0, z: -1, yaw_deg: -90 });
    const added = out.objects[1];
    expect(added).toMatchObject({ id: "lib#1", source: "library", library_id: "sofa", ply: "" });
    expect(added.bbox_min).toEqual([-1, 0, -1.5]);
    expect(added.bbox_max).toEqual([1, 0.8, -0.5]);
    expect(added.transform.yaw_deg).toBe(270);
    expect(m.objects).toHaveLength(1);
    expect(addLibraryObject(out, sofa, { x: 0, z: 0, yaw_deg: 0 }).objects[2].id).toBe("lib#2");
  });

  it("stands on the edited top of another object", () => {
    const m = manifest([object("table#1", { transform: { translation: [0, 0.1, 0], yaw_deg: 0, scale: 1 } })]);
    const out = addLibraryObject(m, sofa, { x: 0, z: 0, yaw_deg: 0, on_object_id: "table#1" });
    expect(out.objects[1].bbox_min[1]).toBeCloseTo(0.8);
  });

  it("stays over the object it stands on, as library.add_to_scene does", () => {
    const m = manifest([object("table#1")]);
    const vase = { id: "vase", label: "vase", size: [0.2, 0.4, 0.2] as [number, number, number] };
    const near = addLibraryObject(m, vase, { x: 1.5, z: 0.23, yaw_deg: 0, on_object_id: "table#1" }).objects[1];
    expect(near.bbox_min[0]).toBeCloseTo(1.4);
    expect(near.bbox_min[2]).toBeCloseTo(1);
    const long = { id: "runner", label: "runner", size: [0.6, 0.1, 0.2] as [number, number, number] };
    const turned = addLibraryObject(m, long, { x: 0, z: 1, yaw_deg: 90, on_object_id: "table#1" }).objects[1];
    expect((turned.bbox_min[0] + turned.bbox_max[0]) / 2).toBeCloseTo(1.1);
    expect((turned.bbox_min[2] + turned.bbox_max[2]) / 2).toBeCloseTo(1.3);
    const wide = addLibraryObject(m, sofa, { x: 9, z: 9, yaw_deg: 0, on_object_id: "table#1" }).objects[1];
    expect((wide.bbox_min[0] + wide.bbox_max[0]) / 2).toBeCloseTo(1.5);
  });

  it("undo hides it", () => {
    const out = addLibraryObject(manifest([]), sofa, { x: 0, z: 0, yaw_deg: 0 });
    expect(applyEdit(out, { op: "undo", object_id: null }).objects[0].visible).toBe(false);
  });
});

describe("roomRows", () => {
  it("lists visible and removed objects, never pipeline-hidden copies or undone adds", () => {
    let m = manifest([object("chair#1"), object("copy#2", { visible: false }), object("table#3")]);
    m = applyEdit(m, { op: "remove", object_id: "chair#1" });
    m = addLibraryObject(m, { id: "sofa", label: "sofa", size: [1, 1, 1] }, { x: 0, z: 0, yaw_deg: 0 });
    m = addLibraryObject(m, { id: "lamp", label: "lamp", size: [1, 1, 1] }, { x: 0, z: 0, yaw_deg: 0 });
    m = applyEdit(m, { op: "undo", object_id: null });
    expect(roomRows(m).map((r) => [r.object.id, r.removed])).toEqual([
      ["lib#1", false],
      ["table#3", false],
      ["chair#1", true],
    ]);
  });
});
