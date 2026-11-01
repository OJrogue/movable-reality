import { describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  isEdited,
  matrixFromRows,
  OPENCV_TO_Y_UP,
  shownPly,
  type SceneObject,
  type Vec3,
} from "./scene";

describe("matrixFromRows", () => {
  it("reads row-major rows, so translation lands in the last column", () => {
    const m = matrixFromRows([
      [1, 0, 0, 2],
      [0, 1, 0, 3],
      [0, 0, 1, 4],
      [0, 0, 0, 1],
    ]);
    const p = new THREE.Vector3(0, 0, 0).applyMatrix4(m);
    expect(p.toArray()).toEqual([2, 3, 4]);
  });

  it("maps OpenCV +Y down to world +Y up", () => {
    const p = new THREE.Vector3(0, 1, 1).applyMatrix4(matrixFromRows(OPENCV_TO_Y_UP));
    expect(p.toArray()).toEqual([0, -1, -1]);
  });
});

describe("shownPly", () => {
  const obj: SceneObject = {
    id: "chair#1",
    label: "chair",
    ply: "objects/chair_1.ply",
    completed_ply: "objects/chair_1_completed.ply",
    bbox_min: [0, 0, 0],
    bbox_max: [1, 1, 1],
    transform: { translation: [0, 0, 0], yaw_deg: 0, scale: 1 },
    visible: true,
    source: "scan",
    prompt: null,
  };
  const t = (translation: Vec3, yaw_deg = 0, scale = 1) => ({ translation, yaw_deg, scale });
  const done = "objects/chair_1_completed.ply";

  it("shows the scan until the object is edited", () => {
    expect(shownPly(obj)).toBe("objects/chair_1.ply");
    expect(shownPly({ ...obj, transform: t([0, 0, 0], 360) })).toBe("objects/chair_1.ply");
  });

  it("swaps to the completed PLY after a move, rotation or scale", () => {
    expect(shownPly({ ...obj, transform: t([0.5, 0, 0]) })).toBe(done);
    expect(shownPly({ ...obj, transform: t([0, 0, 0], 90) })).toBe(done);
    expect(shownPly({ ...obj, transform: t([0, 0, 0], 0, 1.2) })).toBe(done);
  });

  it("shows the completed PLY unedited in the completed style", () => {
    expect(shownPly(obj, "completed")).toBe(done);
    expect(shownPly({ ...obj, completed_ply: null }, "completed")).toBe("objects/chair_1.ply");
    expect(shownPly(obj, "mesh")).toBe("objects/chair_1.ply"); // splat fallback of the mesh style
    const moved = { ...obj, transform: t([0.5, 0, 0]) };
    expect(shownPly(moved, "scan")).toBe("objects/chair_1.ply"); // scan pins the scan
    expect(shownPly(moved, "mesh")).toBe(done);
  });

  it("keeps the scan when there is no completed version", () => {
    const o = { ...obj, completed_ply: null, transform: t([1, 0, 0]) };
    expect(shownPly(o)).toBe("objects/chair_1.ply");
  });

  it("treats float noise as no edit", () => {
    expect(isEdited(t([1e-9, 0, 0], 1e-9, 1 + 1e-9))).toBe(false);
    expect(isEdited(t([0, 0, 0], 360 - 1e-9))).toBe(false);
    expect(isEdited(t([0, 0, 0], -1e-9))).toBe(false);
  });
});
