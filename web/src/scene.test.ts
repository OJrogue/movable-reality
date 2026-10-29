import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { matrixFromRows, OPENCV_TO_Y_UP } from "./scene";

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
