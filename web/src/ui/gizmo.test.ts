import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { gizmoEdit, yawDeg } from "./gizmo";

const yaw = (deg: number) => new THREE.Quaternion().setFromEuler(new THREE.Euler(0, THREE.MathUtils.degToRad(deg), 0));
const pose = (p: [number, number, number], deg = 0) => ({ position: new THREE.Vector3(...p), quaternion: yaw(deg) });

describe("gizmo", () => {
  it("reads the yaw the viewer sets on a pivot (rotation.y)", () => {
    expect(yawDeg(yaw(30))).toBeCloseTo(30);
    expect(yawDeg(yaw(-120))).toBeCloseTo(-120);
  });

  it("turns a translate drag into a move edit in metres", () => {
    expect(gizmoEdit("chair#1", "translate", pose([1, 0.5, 2]), pose([1.25, 0.5, 1.9]))).toEqual({
      op: "move",
      object_id: "chair#1",
      args: { dx: 0.25, dy: 0, dz: -0.1 },
    });
  });

  it("turns a rotate drag into the shortest yaw delta", () => {
    const e = gizmoEdit("chair#1", "rotate", pose([0, 0, 0], 170), pose([0, 0, 0], -170));
    expect(e?.op).toBe("rotate");
    expect(e?.args?.yaw_deg).toBeCloseTo(20);
  });

  it("sends nothing for a press that did not move", () => {
    expect(gizmoEdit("a", "translate", pose([1, 0, 1]), pose([1, 0, 1.0001]))).toBeNull();
    expect(gizmoEdit("a", "rotate", pose([0, 0, 0], 10), pose([0, 0, 0], 10.01))).toBeNull();
  });
});
