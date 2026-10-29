import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { halfSize, pickNearest, rayBoxDistance } from "./pick";

/** Pivot matrix as SceneView builds it: bbox centre + translation, yaw about +Y, scale. */
function pivot(center: [number, number, number], yawDeg = 0, scale = 1): THREE.Matrix4 {
  return new THREE.Matrix4().compose(
    new THREE.Vector3(...center),
    new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(yawDeg)),
    new THREE.Vector3(scale, scale, scale),
  );
}

const ray = (o: [number, number, number], d: [number, number, number]) =>
  new THREE.Ray(new THREE.Vector3(...o), new THREE.Vector3(...d).normalize());

describe("halfSize", () => {
  it("is half the bbox extent", () => {
    expect(halfSize({ bbox_min: [-1, 0, 2], bbox_max: [1, 0.5, 3] })).toEqual([1, 0.25, 0.5]);
  });
});

describe("rayBoxDistance", () => {
  const half: [number, number, number] = [0.5, 0.5, 0.5];

  it("returns the world distance to the box face", () => {
    expect(rayBoxDistance(ray([0, 0, 5], [0, 0, -1]), pivot([0, 0, 0]), half)).toBeCloseTo(4.5);
  });

  it("misses boxes beside or behind the ray", () => {
    expect(rayBoxDistance(ray([2, 0, 5], [0, 0, -1]), pivot([0, 0, 0]), half)).toBeNull();
    expect(rayBoxDistance(ray([0, 0, 5], [0, 0, 1]), pivot([0, 0, 0]), half)).toBeNull();
  });

  it("is 0 from inside the box", () => {
    expect(rayBoxDistance(ray([0, 0, 0], [1, 0, 0]), pivot([0, 0, 0]), half)).toBeCloseTo(0);
  });

  it("follows yaw: a long thin box rotated 90 degrees", () => {
    const thin: [number, number, number] = [1, 0.1, 0.1]; // 2 m along x before the edit
    const r = ray([0, 0, 5], [0, 0, -1]);
    expect(rayBoxDistance(ray([0.8, 0, 5], [0, 0, -1]), pivot([0, 0, 0]), thin)).toBeCloseTo(4.9);
    expect(rayBoxDistance(r, pivot([0, 0, 0], 90), thin)).toBeCloseTo(4); // now along z
    expect(rayBoxDistance(ray([0.8, 0, 5], [0, 0, -1]), pivot([0, 0, 0], 90), thin)).toBeNull();
  });

  it("follows uniform scale in world units", () => {
    expect(rayBoxDistance(ray([0, 0, 5], [0, 0, -1]), pivot([0, 0, 0], 0, 2), half)).toBeCloseTo(4);
    expect(rayBoxDistance(ray([0, 0, 5], [0, 0, -1]), pivot([0, 0, 0], 0, 0), half)).toBeNull();
  });
});

describe("pickNearest", () => {
  it("picks the nearest of overlapping boxes along the ray", () => {
    const half: [number, number, number] = [0.5, 0.5, 0.5];
    const objects = [
      { id: "far", pivotWorld: pivot([0, 0, -3]), half },
      { id: "near", pivotWorld: pivot([0, 0, 0]), half },
      { id: "aside", pivotWorld: pivot([5, 0, 0]), half },
    ];
    expect(pickNearest(ray([0, 0, 5], [0, 0, -1]), objects)?.id).toBe("near");
    expect(pickNearest(ray([5, 0, 5], [0, 0, -1]), objects)?.id).toBe("aside");
    expect(pickNearest(ray([0, 5, 5], [0, 0, -1]), objects)).toBeNull();
  });

  it("tracks a moved object", () => {
    const half: [number, number, number] = [0.25, 0.45, 0.25];
    const before = [{ id: "chair#1", pivotWorld: pivot([-0.85, 0.45, -1.55]), half }];
    const after = [{ id: "chair#1", pivotWorld: pivot([-0.35, 0.45, -1.55]), half }];
    const r = ray([-0.35, 1.6, 0], new THREE.Vector3(0, -1.15, -1.55).normalize().toArray());
    expect(pickNearest(r, before)).toBeNull();
    expect(pickNearest(r, after)?.id).toBe("chair#1");
  });
});
