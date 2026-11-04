import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { clampToOutline, isConvex, MAX_PITCH_RAD, orbitRig, type Outline, panRig } from "./orbit";

const rigAt = (x: number, y: number, z: number) => {
  const rig = new THREE.Group();
  rig.position.set(x, y, z);
  return rig;
};

describe("orbit", () => {
  it("keeps the distance to the pivot", () => {
    const rig = rigAt(0, 1, 3);
    const pivot = new THREE.Vector3(0, 1, 0);
    orbitRig(rig, pivot, 120, -40);
    expect(rig.position.distanceTo(pivot)).toBeCloseTo(3);
  });

  it("keeps the pivot at the same place in view", () => {
    const rig = rigAt(0, 1, 3); // looks along -Z, so the pivot is straight ahead
    const pivot = new THREE.Vector3(0, 1, 0);
    orbitRig(rig, pivot, 200, 60);
    const ahead = new THREE.Vector3(0, 0, -1).applyQuaternion(rig.quaternion).multiplyScalar(3).add(rig.position);
    expect(ahead.distanceTo(pivot)).toBeLessThan(1e-6);
  });

  it("moves the camera against the drag, so the room follows the mouse", () => {
    const rig = rigAt(0, 1, 3);
    orbitRig(rig, new THREE.Vector3(0, 1, 0), 100, 0);
    expect(rig.position.x).toBeLessThan(0); // drag right: camera goes left
    const down = rigAt(0, 1, 3);
    orbitRig(down, new THREE.Vector3(0, 1, 0), 0, 100);
    expect(down.position.y).toBeGreaterThan(1); // drag down: camera goes up
  });

  it("stops before straight down and never rolls", () => {
    const rig = rigAt(0, 1, 3);
    for (let i = 0; i < 40; i++) orbitRig(rig, new THREE.Vector3(0, 1, 0), 30, 200);
    const e = new THREE.Euler().setFromQuaternion(rig.quaternion, "YXZ");
    expect(Math.abs(e.x)).toBeLessThanOrEqual(MAX_PITCH_RAD + 1e-6);
    expect(Math.abs(e.z)).toBeLessThan(1e-6);
  });
});

describe("pan", () => {
  it("moves the rig against the mouse, scaled to the distance", () => {
    const near = rigAt(0, 0, 0);
    const far = rigAt(0, 0, 0);
    panRig(near, 100, 0, 1, 70, 800);
    panRig(far, 100, 0, 4, 70, 800);
    expect(near.position.x).toBeLessThan(0);
    expect(far.position.x / near.position.x).toBeCloseTo(4);
  });
});

describe("room outline", () => {
  const square: Outline = [[0, 0], [4, 0], [4, 3], [0, 3]];
  it("leaves a point inside alone", () => {
    expect(clampToOutline(2, 1, square, 0.2)).toEqual([2, 1]);
  });
  it("pushes a point back inside, also from beyond a corner", () => {
    expect(clampToOutline(5, 1, square, 0.2)[0]).toBeCloseTo(3.8);
    const [x, z] = clampToOutline(-1, 9, square, 0.2);
    expect(x).toBeCloseTo(0.2);
    expect(z).toBeCloseTo(2.8);
  });
  it("works for a room turned against the axes", () => {
    const turned: Outline = [[4.039, -5.079], [4.698, 1.179], [-0.626, 1.74], [-1.285, -4.519]];
    expect(isConvex(turned)).toBe(true);
    expect(clampToOutline(3.6, -3.7, turned, 0.15)).toEqual([3.6, -3.7]);
    const [x] = clampToOutline(8, -3.7, turned, 0.15);
    expect(x).toBeLessThan(4.3);
  });
  it("does not call an L-shaped room convex", () => {
    expect(isConvex([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]])).toBe(false);
    expect(isConvex(square)).toBe(true);
  });
});
