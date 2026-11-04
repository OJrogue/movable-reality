import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { formatMeasurement, measureBetween, surfaceDepth } from "./measure";

describe("surfaceDepth", () => {
  it("takes the nearest place where enough splats agree", () => {
    expect(surfaceDepth([3.0, 3.01, 3.02, 3.03, 3.2, 3.21])).toBeCloseTo(3.02);
  });
  it("skips lone floaters in front of the surface", () => {
    expect(surfaceDepth([0.4, 1.1, 2.0, 2.01, 2.02, 2.03])).toBeCloseTo(2.02);
  });
  it("is null with too few splats", () => {
    expect(surfaceDepth([])).toBeNull();
    expect(surfaceDepth([1, 2, 3, 4])).toBeNull();
  });
});

describe("tape measure", () => {
  it("splits a slanted line into across and up", () => {
    const m = measureBetween(new THREE.Vector3(0, 0, 0), new THREE.Vector3(1.2, 0.9, 0));
    expect(m.total).toBeCloseTo(1.5);
    expect(m.horizontal).toBeCloseTo(1.2);
    expect(m.vertical).toBeCloseTo(0.9);
    expect(formatMeasurement(m)).toBe("1.50 m (1.20 m across, 90 cm up)");
  });

  it("shows only the total for a level or vertical line, in cm under a metre", () => {
    expect(formatMeasurement(measureBetween(new THREE.Vector3(), new THREE.Vector3(0, 0.85, 0)))).toBe("85 cm");
    expect(formatMeasurement(measureBetween(new THREE.Vector3(), new THREE.Vector3(2, 0.01, 1)))).toBe("2.24 m");
  });
});
