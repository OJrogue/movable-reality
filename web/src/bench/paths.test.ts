import { describe, expect, it } from "vitest";
import { getPath, PATHS, yawTowards } from "./paths";

const ROOM = { x: 3, z: 2.5 };
const angleDiff = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));

describe("bench paths", () => {
  it("yawTowards: 0 looks along -Z, +90 deg looks along -X", () => {
    expect(yawTowards(0, 0, 0, -1)).toBeCloseTo(0, 12);
    expect(yawTowards(0, 0, -1, 0)).toBeCloseTo(Math.PI / 2, 12);
  });

  for (const path of Object.values(PATHS)) {
    it(`${path.name}: deterministic, inside the room, gentle on a standing user`, () => {
      const dt = 1 / 72;
      let prev = path.pose(0);
      let maxSpeed = 0;
      let maxTurn = 0;
      for (let i = 1; i <= 72 * 180; i++) {
        const p = path.pose(i * dt);
        expect(Math.abs(p.x)).toBeLessThan(ROOM.x - 0.5);
        expect(Math.abs(p.z)).toBeLessThan(ROOM.z - 0.5);
        maxSpeed = Math.max(maxSpeed, Math.hypot(p.x - prev.x, p.z - prev.z) / dt);
        maxTurn = Math.max(maxTurn, Math.abs(angleDiff(p.yaw, prev.yaw)) / dt);
        prev = p;
      }
      expect(maxSpeed).toBeLessThan(0.5); // m/s
      expect((maxTurn * 180) / Math.PI).toBeLessThan(30); // deg/s
      expect(path.pose(42.123)).toEqual(path.pose(42.123));
    });
  }

  it("defaults to walk and rejects unknown names", () => {
    expect(getPath(null)?.name).toBe("walk");
    expect(getPath("nope")).toBeNull();
  });
});
