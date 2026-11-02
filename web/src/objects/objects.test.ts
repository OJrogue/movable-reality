import { describe, expect, it } from "vitest";
import { backdropParts, backdropStretch, nextStyle, parseStyle, roomFiles, unseenCeilings } from "./index";

describe("roomFiles", () => {
  it("prefers the shell quads, then the room mesh", () => {
    const planes = ["room/shell_floor_0.glb", "room/shell_wall_0.glb"];
    expect(roomFiles({ room_planes: planes, room_mesh: "room/room_mesh.glb" })).toEqual(planes);
    expect(roomFiles({ room_planes: [], room_mesh: "room/room_mesh.glb" })).toEqual(["room/room_mesh.glb"]);
    expect(roomFiles({ room_mesh: null })).toEqual([]);
    expect(roomFiles({})).toEqual([]);
  });
});

describe("object styles", () => {
  it("parses the URL value and falls back to scan", () => {
    expect(parseStyle("mesh")).toBe("mesh");
    expect(parseStyle("completed")).toBe("completed");
    expect(parseStyle("scan")).toBe("scan");
    expect(parseStyle("meshonly")).toBe("meshonly");
    expect(parseStyle(null)).toBe("auto");
    expect(parseStyle("glb")).toBe("auto");
  });

  it("cycles auto, scan, completed, mesh, meshonly", () => {
    expect(nextStyle("auto")).toBe("scan");
    expect(nextStyle("scan")).toBe("completed");
    expect(nextStyle("completed")).toBe("mesh");
    expect(nextStyle("mesh")).toBe("meshonly");
    expect(nextStyle("meshonly")).toBe("auto");
  });
});

describe("unseenCeilings", () => {
  it("returns only ceilings the capture did not see", () => {
    const r = {
      surfaces: [
        { glb: "room/shell_floor_0.glb", kind: "floor", seen_share: 0 },
        { glb: "room/shell_ceiling_0.glb", kind: "ceiling", seen_share: 0 },
        { glb: "room/shell_ceiling_1.glb", kind: "ceiling", seen_share: 0.4 },
      ],
    };
    expect(unseenCeilings(r)).toEqual(["room/shell_ceiling_0.glb"]);
    expect(unseenCeilings({})).toEqual([]);
  });
});

describe("backdropParts", () => {
  it("moves each wall out along its own normal and keeps an unseen ceiling in place", () => {
    const r = {
      surfaces: [
        { glb: "f.glb", kind: "floor", seen_share: 0.9, normal: [0, 1, 0] },
        { glb: "c.glb", kind: "ceiling", seen_share: 0, normal: [0, -1, 0] },
        { glb: "w0.glb", kind: "wall", seen_share: 0.6, normal: [0, 0, 1] },
        { glb: "w1.glb", kind: "wall", seen_share: 0.6, normal: [-0.6, 0, 0.8] }, // angled
        { glb: "old.glb", kind: "wall", seen_share: 0.6 }, // shell from before normals
      ],
    };
    const parts = backdropParts(r, 0.05);
    expect(parts.map((p) => p.glb)).toEqual(["w0.glb", "w1.glb", "c.glb", "f.glb"]);
    expect(parts[3].offset).toEqual([0, -0.01, 0]);
    expect(parts[0].offset[2]).toBeCloseTo(-0.05);
    expect(Math.hypot(...parts[1].offset)).toBeCloseTo(0.05);
    expect(parts[2].offset).toEqual([0, 0, 0]);
  });
});

describe("backdropStretch", () => {
  it("makes a wall reach past both ends and a ceiling past its short sides", () => {
    const f = backdropStretch("wall", { x: 4, y: 2.5, z: 0 }, 0.08);
    expect(4 * f).toBeCloseTo(4.16);
    expect(backdropStretch("ceiling", { x: 4, y: 0, z: 5 }, 0.08) * 4).toBeCloseTo(4.16);
    expect(backdropStretch("wall", { x: 0, y: 2.5, z: 0 })).toBe(1);
  });
});
