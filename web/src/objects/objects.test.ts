import { describe, expect, it } from "vitest";
import { backdropFiles, backdropScale, nextStyle, parseStyle, roomFiles, unseenCeilings } from "./index";

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

describe("backdrop", () => {
  it("draws the walls and only an unseen ceiling", () => {
    const r = {
      surfaces: [
        { glb: "f.glb", kind: "floor", seen_share: 0.9 },
        { glb: "c.glb", kind: "ceiling", seen_share: 0.5 },
        { glb: "w0.glb", kind: "wall", seen_share: 0.6 },
      ],
    };
    expect(backdropFiles(r)).toEqual(["w0.glb"]);
  });
  it("pushes every wall of a 4 x 2 m room out by the same distance", () => {
    const sc = backdropScale([[0, 0], [4, 0], [4, 2], [0, 2]], 0.05)!;
    expect(sc.cx).toBe(2);
    expect(sc.cz).toBe(1);
    expect(sc.cx + (4 - sc.cx) * sc.sx).toBeCloseTo(4.05); // east wall
    expect(sc.cz + (0 - sc.cz) * sc.sz).toBeCloseTo(-0.05); // north wall
    expect(backdropScale([[0, 0], [1, 0]])).toBeNull();
  });
});
