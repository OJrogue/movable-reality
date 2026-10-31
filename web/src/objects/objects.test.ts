import { describe, expect, it } from "vitest";
import { nextStyle, parseStyle, roomFiles } from "./index";

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
