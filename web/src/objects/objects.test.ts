import { describe, expect, it } from "vitest";
import { nextStyle, parseStyle } from "./index";

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
