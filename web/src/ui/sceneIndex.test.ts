import { describe, expect, it } from "vitest";
import type { SceneManifest, SceneObject } from "../scene";
import { sceneLinks, summarize } from "./sceneIndex";

const obj = (id: string, extra: Partial<SceneObject> = {}): SceneObject => ({
  id,
  label: id,
  ply: `objects/${id}.ply`,
  completed_ply: null,
  bbox_min: [0, 0, 0],
  bbox_max: [1, 1, 1],
  transform: { translation: [0, 0, 0], yaw_deg: 0, scale: 1 },
  visible: true,
  source: "scan",
  prompt: null,
  ...extra,
});

const manifest = (extra: Partial<SceneManifest> = {}): SceneManifest => ({
  version: 1,
  scene_id: "room",
  units: "m",
  world_from_recon: [],
  background_ply: "objects/background.ply",
  objects: [obj("a"), obj("b")],
  history: [],
  ...extra,
});

const labels = (m: SceneManifest) => sceneLinks(summarize(m)).map((l) => l.label);

describe("scene index", () => {
  it("offers only the plain views for a bare scene", () => {
    expect(labels(manifest())).toEqual(["Open", "Local edits"]);
  });

  it("needs LOD on every object and the background", () => {
    const partial = manifest({ background_lod: "lod/bg.rad", objects: [obj("a", { lod: "lod/a.rad" }), obj("b")] });
    expect(summarize(partial).lod).toBe(false);
    const full = manifest({ background_lod: "lod/bg.rad", objects: [obj("a", { lod: "lod/a.rad" })] });
    expect(labels(full)).toContain("LOD off");
  });

  it("adds mesh-only only with meshes and a room shell", () => {
    const meshes = [obj("a", { completed_ply: "c.ply", mesh: "a.glb" })];
    expect(labels(manifest({ objects: meshes }))).toEqual(["Open", "Local edits", "Completed", "Meshes"]);
    expect(labels(manifest({ objects: meshes, room_planes: ["planes/floor.glb"] }))).toContain("Mesh only");
  });

  it("counts gaps, hidden, generated objects and edits", () => {
    const s = summarize(
      manifest({
        objects: [obj("a", { visible: false }), obj("gen", { source: "generated" })],
        coverage: { gap_map: null, gaps: [], generated_splats: 0 },
        history: [{}, {}],
      }),
    );
    expect(s).toMatchObject({ objects: 2, visible: 1, generated: 1, edits: 2, gaps: 0 });
    expect(sceneLinks(s).find((l) => l.label === "Gaps")?.url).toBe("/?scene=room&gaps=1");
  });
});
