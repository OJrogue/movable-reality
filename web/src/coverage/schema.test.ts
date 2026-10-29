import { describe, expect, it } from "vitest";
import type { Coverage, SceneManifest } from "../scene";

// Shape written by src/movable_reality/scene.py (Coverage with next-best views).
const fromPython = {
  gap_map: "coverage/gaps.npz",
  gaps: [
    {
      id: "gap_000",
      label: null,
      near_object: "sofa#1",
      center: [0, 0.3, -1.8],
      radius: 0.5,
      score: 0.9,
      status: "open",
      next_best_views: [{ eye: [1, 1.6, 0.5], target: [0, 0.3, -1.8], visible: 120, gain: 0.4 }],
    },
  ],
  generated_splats: 0,
  gap_share: 0.12,
} satisfies Coverage;

describe("coverage schema", () => {
  it("accepts the manifest shape the Python pipeline writes", () => {
    const m = { coverage: fromPython } as Pick<SceneManifest, "coverage">;
    expect(m.coverage?.gaps[0].next_best_views?.[0].eye[1]).toBe(1.6);
  });

  it("allows gaps from older manifests without next-best views", () => {
    const old: Coverage = {
      gap_map: null,
      gaps: [{ id: "g", label: null, near_object: null, center: [0, 0, 0], radius: 0.2, score: 1, status: "open" }],
      generated_splats: 0,
    };
    expect(old.gaps[0].next_best_views ?? []).toEqual([]);
  });
});
