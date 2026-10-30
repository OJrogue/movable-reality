import { describe, expect, it } from "vitest";
import type { Coverage, Gap, SceneManifest } from "../scene";
import {
  EdgeTrigger,
  GAP_COLOR,
  REQUESTED_COLOR,
  gapMarkers,
  gapOpacity,
  gapRadius,
  gapsParam,
  overlayKey,
  overlayModel,
  viewMarkers,
  bestView,
  frustumCorners,
  generatedParam,
} from "./overlay";

function gap(over: Partial<Gap> = {}): Gap {
  return {
    id: "gap_000",
    label: null,
    near_object: "sofa#1",
    center: [0, 0.3, -1.8],
    radius: 0.5,
    score: 1,
    status: "open",
    next_best_views: [{ eye: [1, 1.6, 0], target: [0, 0.3, -1.8], visible: 50, gain: 0.4 }],
    ...over,
  };
}

function cov(gaps: Gap[], extra: Partial<Coverage> = {}): Coverage {
  return { gap_map: null, gaps, generated_splats: 0, ...extra };
}

describe("gap markers", () => {
  it("map score to opacity and radius to sphere size", () => {
    expect(gapOpacity(0)).toBeCloseTo(0.15);
    expect(gapOpacity(1)).toBeCloseTo(0.55);
    expect(gapOpacity(2)).toBeCloseTo(0.55);
    expect(gapOpacity(Number.NaN)).toBeCloseTo(0.15);
    expect(gapRadius(0.01)).toBe(0.08);
    expect(gapRadius(0.5)).toBe(0.5);
    expect(gapRadius(9)).toBe(1.5);
  });

  it("show open and requested gaps, hide accepted and filled ones", () => {
    const m = gapMarkers(
      cov([
        gap({ id: "a" }),
        gap({ id: "b", status: "recapture_requested" }),
        gap({ id: "c", status: "accepted" }),
        gap({ id: "d", status: "generated" }),
        gap({ id: "e", status: "recaptured" }),
      ]),
    );
    expect(m.map((g) => g.id)).toEqual(["a", "b"]);
    expect(m[0].color).toBe(GAP_COLOR);
    expect(m[1].color).toBe(REQUESTED_COLOR);
  });

  it("label by Muse Wonder's name, else the nearest object, else the id", () => {
    const m = gapMarkers(
      cov([gap({ label: "behind the sofa" }), gap({ id: "x" }), gap({ id: "y", near_object: null })]),
    );
    expect(m.map((g) => g.label)).toEqual(["behind the sofa", "near sofa#1", "y"]);
  });

  it("handle a missing coverage section (older manifests)", () => {
    expect(gapMarkers(undefined)).toEqual([]);
    expect(viewMarkers(null)).toEqual([]);
    const model = overlayModel({ coverage: undefined } as unknown as SceneManifest);
    expect(model.summary).toBe("coverage: not scored");
  });
});

describe("next-best view markers", () => {
  it("stand on the floor below the eye and point at the target", () => {
    const [v] = viewMarkers(cov([gap()]));
    expect(v.floor).toEqual([1, 0, 0]);
    expect(v.eye).toEqual([1, 1.6, 0]);
    const len = Math.hypot(...v.dir);
    expect(len).toBeCloseTo(1);
    const d = [0 - 1, 0.3 - 1.6, -1.8 - 0];
    const n = Math.hypot(...d);
    v.dir.forEach((c, i) => expect(c).toBeCloseTo(d[i] / n));
    expect(v.length).toBeGreaterThanOrEqual(0.25);
    expect(v.length).toBeLessThanOrEqual(0.6);
  });

  it("skip gaps without views, hidden gaps and degenerate views", () => {
    const views = viewMarkers(
      cov([
        gap({ id: "a", next_best_views: undefined }),
        gap({ id: "b", status: "accepted" }),
        gap({ id: "c", next_best_views: [{ eye: [1, 1, 1], target: [1, 1, 1], visible: 1, gain: 1 }] }),
        gap({ id: "d" }),
      ]),
    );
    expect(views.map((v) => v.gapId)).toEqual(["d"]);
  });

  it("can show more than one view per gap", () => {
    const g = gap();
    g.next_best_views = [g.next_best_views![0], { eye: [-1, 1.6, 0], target: [0, 0.3, -1.8], visible: 30, gain: 0.2 }];
    expect(viewMarkers(cov([g]), 1)).toHaveLength(1);
    expect(viewMarkers(cov([g]), 3)).toHaveLength(2);
  });
});

describe("overlay model", () => {
  it("summarises gaps, gap share and generated splats", () => {
    const m = { coverage: cov([gap(), gap({ id: "b" })], { gap_share: 0.123, generated_splats: 42 }) };
    const model = overlayModel(m as unknown as SceneManifest);
    expect(model.summary).toBe("coverage: 2 gaps, 12% of visible surfaces, 42 generated splats");
  });

  it("has a key that only changes when the drawing changes", () => {
    const a = overlayModel({ coverage: cov([gap()]) } as unknown as SceneManifest);
    const b = overlayModel({ coverage: cov([gap()], { generated_splats: 3 }) } as unknown as SceneManifest);
    const c = overlayModel({ coverage: cov([gap({ radius: 0.7 })]) } as unknown as SceneManifest);
    expect(overlayKey(a)).toBe(overlayKey(b));
    expect(overlayKey(a)).not.toBe(overlayKey(c));
  });
});

describe("toggles", () => {
  it("reads ?gaps=1", () => {
    expect(gapsParam(new URLSearchParams("gaps=1"))).toBe(true);
    expect(gapsParam(new URLSearchParams("gaps=on"))).toBe(true);
    expect(gapsParam(new URLSearchParams("gaps=0"))).toBe(false);
    expect(gapsParam(new URLSearchParams(""))).toBe(false);
  });

  it("fires once per press per source", () => {
    const t = new EdgeTrigger<object>();
    const left = {};
    const right = {};
    expect(t.pressed(left, true)).toBe(true);
    expect(t.pressed(left, true)).toBe(false);
    expect(t.pressed(right, true)).toBe(true);
    expect(t.pressed(left, false)).toBe(false);
    expect(t.pressed(left, true)).toBe(true);
  });
});

describe("best next view", () => {
  it("prefers the view with most gain on the biggest, least covered gap", () => {
    const small = gap({ id: "gap_001", radius: 0.2, score: 1 });
    const big = gap({
      id: "gap_000",
      radius: 1.0,
      score: 0.9,
      next_best_views: [
        { eye: [2, 1.6, 0], target: [0, 0.3, -1.8], visible: 9, gain: 0.2 },
        { eye: [-2, 1.6, 0], target: [0, 0.3, -1.8], visible: 30, gain: 0.6 },
      ],
    });
    const best = bestView(cov([small, big]));
    expect(best?.gapId).toBe("gap_000");
    expect(best?.eye).toEqual([-2, 1.6, 0]);
  });

  it("ignores gaps that are no longer shown and returns null without views", () => {
    expect(bestView(cov([gap({ status: "generated" })]))).toBeNull();
    expect(bestView(cov([gap({ next_best_views: [] })]))).toBeNull();
    expect(bestView(null)).toBeNull();
  });

  it("is part of the overlay model and its key", () => {
    const m = { coverage: cov([gap()]) } as SceneManifest;
    const model = overlayModel(m, 0);
    expect(model.views).toEqual([]); // ?gapViews=0: only the best view
    expect(model.best?.gapId).toBe("gap_000");
    expect(overlayKey(model)).not.toBe(overlayKey({ ...model, best: null }));
  });
});

describe("phone frustum", () => {
  it("puts four corners ahead of the eye, wider than tall, around the aim", () => {
    const c = frustumCorners([0, 1.6, 0], [0, 0, -1], 1, 90, 2);
    for (const p of c) expect(p[2]).toBeCloseTo(-1);
    expect(c[0][0]).toBeCloseTo(-1); // top-left: x = -tan(45 deg)
    expect(c[0][1]).toBeCloseTo(2.1); // y = 1.6 + 0.5
    expect(c[2][0]).toBeCloseTo(1);
    expect(c[2][1]).toBeCloseTo(1.1);
  });

  it("does not degenerate when aimed straight down", () => {
    const c = frustumCorners([0, 1.6, 0], [0, -1, 0]);
    for (const p of c) for (const x of p) expect(Number.isFinite(x)).toBe(true);
    expect(Math.hypot(c[0][0] - c[2][0], c[0][2] - c[2][2])).toBeGreaterThan(0.1);
  });
});

describe("generated toggle parameter", () => {
  it("shows generated splats unless ?generated=0", () => {
    expect(generatedParam(new URLSearchParams(""))).toBe(true);
    expect(generatedParam(new URLSearchParams("generated=0"))).toBe(false);
    expect(generatedParam(new URLSearchParams("generated=off"))).toBe(false);
    expect(generatedParam(new URLSearchParams("generated=1"))).toBe(true);
  });
});
