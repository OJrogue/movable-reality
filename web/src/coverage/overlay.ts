// Pure logic for the coverage overlay (lane coverage): what to draw for each gap and each
// next-best view, from the manifest alone. No three.js objects here, so it is unit-tested.
import type { Coverage, Gap, GapStatus, SceneManifest, Vec3 } from "../scene";

/** Statuses shown as gaps; a gap the user accepted, or one already filled, is hidden. */
export const SHOWN_STATUSES: ReadonlySet<GapStatus> = new Set(["open", "recapture_requested"]);

export interface GapMarker {
  id: string;
  center: Vec3;
  radius: number; // metres, sphere radius
  opacity: number; // 0..1, from the gap score
  color: number; // 0xRRGGBB
  label: string; // for the debug handle and the HUD line
}

export interface ViewMarker {
  gapId: string;
  floor: Vec3; // where to stand (y = 0)
  eye: Vec3; // phone position
  dir: Vec3; // unit aim direction, eye -> target
  length: number; // arrow length, metres
  gain: number;
}

export interface OverlayModel {
  gaps: GapMarker[];
  views: ViewMarker[];
  best: ViewMarker | null; // the view to film first (phone frustum)
  summary: string;
}

export const GAP_COLOR = 0xff2a2a;
export const REQUESTED_COLOR = 0xff8a1f; // recapture requested: orange
export const VIEW_COLOR = 0x29d3ff;

/** Translucent red: 0.15 for a barely flagged gap up to 0.55 for a never-seen one. */
export function gapOpacity(score: number): number {
  const s = Math.min(1, Math.max(0, Number.isFinite(score) ? score : 0));
  return 0.15 + 0.4 * s;
}

/** Sphere radius: the gap radius, clamped so tiny gaps stay visible and huge ones do not
 * swallow the room. */
export function gapRadius(radius: number): number {
  return Math.min(1.5, Math.max(0.08, Number.isFinite(radius) ? radius : 0));
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function norm(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

export function gapLabel(g: Gap): string {
  return g.label ?? (g.near_object ? `near ${g.near_object}` : g.id);
}

export function gapMarkers(cov: Coverage | null | undefined): GapMarker[] {
  if (!cov) return [];
  return cov.gaps
    .filter((g) => SHOWN_STATUSES.has(g.status))
    .map((g) => ({
      id: g.id,
      center: g.center,
      radius: gapRadius(g.radius),
      opacity: gapOpacity(g.score),
      color: g.status === "recapture_requested" ? REQUESTED_COLOR : GAP_COLOR,
      label: gapLabel(g),
    }));
}

/** The best next view of each shown gap (or up to `perGap` of them). */
export function viewMarkers(cov: Coverage | null | undefined, perGap = 1): ViewMarker[] {
  if (!cov) return [];
  const out: ViewMarker[] = [];
  for (const g of cov.gaps) {
    if (!SHOWN_STATUSES.has(g.status)) continue;
    for (const v of (g.next_best_views ?? []).slice(0, perGap)) {
      const d = sub(v.target, v.eye);
      const n = norm(d);
      if (!(n > 1e-6)) continue;
      out.push({
        gapId: g.id,
        floor: [v.eye[0], 0, v.eye[2]],
        eye: v.eye,
        dir: [d[0] / n, d[1] / n, d[2] / n],
        length: Math.min(0.6, Math.max(0.25, 0.3 * n)),
        gain: v.gain,
      });
    }
  }
  return out;
}

/** How much a gap matters on screen: score x (clamped radius)^2, as in bestView. */
export function gapImportance(g: Gap): number {
  const r = gapRadius(g.radius);
  return (Number.isFinite(g.score) ? g.score : 0) * r * r;
}

/** The `limit` most important shown gaps (all when limit is not finite or <= 0). Real
 * captures score 170 to 200 gaps, mostly a few voxels each, and drawing all of them hid
 * the room (31 Oct); the few big ones are what to film again. */
export function topGaps(cov: Coverage, limit: number): Coverage {
  const shown = cov.gaps.filter((g) => SHOWN_STATUSES.has(g.status));
  if (!(limit > 0) || !Number.isFinite(limit) || shown.length <= limit) return cov;
  const keep = new Set(
    [...shown].sort((a, b) => gapImportance(b) - gapImportance(a)).slice(0, limit).map((g) => g.id),
  );
  return { ...cov, gaps: cov.gaps.filter((g) => keep.has(g.id)) };
}

/** `?gapLimit=<n>`: most important gaps drawn, default 12; 0 draws every gap. */
export function gapLimitParam(params: URLSearchParams): number {
  const raw = params.get("gapLimit");
  if (raw === null) return DEFAULT_GAP_LIMIT;
  const n = Math.floor(Number(raw));
  return Number.isFinite(n) && n > 0 ? n : Infinity;
}

export const DEFAULT_GAP_LIMIT = 12;

export function overlayModel(m: SceneManifest | null, perGap = 1, limit = Infinity): OverlayModel {
  const all = m?.coverage ?? null;
  const cov = all ? topGaps(all, limit) : null;
  const gaps = gapMarkers(cov);
  const views = viewMarkers(cov, perGap);
  const share = cov?.gap_share;
  const pct = share == null ? "" : `, ${(100 * share).toFixed(0)}% of visible surfaces`;
  const gen = cov && cov.generated_splats > 0 ? `, ${cov.generated_splats} generated splats` : "";
  const total = gapMarkers(all).length;
  const count = total > gaps.length ? `${gaps.length} of ${total} gaps (largest)` : `${total} gaps`;
  const summary = cov ? `coverage: ${count}${pct}${gen}` : "coverage: not scored";
  return { gaps, views, best: bestView(cov), summary };
}

/** Stable key of what the overlay draws, so unchanged manifests do not rebuild meshes. */
export function overlayKey(model: OverlayModel): string {
  return JSON.stringify([model.gaps, model.views, model.best]);
}

/** `?gaps=1` (or `true`, `on`) turns the overlay on at load; default off. */
export function gapsParam(params: URLSearchParams): boolean {
  const v = (params.get("gaps") ?? "").toLowerCase();
  return v === "1" || v === "true" || v === "on";
}

/** Edge detector for one button per input source: true once per press. */
export class EdgeTrigger<K extends object> {
  private readonly held = new WeakMap<K, boolean>();
  pressed(source: K, down: boolean): boolean {
    const was = this.held.get(source) ?? false;
    this.held.set(source, down);
    return down && !was;
  }
}

export const BEST_VIEW_COLOR = 0xffd21f;

/**
 * The single next view to film first: over every shown gap's views, the largest
 * gain x score x (clamped radius)^2, so a big never-seen gap outranks a small, nearly
 * covered one. Drawn as a phone frustum at the eye, aimed at the target. Null when no
 * shown gap has a view.
 */
export function bestView(cov: Coverage | null | undefined): ViewMarker | null {
  if (!cov) return null;
  let best: ViewMarker | null = null;
  let bestValue = -Infinity;
  for (const g of cov.gaps) {
    if (!SHOWN_STATUSES.has(g.status)) continue;
    const r = gapRadius(g.radius);
    const score = Number.isFinite(g.score) ? g.score : 0;
    for (const v of viewMarkers({ ...cov, gaps: [g] }, Infinity)) {
      const value = v.gain * score * r * r;
      if (value > bestValue) {
        bestValue = value;
        best = v;
      }
    }
  }
  return best;
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function unit(a: Vec3): Vec3 {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

/**
 * Far corners of a phone frustum held at `eye` and aimed along `dir` (unit), +Y up:
 * four points `depth` metres ahead at the given horizontal FOV and landscape aspect, in
 * the order top-left, top-right, bottom-right, bottom-left. A view straight up or down
 * uses +Z as its up hint.
 */
export function frustumCorners(eye: Vec3, dir: Vec3, depth = 0.35, hfovDeg = 65, aspect = 16 / 9): Vec3[] {
  const upHint: Vec3 = Math.abs(dir[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
  const right = unit(cross(dir, upHint));
  const up = cross(right, dir);
  const hw = depth * Math.tan(((hfovDeg / 2) * Math.PI) / 180);
  const hh = hw / aspect;
  const c: Vec3 = [eye[0] + dir[0] * depth, eye[1] + dir[1] * depth, eye[2] + dir[2] * depth];
  const corners: [number, number][] = [
    [-1, 1],
    [1, 1],
    [1, -1],
    [-1, -1],
  ];
  return corners.map(([sx, sy]) => [
    c[0] + sx * hw * right[0] + sy * hh * up[0],
    c[1] + sx * hw * right[1] + sy * hh * up[1],
    c[2] + sx * hw * right[2] + sy * hh * up[2],
  ]);
}

/** `?generated=0` (or `false`, `off`) hides generated splats at load; default shown. */
export function generatedParam(params: URLSearchParams): boolean {
  const v = (params.get("generated") ?? "").toLowerCase();
  return !(v === "0" || v === "false" || v === "off");
}
