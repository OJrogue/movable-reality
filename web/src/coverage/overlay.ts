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

export function overlayModel(m: SceneManifest | null, perGap = 1): OverlayModel {
  const cov = m?.coverage ?? null;
  const gaps = gapMarkers(cov);
  const views = viewMarkers(cov, perGap);
  const share = cov?.gap_share;
  const pct = share == null ? "" : `, ${(100 * share).toFixed(0)}% of visible room`;
  const gen = cov && cov.generated_splats > 0 ? `, ${cov.generated_splats} generated splats` : "";
  const summary = cov ? `coverage: ${gaps.length} gaps${pct}${gen}` : "coverage: not scored";
  return { gaps, views, summary };
}

/** Stable key of what the overlay draws, so unchanged manifests do not rebuild meshes. */
export function overlayKey(model: OverlayModel): string {
  return JSON.stringify([model.gaps, model.views]);
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
