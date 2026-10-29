// Lane lod: pure priority logic (no three.js scene, no Spark), tested in priority.test.ts.
// The object under the pointer or controller ray and the object of the latest edit get a
// higher SplatMesh.lodScale; the others return to the base scale. Changes ramp smoothly so
// the LOD cut does not pop.
import type { SceneManifest } from "../scene";

export interface PriorityConfig {
  /** lodScale for prioritised objects. 2.0 = about 2x finer splats on that object. */
  priorityScale: number;
  /** lodScale for every other mesh, background included. */
  baseScale: number;
  /** Time to cover about 95% of a change in lodScale. 0 = jump. */
  rampMs: number;
}

export const DEFAULT_PRIORITY: PriorityConfig = { priorityScale: 2, baseScale: 1, rampMs: 300 };

/** Read ?lodPriority=<scale>&lodRamp=<ms>; invalid values fall back to the defaults. */
export function parsePriorityConfig(params: URLSearchParams): PriorityConfig {
  const num = (key: string, fallback: number, min: number, max: number): number => {
    const raw = params.get(key);
    const v = raw === null ? NaN : Number(raw);
    return Number.isFinite(v) && v >= min && v <= max ? v : fallback;
  };
  return {
    priorityScale: num("lodPriority", DEFAULT_PRIORITY.priorityScale, 0.1, 16),
    baseScale: DEFAULT_PRIORITY.baseScale,
    rampMs: num("lodRamp", DEFAULT_PRIORITY.rampMs, 0, 10_000),
  };
}

export interface Focus {
  /** Object under the controller ray (XR) or mouse (desktop). */
  hovered: string | null;
  /** Object of the latest edit in the manifest history. */
  edited: string | null;
}

interface HistoryLike {
  object_id?: string | null;
}

/**
 * Object of the latest history entry that names one, or null. A removed (hidden) object
 * gets no priority: nothing of it is drawn.
 */
export function latestEditedObject(m: SceneManifest): string | null {
  for (let i = m.history.length - 1; i >= 0; i--) {
    const id = (m.history[i] as HistoryLike | null)?.object_id;
    if (!id) continue;
    const obj = m.objects.find((o) => o.id === id);
    return obj && obj.visible ? id : null;
  }
  return null;
}

/** Target lodScale for every object in the manifest. */
export function targetScales(m: SceneManifest, focus: Focus, cfg: PriorityConfig): Map<string, number> {
  const out = new Map<string, number>();
  for (const o of m.objects) {
    const prioritised = o.id === focus.hovered || o.id === focus.edited;
    out.set(o.id, prioritised ? cfg.priorityScale : cfg.baseScale);
  }
  return out;
}

/** Below this difference a ramp snaps to its target, so it ends in finite time. */
export const RAMP_EPSILON = 0.005;

/**
 * One step of an exponential ramp from `current` toward `target`. The time constant is
 * rampMs / 3, so about 95% of the change is covered after rampMs. Frame-rate independent:
 * two steps of dt give the same result as one step of 2 dt.
 */
export function rampScale(current: number, target: number, dtMs: number, rampMs: number): number {
  if (rampMs <= 0 || Math.abs(target - current) <= RAMP_EPSILON) return target;
  const k = 1 - Math.exp((-3 * Math.max(0, dtMs)) / rampMs);
  const next = current + (target - current) * k;
  return Math.abs(target - next) <= RAMP_EPSILON ? target : next;
}

/** "chair#1 (hover, edit)" style labels for the log. */
export function focusReason(id: string, focus: Focus): string {
  const why = [id === focus.hovered ? "hover" : "", id === focus.edited ? "edit" : ""].filter(Boolean);
  return why.length ? `${id} (${why.join(", ")})` : id;
}
