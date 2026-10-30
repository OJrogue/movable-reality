// Lane ui: TypeScript mirror of Scene.apply_edit and Scene.undo in
// src/movable_reality/scene.py. Used only when the scene server is unreachable, so the
// desktop viewer can still edit (the edits are lost on reload). Keep the semantics in step
// with scene.py: history is append-only and undo appends an entry that points at an index.
import type { SceneManifest, SceneObject, Vec3 } from "../scene";

export type EditOp = "move" | "place" | "rotate" | "remove" | "restore" | "undo";

export interface EditRequest {
  op: EditOp;
  object_id: string | null;
  args?: Record<string, number>;
  instruction?: string | null;
}

export interface HistoryEntry {
  ts: string;
  op: string;
  object_id: string | null;
  args: Record<string, unknown>;
  instruction: string | null;
  verified: boolean | null;
}

/** Same list as UNDOABLE_OPS in scene.py. */
export const UNDOABLE_OPS = ["move", "place", "rotate", "remove", "restore", "insert"];

export class EditError extends Error {}

/** Python's datetime.now(UTC).isoformat(timespec="seconds"). */
export function isoSeconds(now: Date): string {
  return `${now.toISOString().slice(0, 19)}+00:00`;
}

/** Python's float modulo: the result takes the sign of the divisor. */
const mod360 = (x: number) => ((x % 360) + 360) % 360;

export function bboxCenter(o: Pick<SceneObject, "bbox_min" | "bbox_max">): Vec3 {
  return [0, 1, 2].map((i) => (o.bbox_min[i] + o.bbox_max[i]) / 2) as Vec3;
}

function find(m: SceneManifest, id: string): SceneObject {
  const o = m.objects.find((x) => x.id === id);
  if (!o) {
    const known = m.objects.map((x) => x.id).join(", ") || "none";
    throw new EditError(`No object '${id}'. Known objects: ${known}`);
  }
  return o;
}

function num(args: Record<string, number>, key: string, fallback?: number): number {
  const v = args[key] ?? fallback;
  if (typeof v !== "number" || !Number.isFinite(v)) throw new EditError(`'${key}' must be a number`);
  return v;
}

/**
 * Applies one edit to a copy of the manifest and returns the copy. Throws EditError where
 * scene.py raises (unknown op or object, missing argument). An undo with nothing left to
 * undo returns the copy unchanged, as Scene.undo does.
 */
export function applyEdit(m: SceneManifest, req: EditRequest, now = new Date()): SceneManifest {
  const out = structuredClone(m);
  const history = out.history as HistoryEntry[];
  const a = req.args ?? {};
  const instruction = req.instruction ?? null;
  const log = (op: string, objectId: string | null, args: Record<string, unknown>) =>
    history.push({ ts: isoSeconds(now), op, object_id: objectId, args, instruction, verified: null });

  if (req.op === "undo") {
    undo(out, log);
    return out;
  }
  if (req.object_id === null) throw new EditError(`op '${req.op}' needs an object_id`);
  const obj = find(out, req.object_id);
  const t = obj.transform;
  switch (req.op) {
    case "move": {
      const dx = num(a, "dx", 0);
      const dy = num(a, "dy", 0);
      const dz = num(a, "dz", 0);
      t.translation = [t.translation[0] + dx, t.translation[1] + dy, t.translation[2] + dz];
      log("move", obj.id, { dx, dy, dz });
      break;
    }
    case "place": {
      const x = num(a, "x");
      const z = num(a, "z");
      const prev = [...t.translation] as Vec3;
      const c = bboxCenter(obj);
      t.translation = [x - c[0], prev[1], z - c[2]];
      log("place", obj.id, { x, z, prev });
      break;
    }
    case "rotate": {
      const yaw = num(a, "yaw_deg");
      t.yaw_deg = mod360(t.yaw_deg + yaw);
      log("rotate", obj.id, { yaw_deg: yaw });
      break;
    }
    case "remove":
    case "restore":
      obj.visible = req.op === "restore";
      log(req.op, obj.id, {});
      break;
    default:
      throw new EditError(`unknown op '${String(req.op)}'`);
  }
  return out;
}

function undo(m: SceneManifest, log: (op: string, id: string | null, args: Record<string, unknown>) => void): void {
  const history = m.history as HistoryEntry[];
  const undone = new Set(history.filter((e) => e.op === "undo").map((e) => e.args.index));
  for (let i = history.length - 1; i >= 0; i--) {
    const e = history[i];
    if (!UNDOABLE_OPS.includes(e.op) || undone.has(i) || e.object_id === null) continue;
    const obj = find(m, e.object_id);
    const t = obj.transform;
    const a = e.args as Record<string, number>;
    if (e.op === "move") {
      t.translation = [t.translation[0] - a.dx, t.translation[1] - a.dy, t.translation[2] - a.dz];
    } else if (e.op === "place") {
      t.translation = [...(e.args.prev as Vec3)] as Vec3;
    } else if (e.op === "rotate") {
      t.yaw_deg = mod360(t.yaw_deg - a.yaw_deg);
    } else if (e.op === "remove" || e.op === "insert") {
      obj.visible = e.op === "remove";
    } else if (e.op === "restore") {
      obj.visible = false;
    }
    log("undo", e.object_id, { index: i, of: e.op });
    return;
  }
}

/** The object of the newest "remove" in history that is still hidden, for the R key. */
export function lastHidden(m: SceneManifest): string | null {
  const history = m.history as HistoryEntry[];
  for (let i = history.length - 1; i >= 0; i--) {
    const e = history[i];
    if (e.op !== "remove" || e.object_id === null) continue;
    const o = m.objects.find((x) => x.id === e.object_id);
    if (o && !o.visible) return o.id;
  }
  return null;
}
