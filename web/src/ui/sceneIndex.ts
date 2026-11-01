// Scene list for the /scenes/ page: a summary of each manifest (built by the dev server,
// GET /__scenes in vite.config.ts) and the viewer links that make sense for it. Pure, so
// the dev server can import it and vitest can test it.
import type { SceneManifest } from "../scene";

export interface SceneSummary {
  id: string;
  objects: number;
  visible: number;
  generated: number;
  edits: number;
  lod: boolean; // every object and the background have a .RAD tree
  planes: number; // baked floor/ceiling quads
  gaps: number | null; // null without a coverage section
  completed: number; // objects with SAM 3D completions
  meshes: number; // objects with a SAM 3D mesh
  room: boolean; // room_planes or room_mesh, for ?objects=meshonly
}

export interface SceneLink {
  label: string;
  url: string;
  hint: string;
}

export function summarize(m: SceneManifest): SceneSummary {
  const objects = m.objects ?? [];
  return {
    id: m.scene_id,
    objects: objects.length,
    visible: objects.filter((o) => o.visible).length,
    generated: objects.filter((o) => o.source === "generated").length,
    edits: m.history?.length ?? 0,
    lod: !!m.background_lod && objects.length > 0 && objects.every((o) => !!o.lod),
    planes: m.planes?.length ?? 0,
    gaps: m.coverage ? m.coverage.gaps.length : null,
    completed: objects.filter((o) => !!o.completed_ply).length,
    meshes: objects.filter((o) => !!o.mesh).length,
    room: (m.room_planes?.length ?? 0) > 0 || !!m.room_mesh,
  };
}

export function sceneLinks(s: SceneSummary): SceneLink[] {
  const q = (extra: Record<string, string> = {}) =>
    `/?${new URLSearchParams({ scene: s.id, ...extra })}`;
  const links: SceneLink[] = [{ label: "Open", url: q(), hint: "viewer with the agent server" }];
  links.push({ label: "Local edits", url: q({ server: "0" }), hint: "no API calls, edits stay in the tab" });
  if (s.lod) links.push({ label: "LOD off", url: q({ mode: "naive" }), hint: "plain PLYs, for comparison" });
  if (s.gaps !== null) links.push({ label: "Gaps", url: q({ gaps: "1" }), hint: `${s.gaps} coverage gaps` });
  if (s.completed > 0) {
    links.push({ label: "Completed", url: q({ objects: "completed" }), hint: "SAM 3D splats" });
  }
  if (s.meshes > 0) links.push({ label: "Meshes", url: q({ objects: "mesh" }), hint: "SAM 3D meshes" });
  if (s.meshes > 0 && s.room) {
    links.push({ label: "Mesh only", url: q({ objects: "meshonly" }), hint: "no splats at all" });
  }
  return links;
}
