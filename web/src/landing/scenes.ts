// The demo's scenes, shared by the landing page (scene cards) and the viewer's scene
// switcher (ui/chrome.ts). Change the ids here when a scene is renamed or re-run.

export interface DemoScene {
  id: string;
  title: string;
  /** One line under the title on the cards. */
  blurb: string;
  /** "synthetic" for the Blender ground-truth rooms, "real" for a phone capture. */
  kind: "synthetic" | "real";
  /** Card image; the landing page falls back to a frame of the capture, then a gradient. */
  thumb: string;
  /** Web copy of the video the scene was trained on (gitignored, see web/public/media). */
  capture: string;
  /** What the capture is, shown under it. */
  captureNote: string;
  /** Where the viewer starts, world metres (the world origin often faces a wall). */
  view?: { eye: [number, number, number]; target: [number, number, number] };
}

export const DEMO_SCENES: readonly DemoScene[] = [
  {
    id: "blender07",
    title: "Living room",
    blurb: "Sofa, TV wall, bookshelf and two windows.",
    kind: "synthetic",
    thumb: "/media/thumb_blender07.jpg",
    capture: "/media/capture_blender07.mp4",
    captureNote: "A 122 s walk through the whole room; the sharpest 4 frames per second are kept.",
    view: { eye: [3.6, 1.65, -3.7], target: [1.2, 0.6, -0.3] }, // from the TV corner: sofa, table, armchair
  },
  {
    id: "blender06",
    title: "Bedroom",
    blurb: "Cloth-draped bed, desk, toys and posters.",
    kind: "synthetic",
    thumb: "/media/thumb_blender06.jpg",
    capture: "/media/capture_blender06.mp4",
    captureNote: "A 54 s walk around the bed and the desk; the sharpest 4 frames per second are kept.",
    view: { eye: [0.5, 1.65, -1.3], target: [-1.7, 0.5, 0.6] }, // from the desk corner: bed, window, desk
  },
  {
    id: "room01",
    title: "A real room",
    blurb: "Filmed on a phone, nothing staged. Coming soon.",
    kind: "real",
    thumb: "/media/thumb_room01.jpg",
    capture: "/media/capture_room01.mp4",
    captureNote: "One handheld walk with a phone.",
  },
];

export const findDemoScene = (id: string | null): DemoScene | undefined =>
  DEMO_SCENES.find((s) => s.id === id);

/** Viewer URL for a scene. Keeps the current page's base override, if any. */
export function viewerUrl(id: string, base: string | null = null): string {
  const q = new URLSearchParams({ scene: id });
  if (base) q.set("base", base);
  return `/?${q}`;
}

/** "updating": openable, but the pipeline is rerunning a stage; "processing": no manifest yet. */
export type SceneAvailability = "ready" | "updating" | "processing" | "missing";

interface IndexEntry {
  id: string;
  ready: boolean;
  pipeline?: { stages: Record<string, { state: string }> };
}

export function availabilityOf(s: IndexEntry): SceneAvailability {
  const running = Object.values(s.pipeline?.stages ?? {}).some((st) => st.state === "running");
  if (!s.ready) return s.pipeline ? "processing" : "missing";
  return running ? "updating" : "ready";
}

/**
 * Whether a scene can be opened. The dev server's /__scenes index also knows scenes whose
 * pipeline is still running; a static host only answers for scene.json itself.
 */
export async function sceneAvailability(
  ids: readonly string[],
  base = "/scenes",
  fetchFn: typeof fetch = fetch,
): Promise<Map<string, SceneAvailability>> {
  const out = new Map<string, SceneAvailability>(ids.map((id) => [id, "missing"]));
  try {
    const r = await fetchFn("/__scenes", { cache: "no-store" });
    if (r.ok && (r.headers.get("content-type") ?? "").includes("json")) {
      for (const s of (await r.json()) as IndexEntry[]) {
        if (out.has(s.id)) out.set(s.id, availabilityOf(s));
      }
      return out;
    }
  } catch {
    // not the dev server
  }
  await Promise.all(
    ids.map(async (id) => {
      try {
        const r = await fetchFn(`${base}/${id}/scene.json`, { method: "HEAD", cache: "no-store" });
        if (r.ok && (r.headers.get("content-type") ?? "").includes("json")) out.set(id, "ready");
      } catch {
        // stays missing
      }
    }),
  );
  return out;
}
