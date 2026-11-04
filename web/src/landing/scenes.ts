// The demo's scenes, shared by the landing page (scene cards) and the viewer's scene
// switcher (ui/chrome.ts). Change the ids here when a scene is renamed or re-run.

/** A video the pipeline was given; web copy under web/public/media (gitignored). */
export interface SourceVideo {
  src: string;
  label: string;
  note: string;
}

export interface DemoScene {
  id: string;
  title: string;
  /** One line under the title on the cards. */
  blurb: string;
  /** Card image; the landing page falls back to a frame of the capture, then a gradient. */
  thumb: string;
  /** Every video that went into this room, first the main walk. */
  sources: SourceVideo[];
  /** Room card: label and value, all taken from this scene's pipeline output. */
  facts: [string, string][];
  /** Room card: what is weak in this room, so nobody finds out by surprise. */
  caveat: string;
  /** Where the viewer starts, world metres (the world origin often faces a wall). */
  view?: { eye: [number, number, number]; target: [number, number, number] };
}

export const DEMO_SCENES: readonly DemoScene[] = [
  {
    id: "blender07",
    title: "Living room",
    blurb: "Sofa, TV wall, bookshelf and two windows.",
    thumb: "/media/thumb_blender07.jpg",
    sources: [
      {
        src: "/media/capture_blender07.mp4",
        label: "Main walk, 122 s",
        note: "Rendered in Blender at 1920 x 1080. Through every free corner at standing height, then round the coffee table at knee height, then tilted up at the walls and ceiling.",
      },
      {
        src: "/media/capture_blender07_topup.mp4",
        label: "Top-up, 10 s",
        note: "A short second clip for a corner the main walk missed.",
      },
    ],
    facts: [
      ["Video in", "132 s, 2 clips"],
      ["Frames used", "528 (the sharpest 4 per second)"],
      ["Also given", "Focal length and ceiling height"],
      ["Splats trained", "3.37 million"],
      ["Movable objects", "39"],
      ["Pipeline run", "About 3.5 hours"],
      ["Views it never saw", "28.4 dB PSNR, 0.94 SSIM, 90 views from a separate walk"],
      ["Download", "About 140 MB"],
    ],
    caveat: "The rug and the ceiling are badly smeared, and the corner by the west window is the weakest view. The floor plan came out about 12% too large against the Blender model, and coverage scoring still marks 131 thin spots.",
    view: { eye: [3.6, 1.65, -3.7], target: [1.2, 0.6, -0.3] }, // from the TV corner: sofa, table, armchair
  },
  {
    id: "blender06",
    title: "Bedroom",
    blurb: "Bed with a draped blanket, desk, chair and plants.",
    thumb: "/media/thumb_blender06.jpg",
    sources: [
      {
        src: "/media/capture_blender06.mp4",
        label: "Main walk, 54 s",
        note: "Rendered in Blender at 1920 x 1080. One slow walk around the bed and the desk.",
      },
    ],
    facts: [
      ["Video in", "54 s, 1 clip"],
      ["Frames used", "200 (the sharpest 4 per second)"],
      ["Also given", "Focal length and ceiling height"],
      ["Splats trained", "1.24 million"],
      ["Movable objects", "10"],
      ["Pipeline run", "About 1 hour"],
      ["Frames held out", "36.9 dB PSNR on every 8th frame of the same walk. No separate test walk scored"],
      ["Download", "About 30 MB"],
    ],
    caveat: "The room came out about 5% too large against the Blender model. The curtains and the poster above the bed are smeared, and there is a dark smudge on the floor at the foot of the bed.",
    view: { eye: [0.2, 1.5, -0.9], target: [-1.6, 0.6, 0.5] }, // clear of the floaters by the dresser: bed, window, chair
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
