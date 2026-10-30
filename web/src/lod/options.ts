// Lane lod: SplatMesh options per mesh. A precomputed Spark .RAD tree when the manifest
// names one (scripts/build_lod.py), otherwise the PLY with a tree built at load time.
import type { SplatMeshOptions } from "@sparkjsdev/spark";
import type { SceneManifest } from "../scene";

export type LodSource = "rad" | "load-time";

export interface LodChoice {
  source: LodSource;
  /** Path relative to the scene folder that the mesh loads. */
  path: string;
  options: Partial<SplatMeshOptions>;
}

/**
 * Path of the .RAD tree for an object (null = background), if the manifest has one. `ply`
 * is the PLY the mesh shows: the object's completed_ply selects completed_lod (the viewer
 * swaps to the completed object once it has been edited, see SceneView), anything else
 * selects lod.
 */
export function radPath(m: SceneManifest, objectId: string | null, ply?: string): string | null {
  if (objectId === null) return m.background_lod ?? null;
  const o = m.objects.find((x) => x.id === objectId);
  if (!o) return null;
  if (o.completed_ply && ply === o.completed_ply) return o.completed_lod ?? null;
  return o.lod ?? null;
}

/**
 * Options for one splat mesh. A .RAD file carries its tree, so it needs no `lod` flag
 * (Spark decodes it straight into lodSplats); a PLY gets `lod: true` (tiny-lod in a
 * worker, about 1-3 s per 1M splats) or "quality" (bhatt-lod, slower).
 */
export function lodChoice(
  ply: string,
  objectId: string | null,
  m: SceneManifest,
  sceneBase: string,
  loadTimeLod: true | "quality",
  lodScale: number,
  paged = false,
): LodChoice {
  const rad = radPath(m, objectId, ply);
  if (rad) {
    const options: Partial<SplatMeshOptions> = { url: `${sceneBase}/${rad}`, lodScale };
    // Paged: chunks stream into Spark's shared page pool (needs HTTP Range or chunked .RAD).
    if (paged) options.paged = true;
    return { source: "rad", path: rad, options };
  }
  return { source: "load-time", path: ply, options: { lod: loadTimeLod, lodScale } };
}
