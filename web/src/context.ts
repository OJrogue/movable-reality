// Shared handle that lane modules plug into. main.ts builds one ViewerContext and passes it
// to each lane's install function (installLod, installCoverage, installPlanes, ...).
// Owned by the renderer lane; other lanes import it but do not change it without a PR note.
import * as THREE from "three";
import type { SparkRenderer } from "@sparkjsdev/spark";
import type { SceneManifest, SceneView } from "./scene";

export type FrameCallback = (time: number, frame: XRFrame | undefined) => void;
export type ManifestCallback = (manifest: SceneManifest, view: SceneView) => void;

export interface ViewerContext {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly spark: SparkRenderer;
  readonly camera: THREE.PerspectiveCamera;
  /** Moved by controls on desktop and in XR; the camera is its child. */
  readonly rig: THREE.Group;
  readonly params: URLSearchParams;
  /** URL of the scene folder, e.g. "/scenes/room01"; null for ?ply= quick looks. */
  sceneBase: string | null;
  /** Set once a manifest has loaded. */
  view: SceneView | null;
  /** Latest manifest; updated on every live change. */
  manifest: SceneManifest | null;
  /**
   * Set at install time by a lane that changes how splat meshes are built (lane lod).
   * main.ts copies it onto the SceneView before the first manifest is applied.
   */
  meshOptions?: SceneView["meshOptions"];
  /** Called every frame before rendering, in registration order. */
  onFrame(cb: FrameCallback): void;
  /** Called after the first manifest loads and after every live change. */
  onManifest(cb: ManifestCallback): void;
}

export function createViewerContext(init: {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  spark: SparkRenderer;
  camera: THREE.PerspectiveCamera;
  rig: THREE.Group;
  params: URLSearchParams;
}): ViewerContext & {
  runFrame(time: number, frame: XRFrame | undefined): void;
  runManifest(manifest: SceneManifest, view: SceneView): void;
} {
  const frameCbs: FrameCallback[] = [];
  const manifestCbs: ManifestCallback[] = [];
  return {
    ...init,
    sceneBase: null,
    view: null,
    manifest: null,
    onFrame: (cb) => void frameCbs.push(cb),
    onManifest: (cb) => void manifestCbs.push(cb),
    runFrame(time, frame) {
      for (const cb of frameCbs) cb(time, frame);
    },
    runManifest(manifest, view) {
      this.manifest = manifest;
      this.view = view;
      for (const cb of manifestCbs) cb(manifest, view);
    },
  };
}
