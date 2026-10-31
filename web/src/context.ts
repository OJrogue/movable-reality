// Shared handle that lane modules plug into. main.ts builds one ViewerContext and passes it
// to each lane's install function (installLod, installCoverage, installPlanes, ...).
// Owned by the renderer lane; other lanes import it but do not change it without a PR note.
import * as THREE from "three";
import type { SparkRenderer } from "@sparkjsdev/spark";
import type { DeviceSettings } from "./devices/profile";
import type { SceneManifest, SceneView } from "./scene";

export type FrameCallback = (time: number, frame: XRFrame | undefined) => void;
/** `cpuMs`: main-thread time from the start of the animation callback to after render(). */
export type FrameEndCallback = (time: number, frame: XRFrame | undefined, cpuMs: number) => void;
export type ManifestCallback = (manifest: SceneManifest, view: SceneView) => void;
export type DeviceCallback = (settings: DeviceSettings) => void;

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
  /** WebXR framebuffer scale requested for the session (?xrScale=, Spark's default 0.5). */
  readonly xrFramebufferScale: number;
  /** False while something else drives the rig (bench mode); desktop controls then pause. */
  controlsEnabled: boolean;
  /** One-line viewer status (loading, errors, object counts); the HUD shows it. */
  status: string;
  /** Called every frame before rendering, in registration order. */
  onFrame(cb: FrameCallback): void;
  /** Called every frame after rendering, in registration order. */
  onFrameEnd(cb: FrameEndCallback): void;
  /** Called after the first manifest loads and after every live change. */
  onManifest(cb: ManifestCallback): void;
  /**
   * Active device profile (lane devices, devices/profile.ts); null until installDevices
   * runs. Lanes read their quality knobs from it.
   */
  device: DeviceSettings | null;
  /** Called when the profile changes (XR enter or exit, a resize across the mobile limit). */
  onDeviceChange(cb: DeviceCallback): void;
}

export function createViewerContext(init: {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  spark: SparkRenderer;
  camera: THREE.PerspectiveCamera;
  rig: THREE.Group;
  params: URLSearchParams;
  xrFramebufferScale: number;
}): ViewerContext & {
  runFrame(time: number, frame: XRFrame | undefined): void;
  runFrameEnd(time: number, frame: XRFrame | undefined, cpuMs: number): void;
  runManifest(manifest: SceneManifest, view: SceneView): void;
  runDeviceChange(settings: DeviceSettings): void;
} {
  const frameCbs: FrameCallback[] = [];
  const frameEndCbs: FrameEndCallback[] = [];
  const manifestCbs: ManifestCallback[] = [];
  const deviceCbs: DeviceCallback[] = [];
  return {
    ...init,
    sceneBase: null,
    view: null,
    manifest: null,
    device: null,
    controlsEnabled: true,
    status: "loading…",
    onFrame: (cb) => void frameCbs.push(cb),
    onFrameEnd: (cb) => void frameEndCbs.push(cb),
    onManifest: (cb) => void manifestCbs.push(cb),
    onDeviceChange: (cb) => void deviceCbs.push(cb),
    runFrame(time, frame) {
      for (const cb of frameCbs) cb(time, frame);
    },
    runFrameEnd(time, frame, cpuMs) {
      for (const cb of frameEndCbs) cb(time, frame, cpuMs);
    },
    runManifest(manifest, view) {
      this.manifest = manifest;
      this.view = view;
      for (const cb of manifestCbs) cb(manifest, view);
    },
    runDeviceChange(settings) {
      this.device = settings;
      for (const cb of deviceCbs) cb(settings);
    },
  };
}
