// Scripted views for scripts/blender_viewer_eval.py (dev server only). In the console, or
// from a browser driver:
//   await mrEvalRun("baseline")                       // eval/viewer_views.json -> eval/viewer/baseline/
//   await mrEvalRun("recon", "viewer_views_recon.json", 2000)
// Each view puts the camera at the pose (window.mrCamera, main.ts), waits for Spark's sort
// and the LOD cut to catch up, and POSTs the canvas as PNG to /__eval (vite.config.ts).
// Load the viewer with ?pristine=1&hud=0&budget=0&onboarding=0 at the GT aspect ratio.

interface EvalViews {
  fov_deg: number;
  views: { name: string; c2w: number[][] }[];
}

import * as THREE from "three";
import { matrixFromRows } from "../scene";

export interface EvalHost {
  sceneId: string | null;
  canvas: HTMLCanvasElement;
  camera: THREE.PerspectiveCamera;
  rig: THREE.Object3D; // the camera's parent (desktop walk, XR reference space)
  scene: THREE.Scene;
  setControls: (on: boolean) => void;
}

type Win = {
  mrCamera: (rows?: number[][], fovDeg?: number) => void;
  mrCapture: () => string;
  mrEvalRun?: (run: string, file?: string, waitMs?: number, warmMs?: number) => Promise<{ saved: number; size: number[] }>;
};

export function installEvalViews(host: EvalHost): void {
  const { sceneId, canvas, camera, rig } = host;
  const w = window as unknown as Win & { mrScene?: THREE.Scene };
  w.mrScene = host.scene; // console inspection
  // Camera-to-world rows in the viewer's world frame (three.js camera axes) and a vertical
  // field of view in degrees. Turns the desktop controls off until called with no pose.
  w.mrCamera = (rows, fovDeg) => {
    host.setControls(!rows);
    if (!rows) return;
    rig.position.set(0, 0, 0);
    rig.quaternion.identity();
    matrixFromRows(rows).decompose(camera.position, camera.quaternion, new THREE.Vector3());
    if (fovDeg) camera.fov = fovDeg;
    camera.updateProjectionMatrix();
  };
  w.mrEvalRun = async (run, file = "viewer_views.json", waitMs = 2500, warmMs = 15000) => {
    const meta = (await (await fetch(`/scenes/${sceneId}/eval/${file}`, { cache: "no-store" })).json()) as EvalViews;
    let saved = 0;
    // A hidden tab (the desktop app's browser pane in the background) runs no animation
    // frames, so the viewer drew only on mrCapture and Spark's LOD cut, which updates as
    // frames render and lands a frame later, lagged a pose behind: blender01 000000 to
    // 000040 showed only the shell or the coarsest splats (2 Nov). Waits render every 50 ms.
    const settle = async (ms: number) => {
      for (const end = performance.now() + ms; performance.now() < end; ) {
        w.mrCapture();
        await new Promise((r) => setTimeout(r, 50));
      }
    };
    if (meta.views.length) w.mrCamera(meta.views[0].c2w, meta.fov_deg);
    await settle(warmMs); // LOD trees and the shell stream in
    for (const v of meta.views) {
      w.mrCamera(v.c2w, meta.fov_deg);
      await settle(waitMs);
      w.mrCapture(); // renders now, so the drawing buffer holds this view
      const body = canvas.toDataURL("image/png");
      const q = `scene=${sceneId}&run=${encodeURIComponent(run)}&view=${v.name}`;
      if ((await fetch(`/__eval?${q}`, { method: "POST", body })).ok) saved++;
    }
    w.mrCamera();
    return { saved, size: [canvas.width, canvas.height] };
  };
}
