// Viewer entry. URL parameters:
//   ?scene=<id>            load <base>/<id>/scene.json (falls back to <id>/splat/scene.ply)
//   ?ply=<url>             quick look at a single PLY (OpenCV frame assumed)
//   ?base=<url>            where scene folders live (default /scenes)
//   ?xrScale=<f>           WebXR framebuffer scale, 0.25 to 1 (default 0.5, Spark's default)
//   perf and bench parameters (?mode, ?budget, ?hud, ?bench, ...): see perf/index.ts
// Owned by the renderer lane. Other lanes plug in through ViewerContext: export an
// install...(ctx) function from your folder and add one call in the "Lane installs" block.
import * as THREE from "three";
import { SparkControls, SparkRenderer, SparkXr, SplatMesh } from "@sparkjsdev/spark";
import { createViewerContext } from "./context";
import { installCoverage } from "./coverage";
import { installLod } from "./lod";
import { installPlanes } from "./planes";
import { matrixFromRows, OPENCV_TO_Y_UP, SceneView, type SceneManifest } from "./scene";
import { installPerf } from "./perf";
import { installUi } from "./ui";

const params = new URLSearchParams(location.search);
const sceneId = params.get("scene");
const plyUrl = params.get("ply");
const base = (params.get("base") ?? "/scenes").replace(/\/$/, "");
const xrScaleParam = Number(params.get("xrScale"));
const xrFramebufferScale = xrScaleParam >= 0.25 && xrScaleParam <= 1 ? xrScaleParam : 0.5;

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const spark = new SparkRenderer({ renderer });
scene.add(spark);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(0, 1.6, 0); // standing eye height; world floor is y = 0
const rig = new THREE.Group(); // moved by controls on desktop and in XR
rig.add(camera);
scene.add(rig);

const controls = new SparkControls({ canvas: renderer.domElement });
new SparkXr({
  renderer,
  mode: "vr",
  referenceSpaceType: "local-floor",
  frameBufferScaleFactor: xrFramebufferScale,
});

const ctx = createViewerContext({
  renderer,
  scene,
  spark,
  camera,
  rig,
  params,
  xrFramebufferScale,
});
const status = (msg: string) => {
  ctx.status = msg;
  console.info(`[viewer] ${msg}`);
};

// ---- Lane installs (one call per lane, in this order) ----
installPlanes(ctx); // lane bake
installLod(ctx); // lane lod
installCoverage(ctx); // lane coverage: gap overlay (?gaps=1, key G, XR squeeze)
installUi(ctx); // lane ui: desktop editing, command bar, onboarding (?onboarding=0)
installPerf(ctx); // lane renderer: HUD, frame log, budget controller, bench mode
// ----------------------------------------------------------

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// In XR, `time` is the XRFrame's predicted display time (three.js passes it through).
renderer.setAnimationLoop((time: number, frame?: XRFrame) => {
  const t0 = performance.now();
  if (ctx.controlsEnabled) controls.update(rig);
  ctx.runFrame(time, frame);
  renderer.render(scene, camera);
  ctx.runFrameEnd(time, frame, performance.now() - t0);
});

// Screenshot hook for the edit verifier (the server will call this over WebSocket).
(window as unknown as { mrCapture: () => string }).mrCapture = () => {
  renderer.render(scene, camera);
  return renderer.domElement.toDataURL("image/jpeg", 0.85);
};

function addSinglePly(url: string): void {
  const recon = new THREE.Group();
  recon.matrixAutoUpdate = false;
  recon.matrix.copy(matrixFromRows(OPENCV_TO_Y_UP));
  recon.add(new SplatMesh({ url, ...(ctx.meshOptions?.(url, null, emptyManifest(url)) ?? {}) }));
  scene.add(recon);
  status(`single PLY: ${url}`);
}

function emptyManifest(ply: string): SceneManifest {
  return {
    version: 1,
    scene_id: "ply",
    units: "m",
    world_from_recon: OPENCV_TO_Y_UP,
    background_ply: ply,
    objects: [],
    history: [],
  };
}

async function fetchManifest(url: string): Promise<{ text: string; manifest: SceneManifest } | null> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) return null;
  const text = await r.text();
  return { text, manifest: JSON.parse(text) as SceneManifest };
}

async function start(): Promise<void> {
  if (plyUrl) return addSinglePly(plyUrl);
  if (!sceneId) return void status("add ?scene=<id> or ?ply=<url> to the URL");

  const sceneBase = `${base}/${sceneId}`;
  ctx.sceneBase = sceneBase;
  const first = await fetchManifest(`${sceneBase}/scene.json`);
  if (!first) return addSinglePly(`${sceneBase}/splat/scene.ply`);

  const view = new SceneView(sceneBase);
  view.meshOptions = ctx.meshOptions;
  view.onLoadError = (ply, err) => status(`failed to load ${ply}: ${String(err)}`);
  scene.add(view.root);
  view.apply(first.manifest);
  ctx.runManifest(first.manifest, view);
  let last = first.text;
  status(`${sceneId}: ${first.manifest.objects.length} objects`);

  // Live updates from the agent. Replace with the server WebSocket at M3. The next poll
  // starts 1 s after the last one settles, so a slow response is never applied after a
  // newer one.
  const poll = async () => {
    const next = await fetchManifest(`${sceneBase}/scene.json`).catch(() => null);
    if (next && next.text !== last) {
      last = next.text;
      view.apply(next.manifest);
      ctx.runManifest(next.manifest, view);
      const visible = next.manifest.objects.filter((o) => o.visible).length;
      status(`${sceneId}: ${visible}/${next.manifest.objects.length} objects visible`);
    }
    setTimeout(poll, 1000);
  };
  setTimeout(poll, 1000);
}

start().catch((err) => status(`error: ${String(err)}`));
