// Viewer entry. URL parameters:
//   ?scene=<id>            load <base>/<id>/scene.json (falls back to <id>/splat/scene.ply)
//   ?ply=<url>             quick look at a single PLY (OpenCV frame assumed)
//   ?base=<url>            where scene folders live (default /scenes)
import * as THREE from "three";
import { SparkControls, SparkRenderer, SparkXr, SplatMesh } from "@sparkjsdev/spark";
import { matrixFromRows, OPENCV_TO_Y_UP, SceneView, type SceneManifest } from "./scene";

const params = new URLSearchParams(location.search);
const sceneId = params.get("scene");
const plyUrl = params.get("ply");
const base = (params.get("base") ?? "/scenes").replace(/\/$/, "");
const hud = document.getElementById("hud")!;
const status = (msg: string) => (hud.textContent = msg);

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
scene.add(new SparkRenderer({ renderer }));

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 200);
camera.position.set(0, 1.6, 0); // standing eye height; world floor is y = 0
const rig = new THREE.Group(); // moved by controls on desktop and in XR
rig.add(camera);
scene.add(rig);

const controls = new SparkControls({ canvas: renderer.domElement });
new SparkXr({ renderer, mode: "vr", referenceSpaceType: "local-floor" });

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.setAnimationLoop(() => {
  controls.update(rig);
  renderer.render(scene, camera);
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
  recon.add(new SplatMesh({ url }));
  scene.add(recon);
  status(`single PLY: ${url}`);
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
  const first = await fetchManifest(`${sceneBase}/scene.json`);
  if (!first) return addSinglePly(`${sceneBase}/splat/scene.ply`);

  const view = new SceneView(sceneBase);
  scene.add(view.root);
  view.apply(first.manifest);
  let last = first.text;
  status(`${sceneId}: ${first.manifest.objects.length} objects`);

  // Live updates from the agent. Replace with the server WebSocket at M3.
  setInterval(async () => {
    const next = await fetchManifest(`${sceneBase}/scene.json`).catch(() => null);
    if (next && next.text !== last) {
      last = next.text;
      view.apply(next.manifest);
      const visible = next.manifest.objects.filter((o) => o.visible).length;
      status(`${sceneId}: ${visible}/${next.manifest.objects.length} objects visible`);
    }
  }, 1000);
}

start().catch((err) => status(`error: ${String(err)}`));
