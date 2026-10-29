// Lane bake, dev-only check: do Spark splats depth-test against opaque three.js meshes?
// Open http://localhost:5173/src/planes/depthcheck.html?case=<case> under `npm run dev`.
//
// The camera looks down -Z. A red, opaque quad (depthWrite on, as installPlanes sets it)
// sits at z = -2 and covers the left half of the view. Two splat walls cover the full width:
//   case=behind  green splats at z = -3, behind the quad: hidden on the left, visible right
//   case=front   blue splats at z = -1, in front of the quad: visible everywhere
//   case=both    both walls: left half blue (front wins), right half blue over green
//   case=crossing  one splat wall tilted through the quad's depth (top at z = -1, bottom at
//                  z = -3): on the left, top half blue (in front), bottom half red (hidden)
//   case=quad    quad only; case=splats  splats only (references)
// window.depthCheck resolves with pixel samples for the headless test.
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { makeOpaque } from "./planes";

const params = new URLSearchParams(location.search);
const which = params.get("case") ?? "both";
const hud = document.getElementById("hud")!;
const W = 640;
const H = 400;

const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const spark = new SparkRenderer({ renderer });
scene.add(spark);
const camera = new THREE.PerspectiveCamera(60, W / H, 0.05, 50);
camera.position.set(0, 0, 0);
camera.lookAt(0, 0, -1);

/** Splat wall at depth z, or tilted from z (top edge) to zBottom (bottom edge). */
function splatWall(z: number, color: THREE.Color, zBottom = z): SplatMesh {
  return new SplatMesh({
    constructSplats: (splats) => {
      const q = new THREE.Quaternion();
      const s = new THREE.Vector3(0.03, 0.03, 0.003);
      const far = -Math.min(z, zBottom);
      const halfW = Math.tan(THREE.MathUtils.degToRad(30)) * far * (W / H) * 1.1;
      const halfH = Math.tan(THREE.MathUtils.degToRad(30)) * far * 1.1;
      for (let x = -halfW; x <= halfW; x += 0.03) {
        for (let y = -halfH; y <= halfH; y += 0.03) {
          const t = (halfH - y) / (2 * halfH); // 0 at the top, 1 at the bottom
          splats.pushSplat(new THREE.Vector3(x, y, z + t * (zBottom - z)), s, q, 0.95, color);
        }
      }
    },
  });
}

// Opaque quad over the left half of the view at z = -2.
const halfW2 = Math.tan(THREE.MathUtils.degToRad(30)) * 2 * (W / H);
const quad = new THREE.Mesh(
  new THREE.PlaneGeometry(halfW2 * 1.2, 4),
  new THREE.MeshBasicMaterial({ color: 0xff0000 }),
);
quad.position.set(-halfW2 * 0.6, 0, -2);
makeOpaque(quad);

const meshes: SplatMesh[] = [];
if (which !== "splats") scene.add(quad);
if (which === "behind" || which === "both" || which === "splats") {
  meshes.push(splatWall(-3, new THREE.Color(0x00ff00)));
}
if (which === "front" || which === "both" || which === "splats") {
  meshes.push(splatWall(-1, new THREE.Color(0x0000ff)));
}
if (which === "crossing") meshes.push(splatWall(-1, new THREE.Color(0x0000ff), -3));
if (which === "quad") meshes.length = 0;
for (const m of meshes) scene.add(m);

type Rgb = [number, number, number];
function sample(x: number, y: number): Rgb {
  const gl = renderer.getContext();
  const px = new Uint8Array(4);
  gl.readPixels(x, H - 1 - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  return [px[0], px[1], px[2]];
}

async function run() {
  await Promise.all(meshes.map((m) => m.initialized));
  for (let i = 0; i < 30; i++) {
    renderer.render(scene, camera);
    await new Promise((r) => requestAnimationFrame(r));
  }
  renderer.render(scene, camera);
  const mat = spark.material;
  const result = {
    case: which,
    left: sample(W * 0.25, H * 0.5),
    right: sample(W * 0.75, H * 0.5),
    leftTop: sample(W * 0.25, H * 0.2),
    leftBottom: sample(W * 0.25, H * 0.8),
    spark: {
      transparent: mat.transparent,
      depthTest: mat.depthTest,
      depthWrite: mat.depthWrite,
      renderOrder: spark.renderOrder,
    },
    quad: { depthWrite: quad.material.depthWrite, transparent: quad.material.transparent },
  };
  hud.textContent =
    `case=${which}\nleft ${result.left}  right ${result.right}\n` +
    `left top ${result.leftTop}  left bottom ${result.leftBottom}`;
  return result;
}

(window as unknown as { depthCheck: Promise<unknown> }).depthCheck = run();
