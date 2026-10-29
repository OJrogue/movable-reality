// Lane coverage: gap overlay in the viewer (desktop and XR).
//
// Every open gap in scene.json `coverage.gaps` is a translucent red sphere with a rim glow
// (size = gap radius, intensity = gap score; orange once a recapture is requested). Each gap's best
// next view is a standing marker on the floor (ring + post up to eye height) with an arrow
// from the eye towards the gap: where to stand and point the phone for a 10 s top-up.
//
// Toggle: ?gaps=1 turns it on at load; key G on desktop; the squeeze (grip) button of
// either XR controller (xr-standard button 1, not used by Spark's or the perf lane's
// controls). URL options: ?gapViews=<n> next views per gap (default 1).
// Markers are plain three.js meshes in world coordinates (scene.json frame), drawn after
// the opaque pass with depth test on and depth write off, so they blend over the splats.
import * as THREE from "three";
import type { ViewerContext } from "../context";
import {
  EdgeTrigger,
  VIEW_COLOR,
  gapsParam,
  overlayKey,
  overlayModel,
  type GapMarker,
  type OverlayModel,
  type ViewMarker,
} from "./overlay";

export interface CoverageHandle {
  readonly group: THREE.Group;
  visible(): boolean;
  setVisible(on: boolean): void;
  toggle(): void;
  model(): OverlayModel;
}

const XR_TOGGLE_BUTTON = 1; // squeeze / grip

// Rim glow: nearly clear in the middle, strongest at the silhouette, so a gap reads as a
// glowing shell and the splats inside stay visible. `intensity` is the peak alpha.
const RIM_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vView;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vNormal = normalize(normalMatrix * normal);
  vView = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;
const RIM_FRAGMENT = /* glsl */ `
uniform vec3 color;
uniform float intensity;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  float rim = 1.0 - abs(dot(normalize(vNormal), normalize(vView)));
  gl_FragColor = vec4(color, intensity * (0.12 + 0.88 * rim * rim));
}`;

function rimMaterial(color: number, intensity: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { color: { value: new THREE.Color(color) }, intensity: { value: intensity } },
    vertexShader: RIM_VERTEX,
    fragmentShader: RIM_FRAGMENT,
    transparent: true,
    depthWrite: false,
  });
}

function gapMesh(g: GapMarker, geo: THREE.SphereGeometry): THREE.Mesh {
  const mesh = new THREE.Mesh(geo, rimMaterial(g.color, g.opacity));
  mesh.name = `gap:${g.id}`;
  mesh.position.set(...g.center);
  mesh.scale.setScalar(g.radius);
  mesh.renderOrder = 10;
  return mesh;
}

function viewNode(v: ViewMarker, mat: THREE.MeshBasicMaterial, ring: THREE.RingGeometry): THREE.Group {
  const node = new THREE.Group();
  node.name = `nbv:${v.gapId}`;
  const r = new THREE.Mesh(ring, mat);
  r.rotation.x = -Math.PI / 2;
  r.position.set(v.floor[0], 0.01, v.floor[2]);
  node.add(r);
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, v.eye[1], 8), mat);
  post.position.set(v.floor[0], v.eye[1] / 2, v.floor[2]);
  node.add(post);
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(...v.dir),
    new THREE.Vector3(...v.eye),
    v.length,
    VIEW_COLOR,
    0.35 * v.length,
    0.18 * v.length,
  );
  node.add(arrow);
  node.renderOrder = 11;
  return node;
}

function disposeTree(obj: THREE.Object3D, keep: Set<THREE.BufferGeometry | THREE.Material>): void {
  obj.traverse((o) => {
    const m = o as THREE.Mesh;
    // ArrowHelper's line and cone geometries are module-level and shared by every arrow.
    const sharedGeometry = o.parent instanceof THREE.ArrowHelper;
    if (m.geometry && !sharedGeometry && !keep.has(m.geometry)) m.geometry.dispose();
    const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
    for (const mat of mats) if (!keep.has(mat)) mat.dispose();
  });
}

export function installCoverage(ctx: ViewerContext): CoverageHandle {
  const group = new THREE.Group();
  group.name = "coverage-overlay";
  group.visible = gapsParam(ctx.params);
  ctx.scene.add(group); // world frame: gaps and views are in scene.json coordinates
  const perGap = Math.max(1, Math.floor(Number(ctx.params.get("gapViews")) || 1));
  const sphere = new THREE.SphereGeometry(1, 24, 16);
  const ring = new THREE.RingGeometry(0.18, 0.24, 32);
  const viewMat = new THREE.MeshBasicMaterial({
    color: VIEW_COLOR,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const shared = new Set<THREE.BufferGeometry | THREE.Material>([sphere, ring, viewMat]);
  let model: OverlayModel = overlayModel(null);
  let key = "";

  const rebuild = () => {
    for (const child of [...group.children]) {
      group.remove(child);
      disposeTree(child, shared);
    }
    for (const g of model.gaps) group.add(gapMesh(g, sphere));
    for (const v of model.views) group.add(viewNode(v, viewMat, ring));
  };

  const report = () => {
    if (group.visible) console.info(`[coverage] ${model.summary}`);
  };

  ctx.onManifest((m) => {
    const next = overlayModel(m, perGap);
    const nextKey = overlayKey(next);
    model = next;
    if (nextKey !== key) {
      key = nextKey;
      rebuild();
      report();
    }
  });

  const handle: CoverageHandle = {
    group,
    visible: () => group.visible,
    setVisible(on) {
      group.visible = on;
      console.info(`[coverage] overlay ${on ? "on" : "off"}`);
      report();
    },
    toggle() {
      handle.setVisible(!group.visible);
    },
    model: () => model,
  };

  document.addEventListener("keydown", (e) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.code !== "KeyG") return;
    const target = e.target as HTMLElement | null;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
    handle.toggle();
  });

  const squeeze = new EdgeTrigger<XRInputSource>();
  ctx.onFrame(() => {
    const session = ctx.renderer.xr.getSession();
    for (const source of session?.inputSources ?? []) {
      const pad = source.gamepad;
      if (!pad || pad.mapping !== "xr-standard") continue;
      if (squeeze.pressed(source, pad.buttons[XR_TOGGLE_BUTTON]?.pressed ?? false)) handle.toggle();
    }
  });

  (window as unknown as { mrCoverage: CoverageHandle }).mrCoverage = handle;
  return handle;
}
