// Lane core: switch how objects are drawn, to compare the scan with SAM 3D Objects' output.
//
// ?objects=auto|scan|completed|mesh picks the style at load and key V (or the label, top
// right) cycles it. auto, scan and completed are splat variants that SceneView swaps itself. mesh draws
// each object's vertex-coloured .glb (scripts/complete_objects.py --mesh) and hides that
// object's splats; an object without a mesh keeps its splats. The mesh sits under the
// object's pivot with the same offset and world_from_recon as the splats, so edits move it.
//
// The material is unlit (KHR_materials_unlit, so GLTFLoader builds a MeshBasicMaterial with
// vertex colours), like the splats, and opaque, so it is in the depth buffer before Spark
// blends any splat and the background behind it is hidden.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ViewerContext } from "../context";
import { matrixFromRows, OBJECT_STYLES, type ObjectStyle, type SceneManifest, type SceneView } from "../scene";

export function parseStyle(v: string | null): ObjectStyle {
  return (OBJECT_STYLES as readonly string[]).includes(v ?? "") ? (v as ObjectStyle) : "auto";
}

export function nextStyle(s: ObjectStyle): ObjectStyle {
  return OBJECT_STYLES[(OBJECT_STYLES.indexOf(s) + 1) % OBJECT_STYLES.length];
}

interface MeshEntry {
  url: string;
  group: THREE.Group; // offset(-centre) ─ recon ─ glTF scene, child of the pivot
  loaded: boolean;
}

export interface ObjectsHandle {
  style(): ObjectStyle;
  setStyle(s: ObjectStyle): void;
  /** Object ids whose mesh has loaded. */
  meshes(): string[];
}

export function installObjects(ctx: ViewerContext): ObjectsHandle {
  const loader = new GLTFLoader();
  const entries = new Map<string, MeshEntry>();
  let style = parseStyle(ctx.params.get("objects"));
  let view: SceneView | null = null;
  let manifest: SceneManifest | null = null;

  const label = document.createElement("button");
  label.type = "button";
  label.title =
    "How objects are drawn (key V): auto (scan, SAM 3D splats once moved), scan, SAM 3D splats, SAM 3D mesh";
  Object.assign(label.style, {
    position: "fixed",
    right: "12px",
    top: "160px",
    width: "260px",
    padding: "5px 10px",
    font: "12px system-ui, sans-serif",
    textAlign: "left",
    background: "rgba(0,0,0,0.55)",
    color: "#eee",
    border: "1px solid #666",
    borderRadius: "6px",
    cursor: "pointer",
    zIndex: "1000",
  });
  label.addEventListener("click", () => handle.setStyle(nextStyle(style)));
  document.body.appendChild(label);

  const refresh = () => {
    if (!view || !manifest) return;
    let withMesh = 0;
    for (const o of manifest.objects) {
      const e = entries.get(o.id);
      const showMesh = style === "mesh" && !!e?.loaded;
      if (e) e.group.visible = showMesh;
      if (e?.loaded) withMesh++;
      view.setSplatsShown(o.id, !showMesh);
    }
    const total = manifest.objects.filter((o) => o.mesh).length;
    const done = manifest.objects.filter((o) => o.completed_ply).length;
    const detail = {
      auto: "scan, SAM 3D once moved",
      scan: "scanned splats",
      completed: `${done} SAM 3D splats`,
      mesh: `${withMesh}/${total} SAM 3D meshes`,
    }[style];
    label.textContent = `Objects: ${style} (${detail}) · V`;
  };

  const load = (id: string, url: string, m: SceneManifest) => {
    const node = view?.objectNode(id);
    const obj = m.objects.find((o) => o.id === id);
    if (!node || !obj) return;
    const offset = new THREE.Group();
    offset.name = `mesh:${id}`;
    offset.position.copy(node.pivot.children[0].position); // -centre, as for the splats
    const recon = new THREE.Group();
    recon.matrixAutoUpdate = false;
    recon.matrix.copy(matrixFromRows(m.world_from_recon));
    offset.add(recon);
    offset.visible = false;
    node.pivot.add(offset);
    const entry: MeshEntry = { url, group: offset, loaded: false };
    entries.set(id, entry);
    loader
      .loadAsync(url)
      .then((gltf) => {
        if (entries.get(id) !== entry) return; // replaced while loading
        recon.add(gltf.scene);
        entry.loaded = true;
        let tris = 0;
        gltf.scene.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (mesh.isMesh) tris += (mesh.geometry.index?.count ?? 0) / 3;
        });
        console.info(`[objects] ${id}: ${url} (${tris} triangles)`);
        refresh();
      })
      .catch((err) => {
        console.warn(`[objects] ${id}: failed to load ${url}: ${String(err)}`);
        offset.removeFromParent(); // the entry stays, so the failed URL is not retried
      });
  };

  // Meshes load on first use of the mesh style, so the other styles cost nothing extra.
  const sync = () => {
    if (!view || !manifest || !ctx.sceneBase) return;
    if (style === "mesh") {
      for (const o of manifest.objects) {
        const url = o.mesh ? `${ctx.sceneBase}/${o.mesh}` : null;
        const e = entries.get(o.id);
        if (e && e.url !== url) {
          e.group.removeFromParent();
          dispose(e.group);
          entries.delete(o.id);
        }
        if (url && !entries.has(o.id)) load(o.id, url, manifest);
      }
    }
    refresh();
  };

  const handle: ObjectsHandle = {
    style: () => style,
    setStyle(s: ObjectStyle) {
      style = s;
      view?.setObjectStyle(s);
      console.info(`[objects] style ${s}`);
      sync();
    },
    meshes: () => [...entries].filter(([, e]) => e.loaded).map(([id]) => id),
  };

  ctx.onManifest((m, v) => {
    const first = view === null;
    view = v;
    manifest = m;
    if (first && style !== "auto") v.setObjectStyle(style);
    sync();
  });

  document.addEventListener("keydown", (e) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey || e.code !== "KeyV") return;
    const target = e.target as HTMLElement | null;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
    handle.setStyle(nextStyle(style));
  });

  refresh();
  (window as unknown as { mrObjects: ObjectsHandle }).mrObjects = handle;
  return handle;
}

function dispose(root: THREE.Object3D): void {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry.dispose();
    for (const mat of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) mat.dispose();
  });
}
