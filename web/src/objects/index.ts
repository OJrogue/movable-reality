// Lane core: switch how objects are drawn, to compare the scan with SAM 3D Objects' output.
//
// ?objects=auto|scan|completed|mesh|meshonly picks the style at load and key V (or the label, top
// right) cycles it. auto, scan and completed are splat variants that SceneView swaps itself. mesh draws
// each object's vertex-coloured .glb (scripts/complete_objects.py --mesh) and hides that
// object's splats; an object without a mesh keeps its splats. The mesh sits under the
// object's pivot with the same offset and world_from_recon as the splats, so edits move it.
//
// The material is unlit (KHR_materials_unlit, so GLTFLoader builds a MeshBasicMaterial with
// vertex colours), like the splats, and opaque, so it is in the depth buffer before Spark
// blends any splat and the background behind it is hidden.
import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { ViewerContext } from "../context";
import { matrixFromRows, OBJECT_STYLES, type ObjectStyle, type SceneManifest, type SceneView } from "../scene";

export function parseStyle(v: string | null): ObjectStyle {
  return (OBJECT_STYLES as readonly string[]).includes(v ?? "") ? (v as ObjectStyle) : "auto";
}

export function nextStyle(s: ObjectStyle): ObjectStyle {
  return OBJECT_STYLES[(OBJECT_STYLES.indexOf(s) + 1) % OBJECT_STYLES.length];
}

/** room/shell.json as scripts/room_shell.py writes it (only the fields read here). */
export interface ShellReport {
  outline_xz?: number[][];
  surfaces?: { glb: string; kind: string; seen_share: number }[];
}

/** Shell ceilings the capture never saw (their texture is the upper wall colour). */
export function unseenCeilings(r: ShellReport, minSeen = 0.02): string[] {
  return (r.surfaces ?? []).filter((s) => s.kind === "ceiling" && s.seen_share < minSeen).map((s) => s.glb);
}

/** The backdrop the splat styles draw behind the scan: the shell walls and any unseen
 * ceiling. A walk that never looks up leaves no splats above about 2 m, so the room was
 * open to black there, with the background's stray splats hanging in it (blender01, 1 Nov). */
export function backdropFiles(r: ShellReport): string[] {
  const walls = (r.surfaces ?? []).filter((s) => s.kind === "wall").map((s) => s.glb);
  return [...walls, ...unseenCeilings(r)];
}

/** World-space scale about the outline's centre that moves every wall `push` metres out,
 * so the wall splats stay in front of the backdrop and hide it wherever the scan has them. */
export function backdropScale(outline: number[][], push = 0.05): { cx: number; cz: number; sx: number; sz: number } | null {
  if (outline.length < 3) return null;
  const xs = outline.map((p) => p[0]);
  const zs = outline.map((p) => p[1]);
  const [x0, x1, z0, z1] = [Math.min(...xs), Math.max(...xs), Math.min(...zs), Math.max(...zs)];
  const hx = (x1 - x0) / 2;
  const hz = (z1 - z0) / 2;
  if (hx <= 0 || hz <= 0) return null;
  return { cx: x0 + hx, cz: z0 + hz, sx: 1 + push / hx, sz: 1 + push / hz };
}

/** Room files for the meshonly style: the shell quads, else the fused room mesh, else none. */
export function roomFiles(m: Pick<SceneManifest, "room_planes" | "room_mesh">): string[] {
  return m.room_planes?.length ? m.room_planes : m.room_mesh ? [m.room_mesh] : [];
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
    "How objects are drawn (key V): auto (scan, SAM 3D splats once moved), scan, SAM 3D splats, " +
    "SAM 3D meshes, meshes only (room mesh too, no splats)";
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

  // Room (meshonly): the shell quads (room_planes, scripts/room_shell.py), or the fused
  // room mesh when there is no shell. Under view.root in a world_from_recon group, like
  // the planes. `url` is the list of files joined, so a changed list reloads.
  let room: { url: string; group: THREE.Group; loaded: boolean } | null = null;
  const hiddenSplats = new Set<THREE.Object3D>();
  const roomUrls = (m: SceneManifest): string[] => roomFiles(m).map((f) => `${ctx.sceneBase}/${f}`);

  const loadRoom = (urls: string[], m: SceneManifest) => {
    if (!view) return;
    const group = new THREE.Group();
    group.name = "room-mesh";
    group.matrixAutoUpdate = false;
    group.matrix.copy(matrixFromRows(m.world_from_recon));
    group.visible = false;
    view.root.add(group);
    const entry = { url: urls.join(" "), group, loaded: false };
    room = entry;
    for (const url of urls) {
      loader
        .loadAsync(url)
        .then((gltf) => {
          if (room !== entry) return;
          group.add(gltf.scene);
          entry.loaded = true; // drawn once any part has arrived
          refresh();
        })
        .catch((err) => console.warn(`[objects] room: failed to load ${url}: ${String(err)}`));
    }
    console.info(`[objects] room: ${urls.length} file(s)`);
  };

  // meshonly hides every splat mesh in the scene (background, objects, coverage repairs),
  // each frame, since SceneView may create a variant at any edit. Leaving the style shows
  // exactly the ones hidden here again.
  const hideSplats = () => {
    ctx.scene.traverse((o) => {
      if (o instanceof SplatMesh && o.visible) {
        o.visible = false;
        hiddenSplats.add(o);
      }
    });
  };
  const restoreSplats = () => {
    for (const o of hiddenSplats) o.visible = true;
    hiddenSplats.clear();
  };
  ctx.onFrame(() => {
    if (style === "meshonly") hideSplats();
  });

  const refresh = () => {
    if (!view || !manifest) return;
    const meshes = style === "mesh" || style === "meshonly";
    let withMesh = 0;
    for (const o of manifest.objects) {
      const e = entries.get(o.id);
      const showMesh = meshes && !!e?.loaded;
      if (e) e.group.visible = showMesh;
      if (e?.loaded) withMesh++;
      view.setSplatsShown(o.id, !showMesh);
    }
    if (room) room.group.visible = style === "meshonly" && room.loaded;
    if (cap) cap.group.visible = style !== "meshonly"; // meshonly draws the whole shell
    const planes = view.root.getObjectByName("baked-planes");
    if (planes) planes.visible = style !== "meshonly"; // the room mesh has its own floor
    if (style === "meshonly") hideSplats();
    else restoreSplats();
    const total = manifest.objects.filter((o) => o.mesh).length;
    const done = manifest.objects.filter((o) => o.completed_ply).length;
    const detail = {
      auto: "scan, SAM 3D once moved",
      scan: "scanned splats",
      completed: `${done} SAM 3D splats`,
      mesh: `${withMesh}/${total} SAM 3D meshes`,
      meshonly: `${withMesh}/${total} meshes, room ${room?.loaded ? (manifest.room_planes?.length ? "shell" : "mesh") : roomUrls(manifest).length ? "loading" : "missing"}, no splats`,
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

  // Backdrop for the splat styles (backdropFiles), pushed out behind the wall splats
  // (backdropScale). `key` is the shell's file list, so a rebuilt shell reloads it.
  let cap: { key: string; group: THREE.Group } | null = null;
  const loadCap = (m: SceneManifest) => {
    const key = (m.room_planes ?? []).join(" ");
    if (!view || cap?.key === key) return;
    if (cap) {
      cap.group.removeFromParent();
      dispose(cap.group);
    }
    const group = new THREE.Group(); // world frame: centre, scale, back
    group.name = "room-backdrop";
    const back = new THREE.Group();
    const recon = new THREE.Group();
    recon.matrixAutoUpdate = false;
    recon.matrix.copy(matrixFromRows(m.world_from_recon));
    group.add(back);
    back.add(recon);
    view.root.add(group);
    const entry = { key, group };
    cap = entry;
    if (!key) return;
    fetch(`${ctx.sceneBase}/room/shell.json`, { cache: "no-store" })
      .then((r) => (r.ok ? (r.json() as Promise<ShellReport>) : {}))
      .then((r: ShellReport) => {
        const sc = backdropScale(r.outline_xz ?? []);
        if (!sc) return [];
        group.position.set(sc.cx, 0, sc.cz);
        group.scale.set(sc.sx, 1, sc.sz);
        back.position.set(-sc.cx, 0, -sc.cz);
        return Promise.all(backdropFiles(r).map((f) => loader.loadAsync(`${ctx.sceneBase}/${f}`)));
      })
      .then((gltfs) => {
        if (cap !== entry) return;
        for (const g of gltfs) recon.add(g.scene);
        if (gltfs.length) console.info(`[objects] room backdrop: ${gltfs.length} shell surface(s)`);
      })
      .catch((err) => console.warn(`[objects] room backdrop: ${String(err)}`));
  };

  // Meshes load on first use of the mesh style, so the other styles cost nothing extra.
  const sync = () => {
    if (!view || !manifest || !ctx.sceneBase) return;
    loadCap(manifest);
    const urls = roomUrls(manifest);
    if (style === "meshonly" && urls.length) {
      if (room && room.url !== urls.join(" ")) {
        room.group.removeFromParent();
        dispose(room.group);
        room = null;
      }
      if (!room) loadRoom(urls, manifest);
    }
    if (style === "mesh" || style === "meshonly") {
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
    for (const mat of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      (mat as THREE.MeshBasicMaterial).map?.dispose(); // the shell quads are textured
      mat.dispose();
    }
  });
}
