import { describe, expect, it } from "vitest";
import * as THREE from "three";
import type { BakedPlane, SceneManifest } from "../scene";
import {
  diffPlanes,
  makeOpaque,
  PLANE_RENDER_ORDER,
  planeKey,
  planeUrl,
  reconGroup,
  staleBackgroundLod,
} from "./planes";
import { matrixFromRows } from "../scene";

const floor: BakedPlane = {
  id: "floor",
  label: "floor",
  mesh: "planes/floor.glb",
  removed_splats: 100,
  corners: [
    [0, 0, 0],
    [4, 0, 0],
    [4, 0, -5],
    [0, 0, -5],
  ],
  texel_m: 0.005,
  texture_px: [800, 1000],
};
const ceiling: BakedPlane = { ...floor, id: "ceiling", label: "ceiling", mesh: "planes/ceiling.glb" };

function manifest(planes?: BakedPlane[]): SceneManifest {
  return {
    version: 1,
    scene_id: "t",
    units: "m",
    world_from_recon: [
      [1, 0, 0, 0],
      [0, -1, 0, 0],
      [0, 0, -1, 0],
      [0, 0, 0, 1],
    ],
    background_ply: "objects/background_baked.ply",
    objects: [],
    planes,
    history: [],
  };
}

describe("diffPlanes", () => {
  it("loads every plane on the first manifest and nothing when unchanged", () => {
    const first = diffPlanes([], manifest([floor, ceiling]));
    expect(first.add.map((p) => p.id)).toEqual(["floor", "ceiling"]);
    expect(first.remove).toEqual([]);
    const keys = [planeKey(floor), planeKey(ceiling)];
    expect(diffPlanes(keys, manifest([floor, ceiling]))).toEqual({ add: [], remove: [] });
  });

  it("swaps a re-baked plane and drops removed ones", () => {
    const rebaked = { ...floor, texel_m: 0.004 };
    const d = diffPlanes([planeKey(floor), planeKey(ceiling)], manifest([rebaked]));
    expect(d.add).toEqual([rebaked]);
    expect(d.remove.sort()).toEqual([planeKey(floor), planeKey(ceiling)].sort());
  });

  it("treats a re-bake that only changes the alpha mask as a new plane", () => {
    const masked: BakedPlane = { ...floor, transparent_fraction: 0.07 };
    expect(planeKey(masked)).not.toBe(planeKey(floor));
    expect(planeKey({ ...floor, transparent_fraction: 0 })).toBe(planeKey(floor));
  });

  it("treats a manifest without the planes section as no planes", () => {
    expect(diffPlanes([], manifest(undefined))).toEqual({ add: [], remove: [] });
    expect(diffPlanes([planeKey(floor)], manifest(undefined)).remove).toEqual([planeKey(floor)]);
  });
});

describe("planeUrl", () => {
  it("is relative to the scene folder and changes when the plane is re-baked", () => {
    const a = planeUrl("/scenes/demo", floor);
    expect(a.startsWith("/scenes/demo/planes/floor.glb?v=")).toBe(true);
    expect(planeUrl("/scenes/demo", { ...floor, texel_m: 0.004 })).not.toEqual(a);
  });
});

describe("makeOpaque", () => {
  it("turns every mesh into a depth-writing occluder drawn first", () => {
    const root = new THREE.Group();
    const map = new THREE.Texture();
    const mat = new THREE.MeshBasicMaterial({ map, transparent: true, opacity: 0.5, depthWrite: false });
    root.add(new THREE.Mesh(new THREE.PlaneGeometry(), mat));
    root.add(new THREE.Object3D());
    expect(makeOpaque(root, 16)).toBe(1);
    const mesh = root.children[0] as THREE.Mesh;
    expect(mat.transparent).toBe(false);
    expect(mat.opacity).toBe(1);
    expect(mat.depthTest).toBe(true);
    expect(mat.depthWrite).toBe(true);
    expect(mat.blending).toBe(THREE.NoBlending);
    expect(mesh.renderOrder).toBe(PLANE_RENDER_ORDER);
    expect(map.anisotropy).toBe(8);
    expect(mat.alphaTest).toBe(0);
  });

  it("keeps the alpha test of a masked plane (glTF alphaMode MASK)", () => {
    const root = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ alphaTest: 0.5 });
    root.add(new THREE.Mesh(new THREE.PlaneGeometry(), mat));
    makeOpaque(root);
    expect(mat.alphaTest).toBe(0.5);
    expect(mat.transparent).toBe(false);
    expect(mat.depthWrite).toBe(true);
  });
});

describe("staleBackgroundLod", () => {
  const bake = {
    source_background_ply: "objects/background.ply",
    background_ply: "objects/background_baked.ply",
    ids: "ids/gaussian_ids.npy",
    ids_ply: "splat/scene.ply",
    removed_splats: 100,
  };
  it("flags a background tree built from the full background", () => {
    const m = { ...manifest([floor]), plane_bake: bake, background_lod: "lod/background.rad" };
    expect(staleBackgroundLod(m)).toBe(true);
  });
  it("accepts a tree built from the reduced background, or no tree", () => {
    const m = { ...manifest([floor]), plane_bake: bake, background_lod: "lod/background_baked.rad" };
    expect(staleBackgroundLod(m)).toBe(false);
    expect(staleBackgroundLod({ ...m, background_lod: null })).toBe(false);
    expect(staleBackgroundLod(manifest([floor]))).toBe(false);
  });
});

describe("reconGroup", () => {
  it("applies world_from_recon to quads in the recon frame, like the splats", () => {
    const g = reconGroup(matrixFromRows(manifest().world_from_recon));
    const child = new THREE.Object3D();
    child.position.set(1, 2, 3); // recon frame (OpenCV)
    g.add(child);
    g.updateMatrixWorld(true);
    const p = new THREE.Vector3().setFromMatrixPosition(child.matrixWorld);
    expect(p.toArray()).toEqual([1, -2, -3]);
    expect(g.matrixAutoUpdate).toBe(false);
  });
});
