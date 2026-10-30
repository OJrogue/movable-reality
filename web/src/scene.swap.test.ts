// SceneView swaps an object between its scan and completed PLY. Spark's SplatMesh is
// replaced by a stub whose load the test resolves or rejects by hand.
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";

interface Stub extends THREE.Object3D {
  url: string;
  initialized: Promise<void>;
  resolve(): void;
  reject(): void;
}
const created: Stub[] = [];

vi.mock("@sparkjsdev/spark", async () => {
  const three = await import("three");
  class SplatMesh extends three.Object3D {
    url: string;
    initialized: Promise<void>;
    resolve!: () => void;
    reject!: () => void;
    constructor(opts: { url: string }) {
      super();
      this.url = opts.url;
      this.initialized = new Promise<void>((res, rej) => {
        this.resolve = res;
        this.reject = () => rej(new Error("404"));
      });
      this.initialized.catch(() => undefined);
      created.push(this as unknown as Stub);
    }
    dispose() {}
  }
  return { SplatMesh };
});

const { SceneView } = await import("./scene");
type Manifest = import("./scene").SceneManifest;

const SCAN = "objects/chair.ply";
const DONE = "objects/chair_completed.ply";

function manifest(dx: number): Manifest {
  return {
    version: 1,
    scene_id: "t",
    units: "m",
    world_from_recon: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    background_ply: "objects/background.ply",
    objects: [
      {
        id: "chair#1",
        label: "chair",
        ply: SCAN,
        completed_ply: DONE,
        bbox_min: [0, 0, 0],
        bbox_max: [1, 1, 1],
        transform: { translation: [dx, 0, 0], yaw_deg: 0, scale: 1 },
        visible: true,
        source: "scan",
        prompt: null,
      },
    ],
    history: [],
  };
}

const flush = () => new Promise((r) => setTimeout(r, 0));
const mesh = (ply: string) => created.find((m) => m.url === `/s/${ply}`)!;
const shown = (ply: string) => mesh(ply).parent!.visible;

describe("SceneView completed swap", () => {
  beforeEach(() => void (created.length = 0));

  it("keeps the scan drawn until the completed mesh loads, then hides it", async () => {
    const view = new SceneView("/s");
    view.apply(manifest(0));
    mesh(SCAN).resolve();
    await flush();
    view.apply(manifest(0.5));
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(DONE));
    expect([shown(SCAN), shown(DONE)]).toEqual([true, true]);
    mesh(DONE).resolve();
    await flush();
    expect([shown(SCAN), shown(DONE)]).toEqual([false, true]);
    view.apply(manifest(0)); // undo: instant, no new mesh
    expect([shown(SCAN), shown(DONE)]).toEqual([true, false]);
    expect(created.length).toBe(3); // background, scan, completed
  });

  it("stays on the scan when the edit is undone while the completed mesh loads", async () => {
    const view = new SceneView("/s");
    view.apply(manifest(0));
    mesh(SCAN).resolve();
    view.apply(manifest(0.5));
    view.apply(manifest(0));
    mesh(DONE).resolve();
    await flush();
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(SCAN));
    expect([shown(SCAN), shown(DONE)]).toEqual([true, false]);
  });

  it("falls back to the scan when the completed PLY fails, and undo still works", async () => {
    const view = new SceneView("/s");
    view.apply(manifest(0));
    mesh(SCAN).resolve();
    await flush();
    view.apply(manifest(0.5));
    mesh(DONE).reject();
    await flush();
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(SCAN));
    expect(shown(SCAN)).toBe(true);
    view.apply(manifest(0));
    view.apply(manifest(1));
    expect(shown(SCAN)).toBe(true);
    expect(created.length).toBe(3); // the broken PLY is not fetched again
  });

  it("loads the scan when an edited object's completed PLY is broken at start", async () => {
    const view = new SceneView("/s");
    view.apply(manifest(0.5));
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(DONE));
    mesh(DONE).reject();
    await flush();
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(SCAN));
    mesh(SCAN).resolve();
    await flush();
    expect(shown(SCAN)).toBe(true);
  });
});
