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

describe("SceneView loading", () => {
  beforeEach(() => void (created.length = 0));

  it("loads a hidden object when it is first shown, and only then", () => {
    const hidden = manifest(0);
    hidden.objects[0].visible = false;
    const view = new SceneView("/s");
    view.apply(hidden);
    expect(created.map((m) => m.url)).toEqual(["/s/objects/background.ply"]);
    expect(view.objectNode("chair#1")!.mesh).toBeUndefined();
    view.apply(manifest(0));
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(SCAN));
  });

  it("reports the url the mesh loads when each file starts and ends, loaded or failed", async () => {
    const events: string[] = [];
    const view = new SceneView("/s");
    view.loadObserver = {
      start: (url) => events.push(`start ${url}`),
      progress: () => undefined,
      end: (url) => events.push(`end ${url}`),
    };
    view.meshOptions = (ply) => (ply === SCAN ? { url: "/s/lod/chair.rad" } : {});
    view.apply(manifest(0));
    expect(events).toEqual(["start /s/objects/background.ply", "start /s/lod/chair.rad"]);
    created[0].reject();
    created[1].resolve();
    await flush();
    expect(events.slice(2).sort()).toEqual(["end /s/lod/chair.rad", "end /s/objects/background.ply"]);
  });
});

describe("SceneView completed swap", () => {
  beforeEach(() => void (created.length = 0));

  it("keeps the scan drawn until the completed mesh loads, then hides it", async () => {
    const view = new SceneView("/s");
    view.apply(manifest(0));
    mesh(SCAN).resolve();
    await flush();
    view.apply(manifest(0.5)); // a move alone keeps the scan
    expect(created.length).toBe(2);
    view.setObjectStyle("completed");
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(DONE));
    expect([shown(SCAN), shown(DONE)]).toEqual([true, true]);
    mesh(DONE).resolve();
    await flush();
    expect([shown(SCAN), shown(DONE)]).toEqual([false, true]);
    view.setObjectStyle("auto"); // back: instant, no new mesh
    expect([shown(SCAN), shown(DONE)]).toEqual([true, false]);
    expect(created.length).toBe(3); // background, scan, completed
  });

  it("stays on the scan when the style switches back while the completed mesh loads", async () => {
    const view = new SceneView("/s");
    view.apply(manifest(0));
    mesh(SCAN).resolve();
    view.setObjectStyle("completed");
    view.setObjectStyle("auto");
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
    view.setObjectStyle("completed");
    mesh(DONE).reject();
    await flush();
    expect(view.objectNode("chair#1")!.mesh).toBe(mesh(SCAN));
    expect(shown(SCAN)).toBe(true);
    view.setObjectStyle("auto");
    view.setObjectStyle("completed");
    expect(shown(SCAN)).toBe(true);
    expect(created.length).toBe(3); // the broken PLY is not fetched again
  });

  it("loads the scan when the completed PLY is broken at start", async () => {
    const view = new SceneView("/s");
    view.setObjectStyle("completed");
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
