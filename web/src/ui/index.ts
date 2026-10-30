// Lane ui: desktop editing (U1), command bar with hold-to-talk (U2) and the onboarding panel
// (U4). main.ts calls installUi(ctx) once. Edits go to the scene server when GET /api/health
// answers; otherwise they run through the TypeScript mirror of Scene.apply_edit (edits.ts)
// and stay local to the page (always with ?server=0). All of it is DOM and desktop input, so WebXR is unaffected.
import * as THREE from "three";
import type { ViewerContext } from "../context";
import type { SceneManifest, SceneView, Vec3 } from "../scene";
import { SceneServer, ServerRefused, type CommandResult, type ServerMessage } from "./api";
import { CommandBar } from "./commandBar";
import { applyEdit, EditError, type EditRequest } from "./edits";
import { Editor } from "./editor";
import { installOnboarding } from "./onboarding";
import { UI_CSS } from "./style";

/** main.ts passes the full context, which can also publish a manifest to every lane. */
export type UiContext = ViewerContext & {
  runManifest(manifest: SceneManifest, view: SceneView): void;
};

const RETRY_MS = 5000;
const OFFLINE_TEXT = "offline: edits are local";

export interface UiHandle {
  editor: Editor;
  bar: CommandBar;
  online(): boolean;
  edit(req: EditRequest): Promise<void>;
  manifest(): SceneManifest | null;
}

export function installUi(ctx: UiContext): UiHandle {
  const style = document.createElement("style");
  style.textContent = UI_CSS;
  document.head.appendChild(style);

  const sceneId = ctx.params.get("scene");
  // ?server=0 keeps every edit local even when a server runs (tests, a hosted read-only demo).
  const server = sceneId && ctx.params.get("server") !== "0" ? new SceneServer(sceneId) : null;
  let online = false;
  let socket: WebSocket | null = null;
  let queue: Promise<void> = Promise.resolve();
  let localEdits = 0; // edits applied by the local mirror since the last server scene

  // History is append-only, so a manifest with a shorter history is older than the one on
  // screen: a POST reply, a WebSocket push and main.ts's 1 s poll can arrive out of order.
  let latest: SceneManifest | null = null;
  const publish = (m: SceneManifest) => {
    const view = ctx.view;
    if (!view || (latest && m.history.length < latest.history.length)) return;
    latest = m;
    view.apply(m);
    ctx.runManifest(m, view);
  };
  ctx.onManifest((m) => {
    if (!latest || m === latest) return;
    if (m.history.length >= latest.history.length) {
      latest = m;
      return;
    }
    // The poll in main.ts applied a stale file. Put the newer manifest back before the next
    // frame, after the other lanes' callbacks for this one have run.
    const keep = latest;
    latest = null;
    queueMicrotask(() => publish(keep));
  });

  const frameWaiters: { left: number; resolve: () => void }[] = [];
  ctx.onFrameEnd(() => {
    for (let i = frameWaiters.length - 1; i >= 0; i--) {
      if (--frameWaiters[i].left <= 0) frameWaiters.splice(i, 1)[0].resolve();
    }
  });
  const frames = (n: number) => new Promise<void>((resolve) => frameWaiters.push({ left: n, resolve }));

  const bar = new CommandBar({
    server,
    online: () => online,
    applyResult: (r: CommandResult) => {
      if (r.scene) publish(r.scene);
    },
    xr: () => ctx.renderer.xr.isPresenting,
  });

  const setOnline = (on: boolean) => {
    const was = online;
    online = on;
    bar.setMode(on, on ? "server" : sceneId ? OFFLINE_TEXT : "no scene");
    if (on && !was) {
      latest = null; // the server's scene is the truth from here on
      bar.line("info", "connected to the scene server");
      if (localEdits > 0) bar.line("info", `${localEdits} local edits replaced by the server's scene`);
      localEdits = 0;
    }
    if (!on && was) bar.line("info", "lost the scene server; edits are local until it is back");
  };

  // Edits run one at a time, in order, so a fast Q Q Q never reorders on the server.
  const runEdit = async (req: EditRequest): Promise<void> => {
    const m = ctx.manifest;
    if (!m || !ctx.view) return;
    if (server && online) {
      try {
        publish(await server.edit(req));
        return;
      } catch (err) {
        if (err instanceof ServerRefused) {
          bar.line("error", `edit refused: ${err.message}`);
          publish(m); // snap a dragged object back
          return;
        }
        console.warn("[ui] server edit failed, going offline", err);
        goOffline();
      }
    }
    try {
      const next = applyEdit(m, req);
      if (req.op === "undo" && next.history.length === m.history.length) bar.line("info", "nothing to undo");
      publish(next);
      localEdits++;
    } catch (err) {
      bar.line("error", err instanceof EditError ? err.message : `edit failed: ${String(err)}`);
      publish(m);
    }
  };
  const edit = (req: EditRequest) => (queue = queue.then(() => runEdit(req)));

  const editor = new Editor(ctx, { edit, notify: (t) => bar.line("info", t) });

  // ---- server connection: health check, then the WebSocket; retry while it is down.
  const onMessage = (msg: ServerMessage) => {
    if (msg.type === "scene") publish(msg.scene);
    else if (msg.type === "agent") bar.agent(msg.stage, msg.text);
    else if (msg.type === "capture_request") {
      const ws = socket;
      void capture(ctx, frames, msg.camera).then((jpeg) => {
        // null answers at once, so the server need not wait for its timeout.
        if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "capture", id: msg.id, jpeg }));
      });
    }
  };

  const goOffline = () => {
    setOnline(false);
    socket?.close(); // its close handler schedules the next health check
  };

  // GET /api/health decides online or offline. The WebSocket is reopened after it closes;
  // the 1 s poll in main.ts still picks up scene.json changes while it is down.
  const connect = async () => {
    if (!server) return;
    if (await server.health()) {
      setOnline(true);
      socket = server.connect(onMessage, () => {
        socket = null;
        setTimeout(connect, RETRY_MS);
      });
    } else {
      setOnline(false);
      setTimeout(connect, RETRY_MS);
    }
  };
  setOnline(false);
  void connect();

  installOnboarding({
    params: ctx.params,
    fill: (t) => bar.fill(t),
    // The coverage lane toggles its overlay on key G; main.ts keeps no handle to it.
    toggleGaps: () => document.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyG", key: "g" })),
    toggleGenerated: (() => {
      let on = false;
      return () => {
        on = !on;
        const n = ctx.manifest?.coverage?.generated_splats ?? 0;
        bar.line("info", n > 0 ? `show generated: ${on ? "on" : "off"} (viewer support lands with M4)` : "this scene has no generated splats yet");
        return on && n > 0;
      };
    })(),
  });

  const handle: UiHandle = { editor, bar, online: () => online, edit, manifest: () => ctx.manifest };
  (window as unknown as { mrUi?: UiHandle }).mrUi = handle; // for console checks and tests
  return handle;
}

const _eye = new THREE.Vector3();
const SETTLE_FRAMES = 3;
const SETTLE_TIMEOUT_MS = 1500; // a hidden tab runs no frames

/**
 * Screenshot for the verifier. With a camera request, the rig and camera move to `eye`
 * looking at `target` for a few frames, so Spark's sort and the LOD cut catch up with the
 * new viewpoint, and then go back. In XR the headset renders into its own framebuffer and
 * the canvas holds no image, so the answer is null.
 */
async function capture(
  ctx: ViewerContext,
  frames: (n: number) => Promise<void>,
  camera?: { eye: Vec3; target: Vec3 },
): Promise<string | null> {
  const fn = (window as unknown as { mrCapture?: () => string }).mrCapture;
  if (!fn || ctx.renderer.xr.isPresenting) return null;
  if (!camera) return fn();
  const { rig, camera: cam } = ctx;
  const saved = {
    rigPos: rig.position.clone(),
    rigQuat: rig.quaternion.clone(),
    rigScale: rig.scale.clone(),
    camPos: cam.position.clone(),
    camQuat: cam.quaternion.clone(),
    controls: ctx.controlsEnabled,
  };
  ctx.controlsEnabled = false; // SparkControls must not move the rig meanwhile
  try {
    rig.position.set(0, 0, 0);
    rig.quaternion.identity();
    rig.scale.set(1, 1, 1);
    cam.position.copy(_eye.fromArray(camera.eye));
    cam.lookAt(camera.target[0], camera.target[1], camera.target[2]);
    rig.updateMatrixWorld(true);
    await Promise.race([frames(SETTLE_FRAMES), new Promise((r) => setTimeout(r, SETTLE_TIMEOUT_MS))]);
    return ctx.renderer.xr.isPresenting ? null : fn();
  } finally {
    rig.position.copy(saved.rigPos);
    rig.quaternion.copy(saved.rigQuat);
    rig.scale.copy(saved.rigScale);
    cam.position.copy(saved.camPos);
    cam.quaternion.copy(saved.camQuat);
    rig.updateMatrixWorld(true);
    ctx.controlsEnabled = saved.controls;
  }
}
