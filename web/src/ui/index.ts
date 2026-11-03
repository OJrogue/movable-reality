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
import { installMeasure } from "./measure";
import { installChrome } from "./chrome";
import { installOnboarding, sampleCommands } from "./onboarding";
import { UI_CSS } from "./style";

/** main.ts passes the full context, which can also publish a manifest to every lane. */
export type UiContext = ViewerContext & {
  runManifest(manifest: SceneManifest, view: SceneView): void;
};

const RETRY_MS = 5000;
const OFFLINE_TEXT = "Agent offline";

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

  const bar = new CommandBar({
    server,
    online: () => online,
    applyResult: (r: CommandResult) => {
      if (r.scene) publish(r.scene);
    },
    xr: () => ctx.renderer.xr.isPresenting,
  });

  // Lane devices: the mobile profile lays the bar out for a narrow portrait screen with a
  // hold-to-talk button (style.ts, body[data-device="mobile"]).
  const touchUi = () => ctx.device?.name === "mobile";
  bar.setTouch(touchUi());
  ctx.onDeviceChange(() => bar.setTouch(touchUi()));

  const setOnline = (on: boolean) => {
    const was = online;
    online = on;
    bar.setMode(on, on ? "Agent online" : sceneId ? OFFLINE_TEXT : "No scene");
    if (on && !was) {
      latest = null; // the server's scene is the truth from here on
      bar.line("info", "Connected to the agent. Type or speak a command.");
      if (localEdits > 0) bar.line("info", `${localEdits} local edits replaced by the server's scene`);
      localEdits = 0;
    }
    if (!on && was) bar.line("info", "Lost the agent. Edits stay in this tab until it is back.");
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
  installMeasure(ctx, (t) => bar.line("info", t));

  // ---- server connection: health check, then the WebSocket; retry while it is down.
  const onMessage = (msg: ServerMessage) => {
    if (msg.type === "scene") publish(msg.scene);
    else if (msg.type === "agent") bar.agent(msg.stage, msg.text, msg.say);
    else if (msg.type === "capture_request") {
      const ws = socket;
      void capture(ctx, msg.camera).then((jpeg) => {
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

  const onboarding = installOnboarding({
    params: ctx.params,
    touch: touchUi(),
    reopenButton: false, // the chrome's help button reopens it
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

  ctx.onManifest((m) => onboarding.setSamples(sampleCommands(m.objects.filter((o) => o.visible).map((o) => o.label))));
  installChrome({ ctx, onboarding, notify: (t) => bar.line("info", t) });

  const handle: UiHandle = { editor, bar, online: () => online, edit, manifest: () => ctx.manifest };
  (window as unknown as { mrUi?: UiHandle }).mrUi = handle; // for console checks and tests
  return handle;
}

const _eye = new THREE.Vector3();
const SETTLE_TIMEOUT_MS = 1500; // a hidden tab runs no frames
const SETTLE_RENDERS = 12;

/**
 * Screenshot for the verifier. With a camera request, the rig and camera move to `eye`
 * looking at `target` for about 1.5 s of renders, so Spark's sort and the LOD cut catch up with the
 * new viewpoint, and then go back. In XR the headset renders into its own framebuffer and
 * the canvas holds no image, so the answer is null.
 */
async function capture(
  ctx: ViewerContext,
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
    // Render here instead of waiting for frames: a background tab or a hidden pane runs no
    // animation frames (document.hidden can still be false), and Spark's sort and LOD cut
    // only catch up on render. Without this the verifier saw no furniture (blender06, 3 Nov).
    for (let i = 0; i < SETTLE_RENDERS; i++) {
      ctx.renderer.render(ctx.scene, cam);
      await new Promise((r) => setTimeout(r, SETTLE_TIMEOUT_MS / SETTLE_RENDERS));
    }
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
