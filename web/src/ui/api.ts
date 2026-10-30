// Lane ui: client for the scene server (docs/SERVER_API.md). All URLs are relative; the Vite
// dev server proxies /api and /ws to the FastAPI server on port 8000.
import type { SceneManifest, Vec3 } from "../scene";
import type { EditRequest } from "./edits";

export interface CommandResult {
  transcript?: string;
  reply: string;
  steps: unknown[];
  verified: boolean | null;
  scene?: SceneManifest;
}

export type AgentStage = "transcript" | "thinking" | "tool" | "verify" | "done";

export type ServerMessage =
  | { type: "scene"; scene: SceneManifest }
  | { type: "agent"; stage: AgentStage; text: string }
  | { type: "capture_request"; id: number; camera?: { eye: Vec3; target: Vec3 } };

const EDIT_TIMEOUT_MS = 5000;

/** The request reached the server and it refused (HTTP 4xx); not a reason to go offline. */
export class ServerRefused extends Error {}

async function readError(r: Response): Promise<string> {
  try {
    const body = (await r.json()) as { detail?: unknown };
    if (body.detail !== undefined) return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    // not JSON
  }
  return `HTTP ${r.status}`;
}

async function postJson<T>(url: string, body: BodyInit, headers?: HeadersInit, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { method: "POST", body, headers, signal });
  if (r.status >= 400 && r.status < 500) throw new ServerRefused(await readError(r));
  if (!r.ok) throw new Error(await readError(r));
  return (await r.json()) as T;
}

export class SceneServer {
  constructor(readonly sceneId: string) {}

  private get base(): string {
    return `/api/scenes/${encodeURIComponent(this.sceneId)}`;
  }

  /**
   * True if GET /api/health answers {"ok": true} and the server has this scene, each within
   * `timeoutMs`. A server without the scene would refuse every edit, so it counts as offline.
   */
  async health(timeoutMs = 2000): Promise<boolean> {
    try {
      const r = await fetch("/api/health", { cache: "no-store", signal: AbortSignal.timeout(timeoutMs) });
      if (!r.ok) return false;
      const body = (await r.json()) as { ok?: unknown };
      if (body.ok !== true) return false;
      const scene = await fetch(this.base, { cache: "no-store", signal: AbortSignal.timeout(timeoutMs) });
      return scene.ok;
    } catch {
      return false;
    }
  }

  async edit(req: EditRequest): Promise<SceneManifest> {
    const res = await postJson<{ ok: boolean; scene: SceneManifest }>(
      `${this.base}/edit`,
      JSON.stringify(req),
      { "Content-Type": "application/json" },
      AbortSignal.timeout(EDIT_TIMEOUT_MS), // a hung request would block the edit queue
    );
    return res.scene;
  }

  command(text: string): Promise<CommandResult> {
    return postJson(`${this.base}/command`, JSON.stringify({ text }), { "Content-Type": "application/json" });
  }

  voice(wav: ArrayBuffer): Promise<CommandResult> {
    const form = new FormData();
    form.append("audio", new Blob([wav], { type: "audio/wav" }), "command.wav");
    return postJson(`${this.base}/voice`, form);
  }

  /** Opens /ws/scenes/{id}. `onClose` fires once when the socket closes or fails. */
  connect(onMessage: (m: ServerMessage) => void, onClose: () => void): WebSocket {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/scenes/${encodeURIComponent(this.sceneId)}`);
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      try {
        onMessage(JSON.parse(ev.data) as ServerMessage);
      } catch (err) {
        console.warn("[ui] bad WebSocket message", err);
      }
    };
    ws.onclose = () => onClose();
    return ws;
  }
}
