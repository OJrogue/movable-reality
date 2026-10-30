// Lane ui, U2: the command bar. A text box at the bottom (Enter sends POST .../command) and
// hold-to-talk on Space while the text box is not focused (WAV upload to .../voice). It shows
// the transcript, the agent's progress messages from the WebSocket and the reply.
import type { AgentStage, CommandResult, SceneServer } from "./api";
import { isTyping } from "./editor";
import { Recorder } from "./recorder";

const MAX_LINES = 6;
const MIN_TALK_S = 0.3;
export const NEED_SERVER = "commands need the scene server: uv run python -m movable_reality.server";

export type LineKind = "you" | "agent" | "reply" | "info" | "error";

export interface CommandBarDeps {
  server: SceneServer | null;
  online(): boolean;
  /** Applies the manifest a command returned (the WebSocket push may arrive first). */
  applyResult(r: CommandResult): void;
  xr(): boolean;
}

export class CommandBar {
  readonly root: HTMLDivElement;
  private readonly log: HTMLDivElement;
  private readonly input: HTMLInputElement;
  private readonly mic: HTMLSpanElement;
  private readonly badge: HTMLSpanElement;
  private readonly recorder = new Recorder();
  private talking = false;
  private busy = false;
  private lastHeard: string | null = null; // transcript already shown from the WebSocket

  constructor(private readonly deps: CommandBarDeps) {
    this.root = el("div", "mr-ui-bar");
    this.log = el("div", "mr-ui-log");
    const row = el("div", "mr-ui-row");
    this.mic = el("span", "mr-ui-mic");
    this.mic.textContent = "●";
    this.mic.title = "hold Space to talk";
    this.input = document.createElement("input");
    this.input.type = "text";
    this.input.className = "mr-ui-input";
    this.input.placeholder = "Type a command, or hold Space to talk";
    this.input.autocomplete = "off";
    this.input.spellcheck = false;
    this.badge = el("span", "mr-ui-badge");
    row.append(this.mic, this.input, this.badge);
    this.root.append(this.log, row);
    document.body.appendChild(this.root);

    // Keys typed here must not reach SparkControls (WASD, Q/E) or the lanes' shortcuts,
    // which listen on the document.
    this.input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.isComposing) {
        e.preventDefault();
        void this.send(this.input.value);
      } else if (e.key === "Escape") this.input.blur();
    });
    this.input.addEventListener("keyup", (e) => e.stopPropagation());

    window.addEventListener("keydown", (e) => this.onSpace(e, true), { capture: true });
    window.addEventListener("keyup", (e) => this.onSpace(e, false), { capture: true });
    window.addEventListener("blur", () => {
      if (this.talking) void this.stopTalking();
    });
  }

  /** Called when the server state changes; `text` describes it. */
  setMode(online: boolean, text: string): void {
    this.badge.textContent = text;
    this.badge.dataset.state = online ? "online" : "offline";
  }

  /** Puts text into the box and focuses it (the onboarding sample commands use this). */
  fill(text: string): void {
    this.input.value = text;
    this.input.focus();
  }

  line(kind: LineKind, text: string): void {
    const p = el("div", `mr-ui-line mr-ui-${kind}`);
    p.textContent = text;
    this.log.appendChild(p);
    while (this.log.childElementCount > MAX_LINES) this.log.firstElementChild?.remove();
  }

  /** Agent progress from the WebSocket. */
  agent(stage: AgentStage, text: string): void {
    if (stage === "transcript") {
      this.lastHeard = text;
      this.line("you", `heard: ${text}`);
    }
    else if (stage === "done") this.line("reply", text);
    else this.line("agent", `${stage}: ${text}`);
  }

  private async send(raw: string): Promise<void> {
    const text = raw.trim();
    if (!text) return;
    if (!this.deps.server || !this.deps.online()) {
      this.line("error", NEED_SERVER);
      return;
    }
    if (this.busy) {
      this.line("info", "still working on the last command");
      return;
    }
    this.input.value = "";
    this.line("you", text);
    await this.run(() => this.deps.server!.command(text));
  }

  private async run(call: () => Promise<CommandResult>): Promise<void> {
    this.busy = true;
    this.lastHeard = null;
    this.root.dataset.busy = "1";
    try {
      const r = await call();
      if (r.transcript !== undefined && r.transcript !== this.lastHeard) this.line("you", `heard: ${r.transcript}`);
      this.lastHeard = null;
      const check = r.verified === true ? " (verified)" : r.verified === false ? " (verifier disagreed)" : "";
      this.line("reply", `${r.reply}${check}`);
      this.deps.applyResult(r);
    } catch (err) {
      this.line("error", `command failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      this.busy = false;
      delete this.root.dataset.busy;
    }
  }

  // ------------------------------------------------------------ hold to talk

  private onSpace(e: KeyboardEvent, down: boolean): void {
    if (e.code !== "Space" || this.deps.xr() || isTyping(e.target)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault(); // no page scroll, no click on a focused button
    if (!down) {
      if (this.talking) void this.stopTalking();
      return;
    }
    if (e.repeat || this.talking) return;
    if (!this.deps.server || !this.deps.online()) {
      this.line("error", NEED_SERVER);
      return;
    }
    if (this.busy) {
      this.line("info", "still working on the last command");
      return;
    }
    this.talking = true;
    this.root.dataset.talking = "1";
    this.recorder.start().catch((err: unknown) => {
      this.talking = false;
      delete this.root.dataset.talking;
      this.line("error", `microphone: ${err instanceof Error ? err.message : String(err)}`);
    });
  }

  private async stopTalking(): Promise<void> {
    this.talking = false;
    delete this.root.dataset.talking;
    if (!this.recorder.recording) return;
    let rec: { wav: ArrayBuffer; seconds: number };
    try {
      rec = await this.recorder.stop();
    } catch (err) {
      this.line("error", `microphone: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    if (rec.seconds < MIN_TALK_S) {
      this.line("info", "hold Space while you speak");
      return;
    }
    this.line("info", `sending ${rec.seconds.toFixed(1)} s of audio`);
    await this.run(() => this.deps.server!.voice(rec.wav));
  }
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  return e;
}

export { el };
