// Lane ui, U2: the command bar. A text box at the bottom (Enter sends POST .../command) and
// hold-to-talk on Space while the text box is not focused (WAV upload to .../voice). It shows
// the transcript, the agent's progress messages from the WebSocket and the reply. On touch
// devices (lane devices sets setTouch) a microphone button does hold-to-talk instead.
import type { AgentStage, CommandResult, SceneServer, ViewPose } from "./api";
import { isTyping } from "./editor";
import { Recorder } from "./recorder";

const MAX_LINES = 6;
const MIC_SVG =
  '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" ' +
  'stroke-linecap="round" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/>' +
  '<path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/></svg>';
const MIN_TALK_S = 0.3;
export const NEED_SERVER =
  "Commands need the agent, which is offline right now. You can still click, drag and rotate objects.";
/** Shown in the console (and with developer info on) next to NEED_SERVER. */
export const START_SERVER = "start it with: uv run python -m movable_reality.server";

export type LineKind = "you" | "agent" | "reply" | "info" | "error";

export interface CommandBarDeps {
  server: SceneServer | null;
  online(): boolean;
  /** Applies the manifest a command returned (the WebSocket push may arrive first). */
  applyResult(r: CommandResult): void;
  xr(): boolean;
  /** The user's current pose, sent with every command. */
  camera(): ViewPose | null;
}

export class CommandBar {
  readonly root: HTMLDivElement;
  private readonly log: HTMLDivElement;
  private readonly input: HTMLInputElement;
  private readonly mic: HTMLSpanElement;
  private readonly badge: HTMLSpanElement;
  readonly talkButton: HTMLButtonElement;
  private readonly recorder = new Recorder();
  private talking = false;
  private busy = false;
  private lastHeard: string | null = null; // transcript already shown from the WebSocket
  private status: HTMLElement | null = null; // the agent's progress line, updated in place
  private lastReply: string | null = null;

  constructor(private readonly deps: CommandBarDeps) {
    this.root = el("div", "mr-ui-bar");
    this.log = el("div", "mr-ui-log");
    const row = el("div", "mr-ui-row");
    this.mic = el("span", "mr-ui-mic");
    this.mic.innerHTML = MIC_SVG;
    this.mic.title = "hold Space to talk";
    this.input = document.createElement("input");
    this.input.type = "text";
    this.input.className = "mr-ui-input";
    this.input.placeholder = "Type a command, or hold Space to talk";
    this.input.autocomplete = "off";
    this.input.spellcheck = false;
    this.badge = el("span", "mr-ui-badge");
    this.talkButton = el("button", "mr-ui-talk");
    this.talkButton.type = "button";
    this.talkButton.innerHTML = `${MIC_SVG}<span>Hold to talk</span>`;
    this.talkButton.setAttribute("aria-label", "hold to talk");
    row.append(this.mic, this.input, this.badge, this.talkButton);
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

    // Hold-to-talk by touch: press starts recording, lifting the finger sends it. The
    // button captures the pointer, so sliding off it keeps recording until the lift.
    const talk = this.talkButton;
    talk.addEventListener("pointerdown", (e) => {
      e.preventDefault(); // no focus change, no text selection
      try {
        talk.setPointerCapture(e.pointerId);
      } catch {
        // synthetic events have no pointer to capture
      }
      this.beginTalk();
    });
    const release = () => {
      if (this.talking) void this.stopTalking();
    };
    talk.addEventListener("pointerup", release);
    talk.addEventListener("pointercancel", release);
    talk.addEventListener("contextmenu", (e) => e.preventDefault()); // long press menu
  }

  /** Touch layout: a hold-to-talk button and a shorter hint in the text box. */
  setTouch(on: boolean): void {
    if (on) this.root.dataset.touch = "1";
    else delete this.root.dataset.touch;
    this.input.placeholder = on ? "Type a command" : "Type a command, or hold Space to talk";
  }

  /** Called when the server state changes; `text` describes it. */
  setMode(online: boolean, text: string): void {
    this.badge.textContent = text;
    this.badge.title = text;
    this.badge.dataset.state = online ? "online" : "offline";
  }

  /** Puts text into the box and focuses it (the onboarding sample commands use this). */
  fill(text: string): void {
    this.input.value = text;
    this.input.focus();
  }

  line(kind: LineKind, text: string): HTMLElement {
    const p = el("div", `mr-ui-line mr-ui-${kind}`);
    p.textContent = text;
    this.log.appendChild(p);
    while (this.log.childElementCount > MAX_LINES) this.log.firstElementChild?.remove();
    return p;
  }

  private needServer(): void {
    this.line("error", NEED_SERVER);
    console.info(`[ui] ${START_SERVER}`);
    if (document.body.hasAttribute("data-debug")) this.line("info", START_SERVER);
  }

  /** Agent progress from the WebSocket. One status line ("Moving the bed…") is updated in
   * place; the raw tool calls are developer info. */
  agent(stage: AgentStage, text: string, say?: string): void {
    if (stage === "transcript") {
      this.lastHeard = text;
      this.line("you", `heard: ${text}`);
      return;
    }
    if (stage === "done") {
      // Our own command prints its reply from the HTTP answer, which can arrive first; a
      // different text while idle is another viewer's. `done` is always the last event.
      if (!this.busy && text !== this.lastReply) this.line("reply", text);
      this.clearStatus();
      return;
    }
    if (document.body.hasAttribute("data-debug")) this.line("agent", `${stage}: ${text}`).classList.add("mr-debug");
    if (!this.status || !this.status.isConnected) {
      this.status = this.line("agent", "");
      this.status.classList.add("mr-ui-status");
    }
    const words = say ?? text;
    this.status.textContent = words.endsWith(".") ? words : `${words}…`; // "Looks right." is done
    this.log.appendChild(this.status); // stays the newest line
  }

  private clearStatus(): void {
    this.status?.remove();
    this.status = null;
  }

  private async send(raw: string): Promise<void> {
    const text = raw.trim();
    if (!text) return;
    if (!this.deps.server || !this.deps.online()) {
      this.needServer();
      return;
    }
    if (this.busy) {
      this.line("info", "still working on the last command");
      return;
    }
    this.input.value = "";
    this.line("you", text);
    await this.run(() => this.deps.server!.command(text, this.deps.camera()));
  }

  private async run(call: () => Promise<CommandResult>): Promise<void> {
    this.busy = true;
    this.lastHeard = null;
    this.root.dataset.busy = "1";
    try {
      const r = await call();
      if (r.transcript !== undefined && r.transcript !== this.lastHeard) this.line("you", `heard: ${r.transcript}`);
      this.lastHeard = null;
      this.clearStatus();
      this.lastReply = r.reply;
      const check = r.verified === true ? " ✓ Checked" : r.verified === false ? " (the visual check disagreed)" : "";
      this.line("reply", `${r.reply}${check}`);
      this.deps.applyResult(r);
    } catch (err) {
      this.clearStatus();
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
    if (e.repeat) return;
    this.beginTalk();
  }

  private beginTalk(): void {
    if (this.talking) return;
    if (!this.deps.server || !this.deps.online()) {
      this.needServer();
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
    } catch {
      return; // start() failed and its handler has already shown the error
    }
    if (rec.seconds < MIN_TALK_S) {
      this.line("info", this.root.dataset.touch ? "hold the button while you speak" : "hold Space while you speak");
      return;
    }
    this.line("info", `sending ${rec.seconds.toFixed(1)} s of audio`);
    await this.run(() => this.deps.server!.voice(rec.wav, this.deps.camera()));
  }
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  return e;
}

export { el };
