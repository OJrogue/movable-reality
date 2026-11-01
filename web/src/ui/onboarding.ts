// Lane ui, U4: onboarding panel for first-time visitors (judges). Lists the controls, offers
// sample commands that fill the command bar, and toggles the coverage gap overlay and the
// "show generated" view. ?onboarding=0 starts with it closed; closing it is remembered in
// localStorage. A "?" button reopens it.
import { el } from "./commandBar";

export const DISMISS_KEY = "mr.onboarding.dismissed";

export const SAMPLE_COMMANDS = [
  "Move the chair half a metre to the left",
  "Turn the lamp to face the window",
  "Remove the table",
  "Put the chair next to the table",
];

const CONTROLS: [string, string][] = [
  ["Drag empty space", "look around"],
  ["W A S D, Q / E", "walk, down / up"],
  ["Click an object", "select it"],
  ["Drag an object", "move it on the floor"],
  ["Q / E (selected)", "rotate 15°"],
  ["Delete", "hide the selected object"],
  ["R", "bring back the last hidden object"],
  ["Ctrl+Z / Cmd+Z", "undo"],
  ["Hold Space", "speak a command"],
  ["G", "coverage gap overlay"],
  ["Alt+click twice", "measure in metres"],
];

/** Phones and tablets (lane devices, mobile profile). */
export const TOUCH_CONTROLS: [string, string][] = [
  ["Drag one finger", "look around"],
  ["Drag two fingers", "move sideways and up"],
  ["Pinch", "move forward or back"],
  ["Tap an object", "select it"],
  ["Drag the selected object", "move it on the floor"],
  ["Tap empty space", "clear the selection"],
  ["Hold to talk", "speak a command"],
];

/** Read access to localStorage that never throws (private windows, blocked storage). */
export interface KeyValue {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function safeStorage(): KeyValue | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function shouldShowOnboarding(params: URLSearchParams, storage: KeyValue | null): boolean {
  if (params.get("onboarding") === "0") return false;
  if (params.get("onboarding") === "1") return true;
  try {
    return storage?.getItem(DISMISS_KEY) !== "1";
  } catch {
    return true;
  }
}

export interface OnboardingDeps {
  params: URLSearchParams;
  /** Show the touch controls instead of mouse and keyboard (mobile profile). */
  touch?: boolean;
  fill(text: string): void;
  toggleGaps(): void;
  /** Placeholder until generated splats exist (M4); returns the new state. */
  toggleGenerated(): boolean;
}

export function installOnboarding(deps: OnboardingDeps): void {
  const storage = safeStorage();
  const panel = el("div", "mr-ui-onboarding");
  const reopen = el("button", "mr-ui-help");
  reopen.textContent = "?";
  reopen.title = "controls and sample commands";

  const head = el("div", "mr-ui-ob-head");
  const title = el("strong", "");
  title.textContent = "Movable Reality";
  const close = el("button", "mr-ui-close");
  close.textContent = "×";
  close.title = "close";
  head.append(title, close);

  const intro = el("p", "mr-ui-ob-intro");
  intro.textContent = "A room scanned from a 30 s phone video. Every object is its own splat, so you can rearrange it.";

  const table = el("table", "mr-ui-ob-keys");
  for (const [key, what] of deps.touch ? TOUCH_CONTROLS : CONTROLS) {
    const tr = el("tr", "");
    const k = el("td", "");
    const w = el("td", "");
    k.textContent = key;
    w.textContent = what;
    tr.append(k, w);
    table.appendChild(tr);
  }

  const samplesTitle = el("div", "mr-ui-ob-sub");
  samplesTitle.textContent = "Try a command";
  const samples = el("div", "mr-ui-ob-samples");
  for (const text of SAMPLE_COMMANDS) {
    const b = el("button", "mr-ui-chip");
    b.textContent = text;
    b.addEventListener("click", () => deps.fill(text));
    samples.appendChild(b);
  }

  const toggles = el("div", "mr-ui-ob-toggles");
  const gaps = el("button", "mr-ui-toggle");
  gaps.textContent = "Coverage gaps (G)";
  gaps.addEventListener("click", () => {
    deps.toggleGaps();
    gaps.blur(); // Space must stay hold-to-talk, not click this button again
  });
  const generated = el("button", "mr-ui-toggle");
  generated.textContent = "Show generated";
  generated.title = "highlights generated splats once the scene has any (M4)";
  generated.addEventListener("click", () => {
    generated.dataset.on = deps.toggleGenerated() ? "1" : "0";
    generated.blur();
  });
  toggles.append(gaps, generated);

  panel.append(head, intro, table, samplesTitle, samples, toggles);
  document.body.append(panel, reopen);

  const show = (on: boolean) => {
    panel.hidden = !on;
    reopen.hidden = on;
  };
  show(shouldShowOnboarding(deps.params, storage));
  close.addEventListener("click", () => {
    show(false);
    try {
      storage?.setItem(DISMISS_KEY, "1");
    } catch {
      // storage full or blocked: the panel just reappears next time
    }
  });
  reopen.addEventListener("click", () => {
    show(true);
    reopen.blur();
  });
}
