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
  ["+ in the top bar", "add furniture, list and remove items"],
  ["Delete", "remove the selected object"],
  ["R", "bring back the last removed object"],
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

/**
 * Sample commands that name objects in this scene, so a click on one does something. The
 * generic SAMPLE_COMMANDS fill in when the scene has fewer than four useful labels.
 */
export function sampleCommands(labels: readonly string[]): string[] {
  const seen = new Set<string>();
  const big: string[] = [];
  for (const l of labels) {
    const name = l.trim().toLowerCase();
    if (!name || seen.has(name) || SKIP_LABELS.has(name)) continue;
    seen.add(name);
    big.push(name);
  }
  const pick = (i: number) => big[i % Math.max(big.length, 1)];
  if (big.length < 2) return [...SAMPLE_COMMANDS];
  const out = [
    `Move the ${pick(0)} half a metre to the left`,
    `Turn the ${pick(1)} 90 degrees`,
    `Remove the ${pick(2)}`,
  ];
  // "next to" needs a fourth object, or it names one object twice.
  if (big.length >= 4) out.push(`Put the ${pick(3)} next to the ${pick(0)}`);
  return [...new Set(out)];
}

/** Labels that make poor sample commands (room surfaces, clutter). */
const SKIP_LABELS = new Set(["wall", "floor", "ceiling", "window", "door", "book", "picture frame", "poster", "pillow", "rug", "curtain"]);

export interface OnboardingDeps {
  params: URLSearchParams;
  /** Show the touch controls instead of mouse and keyboard (mobile profile). */
  touch?: boolean;
  /** False when another button (ui/chrome.ts) reopens the panel; default true. */
  reopenButton?: boolean;
  fill(text: string): void;
  toggleGaps(): void;
  /** Placeholder until generated splats exist (M4); returns the new state. */
  toggleGenerated(): boolean;
}

export interface OnboardingHandle {
  show(on: boolean): void;
  shown(): boolean;
  /** Replaces the sample command chips (sampleCommands of the scene's labels). */
  setSamples(texts: readonly string[]): void;
}

export function installOnboarding(deps: OnboardingDeps): OnboardingHandle {
  const storage = safeStorage();
  const panel = el("div", "mr-ui-onboarding");
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-label", "Controls and sample commands");
  const reopen = el("button", "mr-ui-help");
  reopen.textContent = "?";
  reopen.title = "controls and sample commands";

  const head = el("div", "mr-ui-ob-head");
  const title = el("strong", "");
  title.textContent = "How to use";
  const close = el("button", "mr-ui-close");
  close.textContent = "×";
  close.title = "close";
  close.setAttribute("aria-label", "Close");
  head.append(title, close);

  const intro = el("p", "mr-ui-ob-intro");
  intro.textContent = "A room rebuilt from one phone video. Every object is its own splat, so you can rearrange it.";

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
  const setSamples = (texts: readonly string[]) => {
    samples.replaceChildren();
    for (const text of texts) {
      const b = el("button", "mr-ui-chip");
      b.textContent = text;
      b.addEventListener("click", () => deps.fill(text));
      samples.appendChild(b);
    }
  };
  setSamples(SAMPLE_COMMANDS);

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

  panel.append(head, intro, table, samplesTitle, samples);
  // With the chrome's menu the toggles live there (ui/chrome.ts).
  if (deps.reopenButton !== false) panel.append(toggles);
  document.body.append(panel);
  if (deps.reopenButton !== false) document.body.append(reopen);

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
  return { show, shown: () => !panel.hidden, setSamples };
}
