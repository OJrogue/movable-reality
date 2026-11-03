// Lane ui: the viewer's app chrome. A top bar with a way home, the scene switcher (the
// three demo scenes, landing/scenes.ts), Enter VR, help and a settings menu; a loading card
// until the room's splats arrive. Developer info (HUD, frame log button, detail slider,
// object style label) is hidden by default and lives behind the settings menu's last
// switch or the ` key; ?debug=1 or ?bench=1 start with it on, and the choice is remembered.
// ?chrome=0 hides the top bar (clean recordings). Lanes mark their debug widgets with the
// class mr-debug (CSS in style.ts hides them while body has no data-debug).
import * as THREE from "three";
import type { ViewerContext } from "../context";
import { DEMO_SCENES, findDemoScene, viewerUrl } from "../landing/scenes";
import type { ObjectStyle } from "../scene";
import { OBJECT_STYLES } from "../scene";
import { isTyping } from "./editor";
import { type KeyValue, type OnboardingHandle, safeStorage } from "./onboarding";

export const DEBUG_KEY = "mr.debug";
const LOADING_TIMEOUT_MS = 120_000; // never leave the card up for good

/** ?debug wins, bench mode and an explicit ?hud=1 turn it on, then the remembered choice. */
export function debugInitial(params: URLSearchParams, storage: KeyValue | null): boolean {
  const d = params.get("debug");
  if (d === "1" || d === "0") return d === "1";
  if (params.get("bench") === "1" || params.get("hud") === "1") return true;
  try {
    return storage?.getItem(DEBUG_KEY) === "1";
  } catch {
    return false;
  }
}

export const STYLE_NAMES: Record<ObjectStyle, string> = {
  auto: "Automatic",
  scan: "Scanned splats",
  completed: "SAM 3D splats",
  mesh: "SAM 3D meshes",
  meshonly: "Meshes only",
};

interface CoverageLike {
  visible(): boolean;
  toggle(): void;
  generatedVisible(): boolean;
  toggleGenerated(): void;
  generatedCount(): number;
}
interface ObjectsLike {
  style(): ObjectStyle;
  setStyle(s: ObjectStyle): void;
}
/** The coverage and objects lanes install after ui; their handles are read when a menu opens. */
const lanes = () => window as unknown as { mrCoverage?: CoverageLike; mrObjects?: ObjectsLike };

const svg = (d: string) =>
  `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const ICONS = {
  back: svg('<path d="M15 5l-7 7 7 7"/>'),
  chevron: svg('<path d="M7 10l5 5 5-5"/>'),
  help: svg('<circle cx="12" cy="12" r="9"/><path d="M9.6 9.3a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2.2-2.4 3.8"/><circle cx="12" cy="17.2" r=".6" fill="currentColor"/>'),
  more: svg('<circle cx="12" cy="5.5" r="1.4" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/><circle cx="12" cy="18.5" r="1.4" fill="currentColor" stroke="none"/>'),
  vr: svg('<path d="M3 9a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-4l-1.6-2.4a1.7 1.7 0 0 0-2.8 0L9 17H5a2 2 0 0 1-2-2z"/><circle cx="8" cy="12" r="1.4"/><circle cx="16" cy="12" r="1.4"/>'),
  check: svg('<path d="M5 12.5l4.5 4.5L19 7.5"/>'),
};

export interface ChromeDeps {
  ctx: ViewerContext;
  onboarding: OnboardingHandle;
  notify(text: string): void;
}

export interface ChromeHandle {
  debug(): boolean;
  setDebug(on: boolean): void;
}

function h<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, html = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

/** A popover menu under an anchor button; one open at a time, Escape or a click outside closes. */
class Menu {
  readonly el = h("div", "mr-chrome-menu");
  private static open: Menu | null = null;

  constructor(
    private readonly anchor: HTMLElement,
    private readonly build: (menu: HTMLElement) => void,
    align: "left" | "right",
  ) {
    this.el.dataset.align = align;
    this.el.hidden = true;
    // Keys inside the menu must not walk the camera or trigger the viewer's shortcuts.
    this.el.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Escape") this.close();
    });
    anchor.setAttribute("aria-haspopup", "true");
    anchor.setAttribute("aria-expanded", "false");
    anchor.addEventListener("click", (e) => {
      e.stopPropagation();
      if (this.el.hidden) this.show();
      else this.close();
    });
    document.addEventListener("pointerdown", (e) => {
      if (!this.el.hidden && !this.el.contains(e.target as Node) && !anchor.contains(e.target as Node)) this.close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !this.el.hidden) this.close();
    });
  }

  show(): void {
    Menu.open?.close();
    Menu.open = this;
    this.el.replaceChildren();
    this.build(this.el);
    this.el.hidden = false;
    this.anchor.setAttribute("aria-expanded", "true");
    this.el.querySelector<HTMLElement>("button:not(:disabled)")?.focus({ preventScroll: true });
  }

  close(): void {
    if (this.el.hidden) return;
    this.el.hidden = true;
    this.anchor.setAttribute("aria-expanded", "false");
    if (Menu.open === this) Menu.open = null;
    this.anchor.blur(); // Space must stay hold-to-talk
  }
}

function menuItem(label: string, opts: { caption?: string; checked?: boolean; disabled?: boolean; href?: string } = {}): HTMLElement {
  const item = opts.href ? h("a", "mr-chrome-item") : h("button", "mr-chrome-item");
  if (item instanceof HTMLAnchorElement) item.href = opts.href!;
  else item.type = "button";
  // Plain buttons and links in a popover (no arrow-key menu): the current choice is marked.
  if (opts.checked) item.setAttribute("aria-current", "true");
  if (opts.disabled && item instanceof HTMLButtonElement) item.disabled = true;
  const text = h("span", "mr-chrome-item-text");
  const t = h("span", "mr-chrome-item-label");
  t.textContent = label;
  text.append(t);
  if (opts.caption) {
    const c = h("span", "mr-chrome-item-caption");
    c.textContent = opts.caption;
    text.append(c);
  }
  const mark = h("span", "mr-chrome-check", opts.checked ? ICONS.check : "");
  item.append(text, mark);
  return item;
}

function switchItem(label: string, caption: string, on: boolean, disabled: boolean, toggle: () => boolean): HTMLButtonElement {
  const item = h("button", "mr-chrome-item");
  item.type = "button";
  item.setAttribute("role", "menuitemcheckbox");
  item.setAttribute("aria-checked", String(on));
  item.disabled = disabled;
  const text = h("span", "mr-chrome-item-text");
  const t = h("span", "mr-chrome-item-label");
  t.textContent = label;
  const c = h("span", "mr-chrome-item-caption");
  c.textContent = caption;
  text.append(t, c);
  const sw = h("span", "mr-chrome-switch");
  sw.dataset.on = on ? "1" : "0";
  item.append(text, sw);
  item.addEventListener("click", () => {
    const now = toggle();
    sw.dataset.on = now ? "1" : "0";
    item.setAttribute("aria-checked", String(now));
  });
  return item;
}

const section = (title: string) => {
  const s = h("div", "mr-chrome-section");
  s.textContent = title;
  return s;
};

export function installChrome(deps: ChromeDeps): ChromeHandle {
  const { ctx, onboarding } = deps;
  const storage = safeStorage();
  const sceneId = ctx.params.get("scene");
  const demo = findDemoScene(sceneId);
  const base = ctx.params.get("base");

  // ---- developer info
  let debug = false;
  const setDebug = (on: boolean, remember = true) => {
    debug = on;
    document.body.toggleAttribute("data-debug", on);
    if (!remember) return;
    try {
      storage?.setItem(DEBUG_KEY, on ? "1" : "0");
    } catch {
      // blocked storage: the choice lasts for this page
    }
  };
  setDebug(debugInitial(ctx.params, storage), false);
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Backquote" || e.repeat || isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
    setDebug(!debug);
    deps.notify(`Developer info ${debug ? "on" : "off"}`);
  });
  if (ctx.params.get("chrome") === "0") document.body.dataset.chrome = "0";

  // ---- top bar
  const bar = h("header", "mr-chrome-top");
  const left = h("div", "mr-chrome-group");
  const home = h("a", "mr-chrome-icon", ICONS.back);
  home.href = "/";
  home.title = "All rooms";
  home.setAttribute("aria-label", "Back to the start page");
  const title = h("button", "mr-chrome-title");
  title.type = "button";
  const titleText = h("span", "mr-chrome-title-text");
  titleText.textContent = demo?.title ?? sceneId ?? "Movable Reality";
  const sub = h("span", "mr-chrome-title-sub");
  sub.textContent = demo ? (demo.kind === "real" ? "Real capture" : "Synthetic room") : "Scene";
  const titleStack = h("span", "mr-chrome-title-stack");
  titleStack.append(titleText, sub);
  title.append(titleStack, h("span", "mr-chrome-chevron", ICONS.chevron));
  title.title = "Switch room";
  left.append(home, title);

  const right = h("div", "mr-chrome-group");
  // main.ts hands Spark this element as its Enter VR button; Spark shows it when WebXR works.
  // VR is untested on headsets for the demo, so the button only shows with developer info.
  const vr = document.getElementById("mr-vr");
  if (vr) {
    vr.className = "mr-chrome-vr mr-debug";
    vr.innerHTML = `${ICONS.vr}<span>Enter VR</span>`;
    vr.setAttribute("aria-label", "Enter VR");
    right.append(vr);
  }
  const help = h("button", "mr-chrome-icon", ICONS.help);
  help.type = "button";
  help.title = "Controls and sample commands";
  help.setAttribute("aria-label", "Controls and sample commands");
  help.addEventListener("click", () => {
    onboarding.show(!onboarding.shown());
    help.blur();
  });
  const more = h("button", "mr-chrome-icon", ICONS.more);
  more.type = "button";
  more.title = "View options";
  more.setAttribute("aria-label", "View options");
  right.append(help, more);
  bar.append(left, right);

  const scenesMenu = new Menu(
    title,
    (m) => {
      m.append(section("Rooms"));
      for (const s of DEMO_SCENES) {
        m.append(menuItem(s.title, { caption: s.blurb, checked: s.id === sceneId, href: viewerUrl(s.id, base) }));
      }
    },
    "left",
  );

  const settingsMenu = new Menu(
    more,
    (m) => {
      const cov = lanes().mrCoverage;
      const obj = lanes().mrObjects;
      m.append(section("View"));
      m.append(
        switchItem("Coverage gaps", "Where the video saw too little  ·  G", cov?.visible() ?? false, !cov, () => {
          cov?.toggle();
          return cov?.visible() ?? false;
        }),
      );
      const n = cov?.generatedCount() ?? 0;
      m.append(
        switchItem(
          "Generated splats",
          n > 0 ? `${n.toLocaleString()} splats filled in by Muse Image  ·  T` : "None in this room yet",
          n > 0 && (cov?.generatedVisible() ?? false),
          n === 0,
          () => {
            cov?.toggleGenerated();
            return cov?.generatedVisible() ?? false;
          },
        ),
      );
      if (obj) {
        m.append(section("Objects drawn as"));
        const current = obj.style();
        for (const s of OBJECT_STYLES) {
          const item = menuItem(STYLE_NAMES[s], { checked: s === current });
          item.addEventListener("click", () => {
            obj.setStyle(s);
            settingsMenu.close();
          });
          m.append(item);
        }
      }
      m.append(h("div", "mr-chrome-divider"));
      m.append(
        switchItem("Developer info", "Frame times, splat budget, logs  ·  `", debug, false, () => {
          setDebug(!debug);
          return debug;
        }),
      );
    },
    "right",
  );
  document.body.append(bar, scenesMenu.el, settingsMenu.el);

  // ---- loading card, until the background splats are in (or an error)
  const loading = h("div", "mr-chrome-loading");
  loading.setAttribute("role", "status");
  const card = h("div", "mr-chrome-loading-card");
  const lt = h("div", "mr-chrome-loading-title");
  lt.textContent = demo?.title ?? sceneId ?? "";
  const ls = h("div", "mr-chrome-loading-sub");
  ls.textContent = "Loading the room";
  const bar2 = h("div", "mr-chrome-progress", "<span></span>");
  card.append(lt, ls, bar2);
  loading.append(card);
  if (sceneId) document.body.append(loading);
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    loading.dataset.out = "1";
    setTimeout(() => loading.remove(), 400);
  };
  const fail = (text: string) => {
    if (done) return;
    done = true;
    ls.textContent = text;
    bar2.remove();
    const back = h("a", "mr-chrome-loading-back");
    back.href = "/";
    back.textContent = "Back to all rooms";
    card.append(back);
  };
  // ---- start view: a demo room opens looking at its furniture, not at the origin's wall.
  // Bench walks, eval views (mrCamera) and ?view=0 keep the default.
  let placed = !demo?.view || ctx.params.has("bench") || ctx.params.get("view") === "0";
  ctx.onManifest(() => {
    if (placed || !demo?.view) return;
    placed = true;
    const eye = new THREE.Vector3(...demo.view.eye);
    const look = new THREE.Matrix4().lookAt(eye, new THREE.Vector3(...demo.view.target), new THREE.Vector3(0, 1, 0));
    ctx.rig.quaternion.setFromRotationMatrix(look);
    // The camera sits 1.6 m up inside the rig, and the rig is pitched: undo that offset.
    ctx.rig.position.copy(eye).sub(ctx.camera.position.clone().applyQuaternion(ctx.rig.quaternion));
  });
  // A headset needs a level rig on the floor, or the whole room tilts.
  ctx.renderer.xr.addEventListener("sessionstart", () => {
    const yaw = new THREE.Euler().setFromQuaternion(ctx.rig.quaternion, "YXZ").y;
    ctx.rig.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    ctx.rig.position.y = 0;
  });
  ctx.onManifest((m, view) => {
    // The furniture loads after the background; an empty room under a hidden card looked
    // broken (blender06, 3 Nov). Object failures do not hold the card (allSettled).
    const objects = m.objects
      .filter((o) => o.visible)
      .map((o) => view.objectNode(o.id)?.mesh.initialized)
      .filter((p) => p !== undefined);
    view.backgroundMesh()?.initialized.then(
      () => void Promise.allSettled(objects).then(finish),
      () => fail("This room could not be loaded."),
    );
  });
  // A missing scene.json falls back to a plain PLY (main.ts). Only the background failing
  // fails the room (above): a missing object PLY leaves the rest of the room usable.
  // A demo room without a scene.json is still in the pipeline; other scenes get 10 s.
  let checks = 0;
  const started = performance.now();
  const watch = () => {
    if (done) return;
    if (ctx.status.startsWith("single PLY")) {
      if (demo) return fail("This room is still being processed. Try another one.");
      if (++checks > 40) return finish();
    }
    if (performance.now() - started > LOADING_TIMEOUT_MS) return finish();
    setTimeout(watch, 250);
  };
  watch();

  return { debug: () => debug, setDebug };
}
