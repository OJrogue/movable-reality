// Landing page at / (entry.ts loads it when the URL names no scene). A showcase video
// behind the hero, an "Open the demo" button that offers the three scenes, and what the
// project does and how. The video is web/public/media/showcase.mp4 (gitignored like every
// .mp4: copy it next to the build when hosting); without it the hero shows a still.
import { DEMO_SCENES, type DemoScene, sceneAvailability, viewerUrl } from "./scenes";
import { LANDING_CSS } from "./style";

const REPO = "https://github.com/OJrogue/movable-reality";

const icon = (d: string) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const I = {
  play: icon('<path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none"/>'),
  pause: icon('<path d="M8 5v14M16 5v14" stroke-width="3"/>'),
  close: icon('<path d="M6 6l12 12M18 6L6 18"/>'),
  down: icon('<path d="M12 5v14M6 13l6 6 6-6"/>'),
  video: icon('<rect x="3" y="6" width="13" height="12" rx="2.5"/><path d="M16 10.5l5-3v9l-5-3"/>'),
  cube: icon('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>'),
  sparkle: icon('<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 16l.7 1.8 1.8.7-1.8.7L19 21l-.7-1.8-1.8-.7 1.8-.7z"/>'),
  mic: icon('<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>'),
  devices: icon('<rect x="2.5" y="5" width="13" height="10" rx="1.5"/><path d="M6 19h6"/><rect x="17.5" y="8" width="4.5" height="11" rx="1.2"/>'),
  ruler: icon('<path d="M3 17L17 3l4 4L7 21z"/><path d="M7 13l2 2M10 10l2 2M13 7l2 2"/>'),
};

const FEATURES: [string, string, string][] = [
  [I.video, "Film it with a phone", "Walk once around the room with a phone. MapAnything recovers metric cameras and geometry, so a metre in the scene is a metre in the room."],
  [I.cube, "Every object is its own splat", "Muse Wonder lists what is in the room, SAM 3.1 tracks each object through the video, and every Gaussian votes for the object it belongs to."],
  [I.sparkle, "Gaps found and filled", "Coverage scoring finds what the video missed. SAM 3D Objects completes the backs of objects, Muse Image repairs gaps, and generated splats stay tagged."],
  [I.mic, "Rearrange it by voice", "Say “move the sofa under the window”. A Muse Wonder agent edits the scene with tool calls and Muse Spark Open checks a screenshot of the result."],
  [I.devices, "Runs in a browser", "Nothing to install. The same viewer runs on a laptop and on a phone, and picks the level of detail each device can hold."],
  [I.ruler, "Measured, not guessed", "Alt+click twice to measure anything in metres. The floor sits at y = 0 and every edit is an undoable delta in one scene file."],
];

interface Step {
  title: string;
  text: string;
  flow: string[]; // "*Name" marks a Meta model
}

const STEPS: Step[] = [
  {
    title: "Capture",
    text: "One walk around the room with a phone. The sharpest frames are kept at 3 to 5 per second, and the same frames, in the same order, become the clip that SAM tracks.",
    flow: ["phone video", "sharpest frames", "SAM clip"],
  },
  {
    title: "Metric reconstruction",
    text: "MapAnything predicts metric cameras and geometry, exported to COLMAP. gsplat trains a 3D Gaussian Splatting scene on full-resolution frames on a Modal GPU.",
    flow: ["frames", "*MapAnything", "COLMAP (metric)", "gsplat", "scene.ply"],
  },
  {
    title: "Objects",
    text: "Muse Wonder watches the video and names the objects. SAM 3.1 tracks each noun through the clip. Masks, cameras and rendered depth vote each Gaussian into an object, and each object is trained on its own.",
    flow: ["*Muse Wonder", "nouns", "*SAM 3.1", "instance masks", "ID vote", "obj_<id>.ply"],
  },
  {
    title: "Completion and coverage",
    text: "SAM 3D Objects completes every object from its best view, so it can be turned around. A gap map scores what the video saw too little; small gaps are repaired with Muse Image and distilled back, tagged as generated.",
    flow: ["*SAM 3D Objects", "gap map", "*Muse Image", "generated = 1"],
  },
  {
    title: "Made to run smoothly",
    text: "Floor, walls and ceiling become baked, opaque textured planes. Every object gets its own level-of-detail tree, and one joint LOD cut across the room spends the detail on what you are looking at.",
    flow: ["plane fit", "baked quads", "Spark LOD trees", "joint cut", "frame budget"],
  },
  {
    title: "Talk to the room",
    text: "Speech is transcribed, a Muse Wonder agent turns it into scene edits, the viewer updates live, and Muse Spark Open looks at a screenshot to accept the edit or ask for a retry.",
    flow: ["voice", "*Muse Voice Transcribe", "*Muse Wonder agent", "scene.json", "*Muse Spark Open"],
  },
];

const MODELS: [string, string, string][] = [
  ["Muse Wonder", "muse-wonder-1.0", "Agent brain with tool calls; object inventory from the video"],
  ["SAM 3.1", "sam-3.1", "Video segmentation and tracking of every object"],
  ["MapAnything", "facebook/map-anything-apache", "Metric cameras and geometry from the frames"],
  ["SAM 3D Objects", "open weights", "Completing objects and their unseen backs"],
  ["Muse Image", "muse-image-1.0", "Repairing coverage gaps; furniture to insert"],
  ["Muse Spark Open", "muse-spark-open-1026", "Visual verifier after every edit"],
  ["Muse Voice Transcribe", "muse-voice-transcribe-1.0", "Voice commands"],
  ["Muse Spark", "muse-spark-1.3", "Short helper calls, typical object heights for scale"],
];

const flowHtml = (flow: string[]) =>
  flow
    .map((f) => (f.startsWith("*") ? `<span class="lp-model">${f.slice(1)}</span>` : `<span>${f}</span>`))
    .join("<i>→</i>");

const PAGE = `
<header class="lp-nav">
  <div class="lp-wrap">
    <a class="lp-brand" href="#top"><span class="lp-mark">M</span>Movable Reality</a>
    <nav class="lp-links" aria-label="Sections">
      <a href="#what">What it does</a>
      <a href="#pipeline">Pipeline</a>
      <a href="#models">Models</a>
      <a href="#captures">Captures</a>
      <a href="${REPO}" target="_blank" rel="noopener">GitHub</a>
    </nav>
    <button class="lp-btn lp-btn-filled lp-btn-small" data-open-demo>Open the demo</button>
  </div>
</header>

<main id="top">
  <section class="lp-hero">
    <div class="lp-hero-media" aria-hidden="true"></div>
    <div class="lp-wrap lp-hero-body">
      <span class="lp-eyebrow">Meta Global AI Developer Hackathon &middot; Push the Frontier</span>
      <h1>Try the new layout <em>before you lift the sofa.</em></h1>
      <p class="lp-lede">Walk around your room once with your phone. What comes back is a 3D copy of
      it, true to scale, where the sofa, the lamp and every chair come loose. Drag things around, or
      just say what should go where.</p>
      <div class="lp-ctas">
        <button class="lp-btn lp-btn-filled" data-open-demo>${I.play}Open the demo</button>
        <a class="lp-btn lp-btn-tonal" href="#pipeline">${I.down}How it works</a>
      </div>
      <div class="lp-stats">
        <div><b>1 walk</b>with a phone</div>
        <div><b>Every object</b>moves on its own</div>
        <div><b>8</b>Meta models</div>
        <div><b>0 installs</b>runs in the browser</div>
      </div>
    </div>
    <button class="lp-motion" hidden aria-label="Pause the background video"></button>
  </section>

  <section class="lp-section" id="what">
    <div class="lp-wrap">
      <p class="lp-kicker">What it does</p>
      <h2 class="lp-h2">A scan you can rearrange, not just look at.</h2>
      <p class="lp-sub">Most room scans are one frozen blob. Here the room is taken apart into the
      things in it, the parts the camera missed are filled in, and the result opens in a browser.</p>
      <div class="lp-grid">
        ${FEATURES.map(([ic, h, p]) => `<article class="lp-card"><div class="lp-icon">${ic}</div><h3>${h}</h3><p>${p}</p></article>`).join("")}
      </div>
    </div>
  </section>

  <section class="lp-section" id="pipeline">
    <div class="lp-wrap">
      <p class="lp-kicker">How it is done</p>
      <h2 class="lp-h2">From video to a movable room in six stages.</h2>
      <p class="lp-sub">One command runs every stage, resumes where it stopped and skips work that is
      already done. Highlighted steps call a Meta model; the rest is geometry and rendering code.</p>
      <ol class="lp-steps">
        ${STEPS.map((s) => `<li class="lp-step"><div class="lp-step-n"></div><div><h3>${s.title}</h3><p>${s.text}</p><div class="lp-flow">${flowHtml(s.flow)}</div></div></li>`).join("")}
      </ol>
    </div>
  </section>

  <section class="lp-section" id="models">
    <div class="lp-wrap">
      <p class="lp-kicker">Meta models</p>
      <h2 class="lp-h2">Every model it calls at runtime is a Meta model.</h2>
      <p class="lp-sub">Hosted models run through the Meta Model API; open weights run on Modal and Baseten.
      Rendering uses open-source code with no learned weights: gsplat, Spark and three.js.</p>
      <div class="lp-models">
        ${MODELS.map(([n, id, job]) => `<div class="lp-model-card"><b>${n}</b><small>${id}</small><span>${job}</span></div>`).join("")}
      </div>
    </div>
  </section>

  <section class="lp-section" id="try">
    <div class="lp-wrap">
      <p class="lp-kicker">Try it</p>
      <h2 class="lp-h2">Three rooms to rearrange.</h2>
      <p class="lp-sub">Two synthetic rooms with ground truth, used to measure the pipeline, and a real
      room filmed on a phone, coming soon. Mouse and keyboard on a laptop, touch on a phone.</p>
      <div class="lp-scenes" data-scenes></div>
    </div>
  </section>

  <section class="lp-section" id="captures">
    <div class="lp-wrap">
      <p class="lp-kicker">The input</p>
      <h2 class="lp-h2">The videos each room was built from.</h2>
      <p class="lp-sub">Nothing else goes in: no depth sensor, no floor plan, no hand-labelled objects.
      These are the exact walks the pipeline trained on.</p>
      <div class="lp-captures" data-captures></div>
    </div>
  </section>
</main>

<footer class="lp-footer">
  <div class="lp-wrap">
    <span>Movable Reality &middot; team Gaussian Movers &middot; built 26 Oct to 4 Nov 2026</span>
    <span><a href="${REPO}" target="_blank" rel="noopener">Source on GitHub</a></span>
  </div>
</footer>

<dialog class="lp-dialog" aria-labelledby="lp-dialog-title">
  <div class="lp-dialog-inner">
    <div class="lp-dialog-head">
      <div>
        <h2 id="lp-dialog-title">Choose a room</h2>
        <p>Each opens in the viewer. Click an object to select it, drag it to move it, or type a command.</p>
      </div>
      <button class="lp-close" aria-label="Close">${I.close}</button>
    </div>
    <div class="lp-scenes" data-scenes></div>
    <div class="lp-devices">
      <span>Desktop: drag to look, W A S D to walk</span>
      <span>Phone: one finger looks, two fingers move</span>
    </div>
  </div>
</dialog>
`;

const STATE_TEXT = { ready: "", updating: "Updating", processing: "Processing", missing: "Coming soon" };

function sceneCard(s: DemoScene, base: string | null): HTMLAnchorElement {
  const a = document.createElement("a");
  a.className = "lp-scene";
  a.href = viewerUrl(s.id, base);
  a.dataset.scene = s.id;
  const img = document.createElement("div");
  img.className = "lp-scene-img";
  const pic = document.createElement("img");
  pic.alt = "";
  pic.loading = "lazy";
  pic.decoding = "async";
  // Card image, then a frame of the capture, then the gradient behind it.
  const sources = [s.thumb, `/scenes/${s.id}/frames/000040.jpg`];
  pic.addEventListener("error", () => {
    const next = sources.shift();
    if (next) pic.src = next;
    else pic.remove();
  });
  pic.src = sources.shift()!;
  const kind = document.createElement("span");
  kind.className = "lp-chip";
  kind.dataset.kind = s.kind;
  kind.textContent = s.kind === "real" ? "Real capture" : "Synthetic, with ground truth";
  const state = document.createElement("span");
  state.className = "lp-chip lp-chip-state";
  state.dataset.state = "ready";
  img.append(pic, kind, state);
  const body = document.createElement("div");
  body.className = "lp-scene-body";
  const t = document.createElement("b");
  t.textContent = s.title;
  const p = document.createElement("span");
  p.textContent = s.blurb;
  body.append(t, p);
  a.append(img, body);
  return a;
}

function captureFigure(s: DemoScene): HTMLElement {
  const fig = document.createElement("figure");
  fig.className = "lp-capture";
  const video = document.createElement("video");
  video.src = s.capture;
  video.poster = s.thumb;
  video.controls = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.setAttribute("aria-label", `Input video of the ${s.title.toLowerCase()}`);
  // No capture yet (the real room before it is filmed): a placeholder instead of a broken player.
  video.addEventListener("error", () => {
    const empty = document.createElement("div");
    empty.className = "lp-capture-empty";
    empty.textContent = "Capture coming soon";
    video.replaceWith(empty);
  });
  const cap = document.createElement("figcaption");
  const t = document.createElement("b");
  t.textContent = s.title;
  const note = document.createElement("span");
  note.textContent = s.captureNote;
  cap.append(t, note);
  fig.append(video, cap);
  return fig;
}

function heroMedia(root: HTMLElement): void {
  const media = root.querySelector<HTMLElement>(".lp-hero-media")!;
  const toggle = root.querySelector<HTMLButtonElement>(".lp-motion")!;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const still = () => {
    media.replaceChildren();
    const img = document.createElement("img");
    img.alt = "";
    img.src = DEMO_SCENES[0].thumb;
    img.addEventListener("error", () => img.remove());
    media.appendChild(img);
    toggle.hidden = true;
  };
  const video = document.createElement("video");
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.autoplay = !reduced;
  video.preload = "auto";
  video.poster = DEMO_SCENES[0].thumb;
  video.setAttribute("aria-hidden", "true");
  for (const [src, type] of [["/media/showcase.webm", "video/webm"], ["/media/showcase.mp4", "video/mp4"]]) {
    const el = document.createElement("source");
    el.src = src;
    el.type = type;
    video.appendChild(el);
  }
  // The last <source> failing means no showcase video at all.
  video.lastElementChild?.addEventListener("error", still);
  media.appendChild(video);

  // HIG and WCAG 2.2.2: moving backgrounds get a pause control.
  const sync = () => {
    const paused = video.paused;
    toggle.innerHTML = paused ? I.play : I.pause;
    toggle.setAttribute("aria-label", paused ? "Play the background video" : "Pause the background video");
  };
  video.addEventListener("loadeddata", () => (toggle.hidden = false));
  video.addEventListener("play", sync);
  video.addEventListener("pause", sync);
  toggle.addEventListener("click", () => (video.paused ? void video.play() : video.pause()));
  sync();
}

export function installLanding(): void {
  document.documentElement.classList.add("lp");
  document.title = "Movable Reality: rearrange a room you filmed";
  const style = document.createElement("style");
  style.textContent = LANDING_CSS;
  document.head.appendChild(style);
  document.getElementById("hud")?.remove();

  const root = document.createElement("div");
  root.className = "lp";
  root.innerHTML = PAGE;
  document.body.appendChild(root);

  const base = new URLSearchParams(location.search).get("base");
  const cards: HTMLAnchorElement[] = [];
  for (const holder of root.querySelectorAll<HTMLElement>("[data-scenes]")) {
    for (const s of DEMO_SCENES) {
      const card = sceneCard(s, base);
      cards.push(card);
      holder.appendChild(card);
    }
  }
  root.querySelector("[data-captures]")!.append(...DEMO_SCENES.map(captureFigure));
  void sceneAvailability(DEMO_SCENES.map((s) => s.id), base ?? "/scenes").then((avail) => {
    for (const card of cards) {
      const st = avail.get(card.dataset.scene!) ?? "missing";
      const chip = card.querySelector<HTMLElement>(".lp-chip-state")!;
      chip.dataset.state = st;
      chip.textContent = STATE_TEXT[st];
      const closed = st === "processing" || st === "missing";
      card.setAttribute("aria-disabled", String(closed));
      if (closed) {
        card.removeAttribute("href");
        card.setAttribute("role", "link"); // an <a> without href has no role for aria-disabled
        card.tabIndex = -1;
      }
    }
  });

  const dialog = root.querySelector<HTMLDialogElement>(".lp-dialog")!;
  for (const b of root.querySelectorAll("[data-open-demo]")) {
    b.addEventListener("click", () => dialog.showModal());
  }
  root.querySelector(".lp-close")!.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close(); // a click on the backdrop
  });

  const nav = root.querySelector<HTMLElement>(".lp-nav")!;
  const onScroll = () => nav.toggleAttribute("data-solid", window.scrollY > 24);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  heroMedia(root);
}
