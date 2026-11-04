// Landing page at / (entry.ts loads it when the URL names no scene). A showcase video behind
// the hero, the demo rooms, what it does, how it is built and how to build on it. Clicking a room
// opens its room card first: the videos the pipeline was given and how the room was made, then
// a button into the viewer, so people see how little went in before they see what came out.
// Copy: say what it does, and that it is a working demo on synthetic rooms, once each.
// The video is web/public/media/showcase.mp4 (gitignored like every .mp4: copy it next to the
// build when hosting); without it the hero shows a still.
import { DEMO_SCENES, type DemoScene, sceneAvailability, type SourceVideo, viewerUrl } from "./scenes";
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
  [I.video, "One video in", "Give it one walk through a room. MapAnything works out where the camera was, and gsplat trains the 3D scene from the frames. No depth sensor, no floor plan."],
  [I.cube, "The furniture comes apart", "Muse Wonder names what it sees, SAM 3.1 follows each item through the video, and every splat goes to the item it belongs to. The sofa ends up as its own object, in its own file."],
  [I.mic, "Say what you want", "Type “move the sofa under the window”. A Muse Wonder agent edits the scene, then Muse Spark Open checks a screenshot and sends it back if the edit is wrong."],
  [I.sparkle, "It knows what it missed", "No walk sees everything. Coverage scoring marks the spots the camera barely saw, and SAM 3D Objects builds the side of each object it never saw."],
  [I.devices, "Opens in a browser", "Nothing to install. Mouse and keyboard on a laptop, touch on a phone. Each object has its own level of detail, so a weaker machine gets a lighter version of the same room."],
  [I.ruler, "Measure", "Alt+click two points to read the distance. See how much space a sofa takes, move it, and undo any edit you don’t like."],
];

interface Step {
  title: string;
  text: string;
  flow: string[]; // "*Name" marks a Meta model
}

const STEPS: Step[] = [
  {
    title: "Capture",
    text: "One walk through the room. We keep the sharpest 4 frames per second, and those same frames, in the same order, become the clip that SAM tracks.",
    flow: ["video", "sharpest frames", "SAM clip"],
  },
  {
    title: "Cameras and 3D scene",
    text: "MapAnything predicts the cameras and geometry, and COLMAP refines them. gsplat then trains a 3D Gaussian Splatting scene on the full-size frames, on a Modal GPU.",
    flow: ["frames", "*MapAnything", "COLMAP", "gsplat", "scene.ply"],
  },
  {
    title: "Objects",
    text: "Muse Wonder watches the video and names the objects. SAM 3.1 tracks each one through the clip. Masks, cameras and rendered depth decide which object each splat belongs to, and each object is then trained on its own.",
    flow: ["*Muse Wonder", "nouns", "*SAM 3.1", "instance masks", "ID vote", "obj_<id>.ply"],
  },
  {
    title: "Hidden sides and thin spots",
    text: "SAM 3D Objects rebuilds each object from its best view, including the side the camera never saw. A coverage map scores what the video barely saw. A repair path paints thin spots with Muse Image and tags what it painted. So far it has run on test scenes.",
    flow: ["*SAM 3D Objects", "coverage map", "*Muse Image"],
  },
  {
    title: "Fit it in a browser",
    text: "Floor, walls and ceiling become flat textured surfaces instead of splats. Every object gets its own level-of-detail tree, and the viewer spends its splat budget on what you are looking at.",
    flow: ["plane fit", "textured surfaces", "Spark LOD trees", "frame budget"],
  },
  {
    title: "Talk to the room",
    text: "A typed or spoken command goes to a Muse Wonder agent, which edits the scene file. The viewer updates, and Muse Spark Open looks at a screenshot to accept the edit or ask for a retry.",
    flow: ["voice", "*Muse Voice Transcribe", "*Muse Wonder agent", "scene.json", "*Muse Spark Open"],
  },
];

const MODELS: [string, string, string][] = [
  ["Muse Wonder", "muse-wonder-1.0", "Lists the objects in the video and edits the scene"],
  ["SAM 3.1", "sam-3.1", "Finds and follows every object through the video"],
  ["MapAnything", "facebook/map-anything-apache", "Cameras and geometry from the frames"],
  ["SAM 3D Objects", "open weights", "Builds the hidden side of each object"],
  ["Muse Image", "muse-image-1.0", "New furniture to insert, and repairs to thin spots"],
  ["Muse Spark Open", "muse-spark-open-1026", "Looks at a screenshot after every edit"],
  ["Muse Voice Transcribe", "muse-voice-transcribe-1.0", "Voice commands"],
  ["Muse Spark", "muse-spark-1.3", "Short helper calls"],
];

const flowHtml = (flow: string[]) =>
  flow
    .map((f) => (f.startsWith("*") ? `<span class="lp-model">${f.slice(1)}</span>` : `<span>${f}</span>`))
    .join("<i>→</i>");

const PAGE = `
<header class="lp-nav">
  <div class="lp-wrap">
    <a class="lp-brand" href="#top">Movable Reality</a>
    <nav class="lp-links" aria-label="Sections">
      <a href="#try">Rooms</a>
      <a href="#what">What it does</a>
      <a href="#pipeline">Pipeline</a>
      <a href="#models">Models</a>
      <a href="${REPO}" target="_blank" rel="noopener">GitHub</a>
    </nav>
    <a class="lp-btn lp-btn-filled lp-btn-small" href="#try">Open the demo</a>
  </div>
</header>

<main id="top">
  <section class="lp-hero">
    <div class="lp-hero-media" aria-hidden="true"></div>
    <div class="lp-wrap lp-hero-body">
      <span class="lp-eyebrow">Working demo &middot; Meta Global AI Developer Hackathon</span>
      <h1>Film a room. <em>Move the furniture.</em></h1>
      <p class="lp-lede">Movable Reality turns one walk through a room into a 3D scene where every
      piece of furniture is its own object. Drag the sofa across the room, or just ask for it. No depth
      sensor, no floor plan, no hand labelling. Open source, and running today on two rooms rendered in
      Blender.</p>
      <div class="lp-ctas">
        <a class="lp-btn lp-btn-filled" href="#try">${I.play}Open the demo</a>
      </div>
      <div class="lp-stats">
        <div><b>8</b>Meta models</div>
        <div><b>49</b>movable objects</div>
        <div><b>0</b>hand labels</div>
      </div>
      <p class="lp-hero-note">Background: the input video.</p>
    </div>
    <button class="lp-motion" hidden aria-label="Pause the background video"></button>
  </section>

  <section class="lp-section" id="try">
    <div class="lp-wrap">
      <p class="lp-kicker">The demo</p>
      <h2 class="lp-h2">Two rooms, ready to open.</h2>
      <p class="lp-sub">See the video each room was built from, then open it and start moving the furniture.</p>
      <div class="lp-scenes" data-scenes></div>
    </div>
  </section>

  <section class="lp-section" id="what">
    <div class="lp-wrap">
      <p class="lp-kicker">What it does</p>
      <h2 class="lp-h2">A room you can edit.</h2>
      <div class="lp-grid">
        ${FEATURES.map(([ic, h, p]) => `<article class="lp-card"><div class="lp-icon">${ic}</div><h3>${h}</h3><p>${p}</p></article>`).join("")}
      </div>
    </div>
  </section>

  <section class="lp-section" id="captures">
    <div class="lp-wrap">
      <p class="lp-kicker">The input</p>
      <h2 class="lp-h2">Built from these videos.</h2>
      <p class="lp-sub">The videos, the camera’s focal length and the ceiling height are the whole
      input. No depth, no floor plan, no hand labels. Blender’s own depth and masks were only
      used afterwards, to score the result.</p>
      <div class="lp-captures" data-captures></div>
    </div>
  </section>

  <section class="lp-section" id="pipeline">
    <div class="lp-wrap">
      <p class="lp-kicker">How it is done</p>
      <h2 class="lp-h2">From video to movable room.</h2>
      <p class="lp-sub">One command runs all of it, picks up where it stopped and skips what is already
      done. Highlighted steps call a Meta model, and the rest is geometry and rendering code.</p>
      <ol class="lp-steps">
        ${STEPS.map((s) => `<li class="lp-step"><div class="lp-step-n"></div><div><h3>${s.title}</h3><p>${s.text}</p><div class="lp-flow">${flowHtml(s.flow)}</div></div></li>`).join("")}
      </ol>
    </div>
  </section>

  <section class="lp-section" id="models">
    <div class="lp-wrap">
      <p class="lp-kicker">Meta models</p>
      <h2 class="lp-h2">Every model it calls at runtime is from Meta.</h2>
      <p class="lp-sub">Hosted models run through the Meta Model API, and open weights run on Modal and Baseten.
      Rendering is open-source code with no learned weights: gsplat, Spark and three.js.</p>
      <div class="lp-models">
        ${MODELS.map(([n, id, job]) => `<div class="lp-model-card"><b>${n}</b><small>${id}</small><span>${job}</span></div>`).join("")}
      </div>
    </div>
  </section>

  <section class="lp-section" id="build">
    <div class="lp-wrap">
      <p class="lp-kicker">Build on it</p>
      <h2 class="lp-h2">A working demo, and a good place to start.</h2>
      <p class="lp-sub">It runs end to end on two synthetic rooms. If your research or project touches
      object-level splats, room editing or agents that act on 3D scenes, you can start here instead of
      rebuilding all of this. It is MIT licensed. Each stage is a plain script
      you can rerun alone or swap out. Objects come out as separate splat files with one JSON scene file,
      and the Blender rooms and scoring tools are included, so you can measure every change.</p>
      <a class="lp-btn lp-btn-tonal" href="${REPO}" target="_blank" rel="noopener">Source on GitHub</a>
    </div>
  </section>
</main>

<footer class="lp-footer">
  <div class="lp-wrap">
    <span>Movable Reality &middot; team Larracat &middot; built 26 Oct to 4 Nov 2026</span>
    <span><a href="${REPO}" target="_blank" rel="noopener">Source on GitHub</a></span>
  </div>
</footer>

<dialog class="lp-dialog lp-room" aria-labelledby="lp-room-title">
  <div class="lp-dialog-inner">
    <div class="lp-dialog-head">
      <div>
        <h2 id="lp-room-title"></h2>
        <p data-room-blurb></p>
      </div>
      <button class="lp-close" aria-label="Close">${I.close}</button>
    </div>
    <div class="lp-room-body">
      <div class="lp-room-input">
        <h3>What went in</h3>
        <div data-room-videos></div>
        <p class="lp-room-note">That is the whole input, plus the focal length and ceiling height. Blender’s
        depth and masks were only used afterwards, to score the result.</p>
      </div>
      <div class="lp-room-made">
        <h3>What came out</h3>
        <dl data-room-facts></dl>
        <h3>How it was made</h3>
        <ol class="lp-room-steps">
          <li>Blender renders the walk through the room.</li>
          <li>The sharpest frames go to MapAnything and COLMAP for the cameras, then gsplat trains the splats.</li>
          <li>Muse Wonder names the objects, SAM 3.1 tracks them, and each object is trained on its own.</li>
          <li>SAM 3D Objects builds hidden sides. Walls, floor and ceiling become flat textured surfaces.</li>
          <li>Each object gets levels of detail, so the room can load in a browser.</li>
        </ol>
      </div>
    </div>
    <div class="lp-room-foot">
      <a class="lp-btn lp-btn-filled" data-room-open>${I.play}Open the room</a>
      <span class="lp-devices">
        <span>Laptop: drag to orbit, click to select</span>
        <span>Phone: one finger looks, two fingers move</span>
      </span>
    </div>
  </div>
</dialog>
`;

const STATE_TEXT = { ready: "", updating: "Updating", processing: "Processing", missing: "Not available" };

function sceneCard(s: DemoScene, openRoom: (s: DemoScene) => void): HTMLButtonElement {
  const a = document.createElement("button");
  a.type = "button";
  a.className = "lp-scene";
  a.dataset.scene = s.id;
  a.addEventListener("click", () => openRoom(s));
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
  const state = document.createElement("span");
  state.className = "lp-chip lp-chip-state";
  state.dataset.state = "ready";
  img.append(pic, state);
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

function sourceFigure(s: DemoScene, v: SourceVideo, opts: { title?: boolean; autoplay?: boolean } = {}): HTMLElement {
  const fig = document.createElement("figure");
  fig.className = "lp-capture";
  const video = document.createElement("video");
  video.src = v.src;
  video.poster = s.thumb;
  video.controls = true;
  video.muted = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.setAttribute("aria-label", `${v.label}: input video of the ${s.title.toLowerCase()}`);
  if (opts.autoplay && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    video.autoplay = true;
    video.loop = true;
  }
  // A missing file (copied next to the build by hand): a placeholder instead of a broken player.
  video.addEventListener("error", () => {
    const empty = document.createElement("div");
    empty.className = "lp-capture-empty";
    empty.textContent = "Video not uploaded";
    video.replaceWith(empty);
  });
  const cap = document.createElement("figcaption");
  const t = document.createElement("b");
  t.textContent = opts.title ? `${s.title}: ${v.label.toLowerCase()}` : v.label;
  const note = document.createElement("span");
  note.textContent = v.note;
  cap.append(t, note);
  fig.append(video, cap);
  return fig;
}

/** The room card: fills the one dialog for a room. Closing it removes the videos, so they stop. */
function roomCard(root: HTMLElement, base: string | null): (s: DemoScene) => void {
  const dialog = root.querySelector<HTMLDialogElement>(".lp-room")!;
  const videos = dialog.querySelector<HTMLElement>("[data-room-videos]")!;
  const facts = dialog.querySelector<HTMLElement>("[data-room-facts]")!;
  dialog.querySelector(".lp-close")!.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) dialog.close(); // a click on the backdrop
  });
  dialog.addEventListener("close", () => videos.replaceChildren());
  return (s) => {
    dialog.querySelector("#lp-room-title")!.textContent = s.title;
    dialog.querySelector("[data-room-blurb]")!.textContent = s.blurb;
    dialog.querySelector<HTMLAnchorElement>("[data-room-open]")!.href = viewerUrl(s.id, base);
    videos.replaceChildren(...s.sources.map((v, i) => sourceFigure(s, v, { autoplay: i === 0 })));
    facts.replaceChildren(
      ...s.facts.flatMap(([k, v]) => {
        const dt = document.createElement("dt");
        const dd = document.createElement("dd");
        dt.textContent = k;
        dd.textContent = v;
        return [dt, dd];
      }),
    );
    dialog.showModal();
    dialog.scrollTop = 0;
  };
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
  document.title = "Movable Reality: move the furniture in a room from one video";
  const style = document.createElement("style");
  style.textContent = LANDING_CSS;
  document.head.appendChild(style);
  document.getElementById("hud")?.remove();

  const root = document.createElement("div");
  root.className = "lp";
  root.innerHTML = PAGE;
  document.body.appendChild(root);

  const base = new URLSearchParams(location.search).get("base");
  const openRoom = roomCard(root, base);
  const cards: HTMLButtonElement[] = [];
  for (const holder of root.querySelectorAll<HTMLElement>("[data-scenes]")) {
    for (const s of DEMO_SCENES) {
      const card = sceneCard(s, openRoom);
      cards.push(card);
      holder.appendChild(card);
    }
  }
  root.querySelector("[data-captures]")!.append(...DEMO_SCENES.flatMap((s) => s.sources.map((v) => sourceFigure(s, v, { title: true }))));
  void sceneAvailability(DEMO_SCENES.map((s) => s.id), base ?? "/scenes").then((avail) => {
    for (const card of cards) {
      const st = avail.get(card.dataset.scene!) ?? "missing";
      const chip = card.querySelector<HTMLElement>(".lp-chip-state")!;
      chip.dataset.state = st;
      chip.textContent = STATE_TEXT[st];
      card.disabled = st === "processing" || st === "missing";
    }
  });

  const nav = root.querySelector<HTMLElement>(".lp-nav")!;
  const onScroll = () => nav.toggleAttribute("data-solid", window.scrollY > 24);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  heroMedia(root);
}
