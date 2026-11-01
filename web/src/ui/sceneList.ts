// The /scenes/ page (served by the dev server, vite.config.ts): one card per scene with a
// scene.json under data/scenes or web/public/scenes, and links to the views it supports.
import { type SceneSummary, sceneLinks } from "./sceneIndex";

const CSS = `
:root { color-scheme: dark; --bg:#111; --card:#1c1c1c; --line:#333; --text:#eee; --dim:#999; --accent:#f5c542; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.4 system-ui, sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: var(--dim); margin: 0 0 20px; }
.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
.card h2 { font-size: 17px; margin: 0 0 6px; }
.card h2 a { color: var(--text); text-decoration: none; }
.facts { color: var(--dim); font-size: 13px; margin: 0 0 10px; }
.links { display: flex; flex-wrap: wrap; gap: 6px; }
.links a { color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 4px 9px;
  font-size: 13px; text-decoration: none; }
.links a:first-child { border-color: var(--accent); color: var(--accent); }
.links a:hover { background: #2a2a2a; }
.empty { color: var(--dim); }
code { background: #222; padding: 1px 5px; border-radius: 4px; }
`;

function facts(s: SceneSummary): string {
  const parts = [`${s.objects} objects`];
  if (s.objects !== s.visible) parts.push(`${s.objects - s.visible} hidden`);
  if (s.generated) parts.push(`${s.generated} generated`);
  if (s.edits) parts.push(`${s.edits} edits`);
  parts.push(s.lod ? "LOD" : "no LOD");
  if (s.planes) parts.push(`${s.planes} baked planes`);
  if (s.meshes) parts.push(`${s.meshes} meshes`);
  if (s.room) parts.push("room shell");
  return parts.join(" · ");
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}) {
  return Object.assign(document.createElement(tag), props);
}

async function show(): Promise<void> {
  document.title = "Scenes";
  document.head.append(el("style", { textContent: CSS }));
  const main = el("main");
  main.append(el("h1", { textContent: "Movable Reality scenes" }));
  const sub = el("p", { className: "sub", textContent: "Loading..." });
  main.append(sub);
  document.body.append(main);

  let scenes: SceneSummary[];
  try {
    const r = await fetch("/__scenes", { cache: "no-store" });
    scenes = (await r.json()) as SceneSummary[];
  } catch (e) {
    sub.textContent = `Could not list scenes: ${e}`;
    return;
  }
  sub.innerHTML =
    `${scenes.length} scene${scenes.length === 1 ? "" : "s"} with a <code>scene.json</code> ` +
    `under <code>data/scenes</code>. The first link opens the viewer.`;
  if (!scenes.length) {
    main.append(el("p", { className: "empty", textContent: "No scenes yet." }));
    return;
  }
  const grid = el("div", { className: "grid" });
  for (const s of scenes) {
    const links = sceneLinks(s);
    const card = el("section", { className: "card" });
    const h2 = el("h2");
    h2.append(el("a", { href: links[0].url, textContent: s.id }));
    card.append(h2, el("p", { className: "facts", textContent: facts(s) }));
    const row = el("div", { className: "links" });
    for (const l of links) row.append(el("a", { href: l.url, textContent: l.label, title: l.hint }));
    card.append(row);
    grid.append(card);
  }
  main.append(grid);
}

void show();
