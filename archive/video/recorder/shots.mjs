import { launch, sleep, OUT } from "./lib.mjs";
import { appendFileSync } from "node:fs";
const URLV = "http://localhost:5173/?scene=blender07&onboarding=0&hud=0&budget=0";
const only = process.argv.slice(2);
const log = (...a) => { console.log(...a); appendFileSync(OUT + "/shots.log", a.join(" ") + "\n"); };
const { tab, close } = await launch();
const J = JSON.stringify;
const fresh = async (settle = 12000) => {
  await tab.cdp("Network.enable");
  await tab.cdp("Network.clearBrowserCookies");
  await tab.goto(URLV);
  await tab.ready(settle);
  tab.x = 960; tab.y = 540;
};
const pose = (e, t) => tab.js(`__pose(${J(e)},${J(t)})`);
const glide = (e, t, ms) => tab.js(`__glide(${J(e)},${J(t)},${ms})`);
const scr = (p) => tab.js(`__screen(${J(p)})`);
const hideUi = (on) => tab.js(`(()=>{let s=document.getElementById('__h');if(!s){s=document.createElement('style');s.id='__h';document.head.append(s)};s.textContent=${on}?'.mr-ui-bar,[class*="mr-chrome"]{display:none!important}':''})()`);
const center = (id) => tab.js(`(()=>{const o=__obj(${J(id)});const t=o.transform||{};const d=t.translation||[0,0,0];return o.bbox_min.map((m,i)=>(m+o.bbox_max[i])/2+(d[i]||0))})()`);
const objT = (id) => tab.js(`JSON.stringify(__obj(${J(id)}).transform)`);
// TransformControls handle scale in world units at this camera (fov 50, size 0.9).
const hscale = (p) => tab.js(`(()=>{const e=__cam.eye,p=${J(p)};return Math.hypot(e[0]-p[0],e[1]-p[1],e[2]-p[2])*1.9*Math.tan(25*Math.PI/180)*0.9/4})()`);
const clip = async (name, fn) => {
  if (only.length && !only.includes(name)) return;
  log("==", name);
  try { await fn(); } catch (e) { log("FAILED", name, e.message); try { await tab.shot(name + "_fail.jpg"); } catch {} }
};
const rec = async (name, fn, lead = 800, tail = 1500) => {
  const stop = await tab.record(name);
  await sleep(lead);
  await fn();
  await sleep(tail);
  await tab.shot(name + ".jpg");
  await stop();
};
const CORNER = [[3.6, 1.65, -3.7], [1.2, 0.6, -0.3]];
const WIDE = [[0.4, 1.5, -3.6], [2.2, 0.5, 0.6]];

try {
  await clip("drag_armchair", async () => {
    await fresh();
    await pose([3.5, 1.6, -3.4], [0.3, 0.5, -1.5]); await sleep(4000);
    await rec("drag_armchair", async () => {
      const c = await center("armchair#1");
      const a = await scr([c[0], 0.5, c[2]]), b = await scr([c[0] + 0.9, 0.5, c[2] - 0.6]);
      await tab.drag(a[0], a[1], b[0], b[1], 2200);
      await sleep(1800);
      await tab.key("Escape", "Escape");
      await sleep(1500);
    });
    log(await objT("armchair#1"));
  });
  await clip("gizmo_ottoman", async () => {
    await fresh();
    await pose([1.0, 1.5, -0.9], [2.6, 0.35, -3.0]); await sleep(4000);
    await rec("gizmo_ottoman", async () => {
      const c = await center("ottoman#0");
      const a = await scr(c);
      await tab.click(a[0], a[1], 900);
      await sleep(900);
      const s = await hscale(c);
      const h = await scr([c[0] + 0.45 * s, c[1], c[2]]);
      const g = await scr([c[0] + 0.45 * s - 0.9, c[1], c[2]]);
      await tab.drag(h[0], h[1], g[0], g[1], 2000);
      await sleep(1500);
      await tab.key("Escape", "Escape");
      await sleep(1200);
    });
    log(await objT("ottoman#0"));
  });
  await clip("gizmo_rotate", async () => {
    await fresh();
    await pose([2.3, 1.5, -3.2], [0.2, 0.45, -1.7]); await sleep(4000);
    await rec("gizmo_rotate", async () => {
      const c = await center("armchair#1");
      const a = await scr(c);
      await tab.click(a[0], a[1], 900);
      await sleep(600);
      const btn = await tab.js(`(()=>{const b=[...document.querySelectorAll('.mr-ui-gizmo button')].find(x=>x.textContent==='Rotate').getBoundingClientRect();return [b.x+b.width/2,b.y+b.height/2]})()`);
      await tab.click(btn[0], btn[1], 700);
      await sleep(700);
      const s = await hscale(c), r = 0.5 * s;
      const at = (deg) => scr([c[0] + r * Math.cos(deg * Math.PI / 180), c[1], c[2] + r * Math.sin(deg * Math.PI / 180)]);
      const p0 = await at(60);
      await tab.moveTo(p0[0], p0[1]); await sleep(300);
      await tab.mouse("mousePressed", p0[0], p0[1], "left", 1);
      for (let d = 60; d >= -20; d -= 4) { const p = await at(d); await tab.moveTo(p[0], p[1], 90, 1, "left"); }
      const pe = await at(-20);
      await tab.mouse("mouseReleased", pe[0], pe[1], "left", 0);
      await sleep(1400);
      await tab.key("Escape", "Escape");
      await sleep(1000);
    });
    log(await objT("armchair#1"));
  });
  await clip("remove_undo", async () => {
    await fresh();
    await pose(...CORNER); await sleep(4000);
    await rec("remove_undo", async () => {
      const c = await center("coffee table#0");
      const a = await scr([c[0], 0.5, c[2]]);
      await tab.click(a[0], a[1], 900);
      await sleep(900);
      await tab.key("Delete", "Delete");
      await sleep(2200);
      await tab.key("KeyZ", "z", 2);
      await sleep(1800);
      await tab.key("Escape", "Escape");
    });
  });
  await clip("measure", async () => {
    await fresh();
    await pose(...WIDE); await sleep(4000);
    await rec("measure", async () => {
      const a = await scr([0.78, 0.35, 1.0]), b = await scr([3.26, 0.35, 1.0]);
      for (const p of [a, b]) {
        await tab.moveTo(p[0], p[1], 900); await sleep(200);
        await tab.mouse("mousePressed", p[0], p[1], "left", 1, { modifiers: 1 });
        await sleep(70);
        await tab.mouse("mouseReleased", p[0], p[1], "left", 0, { modifiers: 1 });
        await sleep(900);
      }
      await sleep(1500);
    });
  });
  await clip("library_add", async () => {
    await fresh();
    await pose(...WIDE); await sleep(4000);
    await rec("library_add", async () => {
      const q = (sel) => tab.js(`(()=>{const b=${sel}.getBoundingClientRect();return [b.x+b.width/2,b.y+b.height/2]})()`);
      const plus = await q(`document.querySelector('.mr-chrome-icon')`);
      await tab.click(plus[0], plus[1], 900);
      await sleep(1400);
      const card = await q(`[...document.querySelectorAll('.mr-ui-lib-card')].find(c=>/plant/i.test(c.title))`);
      await tab.click(card[0], card[1], 900);
      await sleep(2500);
      await tab.click(plus[0], plus[1], 700);
      await sleep(1000);
      const id = await tab.js(`mrUi.manifest().objects.at(-1).id`);
      log("added", id);
      const c = await center(id);
      const a = await scr([c[0], Math.max(0.25, c[1]), c[2]]), b = await scr([0.35, 0.3, 0.9]);
      await tab.drag(a[0], a[1], b[0], b[1], 2200);
      await sleep(1500);
      await tab.key("Escape", "Escape");
    });
  });
  const say = (name, text, p = WIDE) => clip(name, async () => {
    await fresh();
    await pose(...p); await sleep(4000);
    await rec(name, async () => {
      const r = await tab.command(text);
      log(name, "seconds", r.seconds);
      log(await tab.js(`[...document.querySelectorAll('.mr-ui-log > *')].map(x=>x.textContent).join(' | ')`));
      await sleep(3000);
    });
  });
  await say("agent_move", "move the sofa half a metre towards the window");
  await say("agent_add", "add a potted plant to the right of the sofa");
  await say("agent_fit", "does the armchair fit between the sofa and the bookshelf?");
  await say("agent_generate", "add a small wooden stool in front of the sofa");
  await clip("gaps", async () => {
    await fresh();
    await pose(...CORNER); await sleep(3000);
    await rec("gaps", async () => {
      await tab.key("KeyG", "g");
      await sleep(1500);
      await glide(WIDE[0], WIDE[1], 7000);
      await sleep(1000);
      await tab.key("KeyG", "g");
    });
  });
  await clip("glide", async () => {
    await fresh(20000);
    await hideUi(true);
    await pose(...CORNER); await sleep(5000);
    await rec("glide", async () => {
      await glide([3.2, 1.55, -1.8], [0.9, 0.7, -0.1], 7000);
      await glide([2.2, 1.5, -0.9], [0.0, 0.9, -3.0], 7000);
      await glide([1.6, 1.5, -1.0], [1.8, 0.9, -4.6], 6000);
      await glide([0.4, 1.5, -3.6], [2.2, 0.5, 0.6], 8000);
    });
    await hideUi(false);
  });
} finally {
  close();
}
