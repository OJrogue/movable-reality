const HELPERS_REINJECT = "void 0";
import { launch, sleep, OUT } from "./lib.mjs";
import { appendFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const HERE = dirname(fileURLToPath(import.meta.url));
const URLV = "http://localhost:5173/?scene=blender07&onboarding=0&hud=0&budget=0";
const only = process.argv.slice(2);
const log = (...a) => { console.log(...a); appendFileSync(OUT + "/shots.log", a.join(" ") + "\n"); };
const { tab, close } = await launch();
await sleep(4000); // fullscreen settles
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
const center = (id) => tab.js(`(()=>{const o=__obj(${J(id)});const t=o.transform||{};const d=t.translation||[0,0,0];return o.bbox_min.map((m,i)=>(m+o.bbox_max[i])/2+(d[i]||0))})()`);
const objT = (id) => tab.js(`JSON.stringify(__obj(${J(id)}).transform)`);
const hscale = (p) => tab.js(`(()=>{const e=__cam.eye,p=${J(p)};return Math.hypot(e[0]-p[0],e[1]-p[1],e[2]-p[2])*1.9*Math.tan(25*Math.PI/180)*0.9/4})()`);
const replies = () => tab.js(`[...document.querySelectorAll('.mr-ui-log > *')].map(x=>x.textContent).join(' | ')`);
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
const WIDE = [[0.4, 1.5, -3.6], [2.2, 0.5, 0.6]];
const CHAIR = [[3.5, 1.6, -3.4], [0.3, 0.5, -1.5]];
const PLUS = [1789, 39];

try {
  await clip("landing_load", async () => {
    // Screencast, because a page recorder would not survive the navigation.
    const dir = join(OUT, "landing_frames");
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    const frames = [];
    tab.onEvent = (d) => {
      if (d.method !== "Page.screencastFrame") return;
      const f = `${String(frames.length).padStart(5, "0")}.jpg`;
      writeFileSync(join(dir, f), Buffer.from(d.params.data, "base64"));
      frames.push([f, d.params.metadata.timestamp]);
      tab.cdp("Page.screencastFrameAck", { sessionId: d.params.sessionId }).catch(() => {});
    };
    await tab.cdp("Network.enable");
    await tab.cdp("Network.clearBrowserCookies");
    await tab.goto("http://localhost:5173/");
    await sleep(2500);
    await tab.cdp("Page.startScreencast", { format: "jpeg", quality: 92, everyNthFrame: 1 });
    await sleep(2500);
    for (let i = 0; i < 16; i++) { await tab.js("scrollBy(0,90)"); await sleep(120); }
    await sleep(800);
    await tab.js("(()=>{const a=[...document.querySelectorAll('a')].find(x=>/scene=blender07/.test(x.href));a.scrollIntoView({block:'center',behavior:'smooth'})})()");
    await sleep(1800);
    await tab.js(HELPERS_REINJECT);
    const p = await tab.js("(()=>{const a=[...document.querySelectorAll('a')].find(x=>/scene=blender07/.test(x.href));const r=a.getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()");
    await tab.click(p[0], p[1], 1000);
    await sleep(1000);
    await tab.ready(5000);
    await sleep(3000);
    await tab.cdp("Page.stopScreencast");
    tab.onEvent = null;
    const lines = frames.map(([f, t], i) => `file '${f}'\nduration ${(frames[i + 1]?.[1] ?? t + 0.05) - t}`).join("\n");
    writeFileSync(join(dir, "list.txt"), lines);
    log("landing frames", frames.length);
    tab.x = 960; tab.y = 540;
  });
  await clip("drag_armchair", async () => {
    await fresh();
    await pose(...CHAIR); await sleep(4000);
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
  await clip("gizmo_move", async () => {
    await fresh();
    await pose([2.3, 1.5, -3.2], [0.2, 0.45, -1.7]); await sleep(4000);
    await rec("gizmo_move", async () => {
      const c = await center("armchair#1");
      const a = await scr(c);
      await tab.click(a[0], a[1], 900);
      await sleep(900);
      const s = await hscale(c);
      const h = await scr([c[0] + 0.42 * s, c[1], c[2]]);
      const g = await scr([c[0] + 0.42 * s + 0.8, c[1], c[2]]);
      await tab.drag(h[0], h[1], g[0], g[1], 2000);
      await sleep(1500);
      await tab.key("Escape", "Escape");
      await sleep(1200);
    });
    log(await objT("armchair#1"));
  });
  await clip("remove_undo_pillow", async () => {
    await fresh();
    await pose([1.9, 1.35, -1.6], [1.9, 0.6, 1.3]); await sleep(4000);
    await rec("remove_undo_pillow", async () => {
      const c = await center("pillow#3");
      const a = await scr([c[0], c[1], c[2]]);
      await tab.click(a[0], a[1], 900);
      await sleep(1000);
      await tab.key("Delete", "Delete");
      await sleep(1500);
      await tab.shot("removed.jpg");
      await sleep(700);
      await tab.key("KeyZ", "z", 2);
      await sleep(1800);
      await tab.key("Escape", "Escape");
    });
    log(await replies());
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
      await tab.click(PLUS[0], PLUS[1], 900);
      await sleep(1400);
      const card = await tab.js(`(()=>{const c=[...document.querySelectorAll('.mr-ui-lib-card')].find(c=>/tall potted plant|potted plant/i.test(c.title));c.scrollIntoView({block:'center'});const b=c.getBoundingClientRect();return [b.x+b.width/2,b.y+b.height/2]})()`);
      await sleep(500);
      await tab.click(card[0], card[1], 900);
      await sleep(3000);
      await tab.click(PLUS[0], PLUS[1], 700);
      await sleep(1000);
      const id = await tab.js(`mrUi.manifest().objects.at(-1).id`);
      log("added", id);
      await tab.key("Escape", "Escape");
      await sleep(400);
      const c = await center(id);
      const a = await scr([c[0], 0.5, c[2]]), b = await scr([0.35, 0.5, 0.75]);
      await tab.drag(a[0], a[1], b[0], b[1], 2200);
      await sleep(1500);
      await tab.key("Escape", "Escape");
      await sleep(800);
    });
  });
  const say = (name, text, p = WIDE) => clip(name, async () => {
    await fresh();
    await pose(...p); await sleep(4000);
    await rec(name, async () => {
      const r = await tab.command(text);
      log(name, "seconds", r.seconds);
      log(await replies());
      await sleep(3000);
    });
  });
  await say("agent_add", "add a large plant from the library next to the sofa");
  await say("agent_generate", "generate a new three-legged red stool in front of the sofa, nothing from the library");
  await clip("voice", async () => {
    await fresh();
    await pose(...WIDE); await sleep(4000);
    await rec("voice", async () => {
      const b64 = readFileSync(join(HERE, "voice_raw.wav")).toString("base64");
      await tab.js(`(()=>{const ctx=new AudioContext();const bytes=Uint8Array.from(atob("${b64}"),c=>c.charCodeAt(0)).buffer;
        navigator.mediaDevices.getUserMedia=async()=>{await ctx.resume();const buf=await ctx.decodeAudioData(bytes.slice(0));const d=ctx.createMediaStreamDestination();const n=ctx.createBufferSource();n.buffer=buf;n.connect(d);n.start(ctx.currentTime+0.25);return d.stream};})()`);
      await sleep(800);
      const base = { code: "Space", key: " ", windowsVirtualKeyCode: 32, nativeVirtualKeyCode: 32 };
      await tab.cdp("Input.dispatchKeyEvent", { type: "rawKeyDown", ...base });
      await sleep(3700);
      await tab.cdp("Input.dispatchKeyEvent", { type: "keyUp", ...base });
      await sleep(1500);
      await tab.until("!document.querySelector('.mr-ui-bar').dataset.busy", 400000);
      log(await replies());
      await sleep(3000);
    });
  });
  await clip("before_after", async () => {
    await fresh(15000);
    const hide = `(()=>{const s=document.createElement('style');s.textContent='.mr-ui-bar,[class*="mr-chrome"],.mr-ui-gizmo{display:none!important}';document.head.append(s);document.documentElement.lastChild.style.display='none'})()`;
    await tab.js(hide);
    const A = [[0.9, 1.75, -3.7], [1.9, 0.5, 0.2]], B = [[2.3, 1.75, -3.7], [1.5, 0.5, 0.2]];
    await pose(...A); await sleep(5000);
    await rec("room_before", async () => { await glide(...B, 6000); });
    await tab.js(`(async()=>{for (const r of [
      {op:'move',object_id:'sofa#0',args:{dx:0.2,dy:0,dz:0}},
      {op:'rotate',object_id:'armchair#1',args:{yaw_deg:-30}},
    ]) await mrUi.edit(r)})()`);
    await tab.js(`fetch('/api/scenes/blender07/library',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({library_id:'potted_plant_02',x:0.35,z:0.75,yaw_deg:0})}).then(async r=>{console.log(r.status);return r.status})`).then((x) => log('lib', x));
    await sleep(3000);
    await pose(...A); await sleep(5000);
    await rec("room_after", async () => { await glide(...B, 6000); });
    log(await tab.js("mrUi.manifest().history.length"));
  });
} finally {
  close();
}
