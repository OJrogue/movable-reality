// Minimal CDP driver: real Chrome, real viewer, tab capture through MediaRecorder.
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { createWriteStream, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const OUT = join(HERE, "raw");
mkdirSync(OUT, { recursive: true });
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 9333;
const SINK = 9444;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Recording chunks arrive over plain HTTP from the page.
let sink = null;
createServer((req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  if (req.method === "POST" && sink) req.on("data", (d) => sink.write(d));
  req.on("end", () => res.end("ok"));
  if (req.method !== "POST") res.end("ok");
}).listen(SINK, "127.0.0.1").unref();

export class Tab {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.wait = new Map();
    this.x = 960;
    this.y = 540;
    ws.onmessage = (m) => {
      const d = JSON.parse(m.data); if (d.method && this.onEvent) this.onEvent(d);
      if (d.id && this.wait.has(d.id)) {
        const { ok, no } = this.wait.get(d.id);
        this.wait.delete(d.id);
        d.error ? no(new Error(JSON.stringify(d.error))) : ok(d.result);
      }
    };
  }
  cdp(method, params = {}) {
    const id = ++this.id;
    return new Promise((ok, no) => {
      this.wait.set(id, { ok, no });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async js(expression, gesture = false) {
    const r = await this.cdp("Runtime.evaluate", {
      expression, awaitPromise: true, returnByValue: true, userGesture: gesture,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text);
    return r.result.value;
  }
  async size(width, height, mobile = false) {
    await this.cdp("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile });
  }
  async goto(url) {
    await this.cdp("Page.navigate", { url });
    await sleep(800);
    await this.until("document.readyState === 'complete'", 60000);
    await this.js(HELPERS);
  }
  async until(expr, ms = 120000) {
    const t = Date.now();
    while (Date.now() - t < ms) {
      if (await this.js(`!!(${expr})`).catch(() => false)) return true;
      await sleep(250);
    }
    throw new Error("timeout: " + expr);
  }
  // Scene loaded: the card has gone and meshes drew.
  async ready(settle = 6000) {
    await this.until("window.mrUi && window.mrUi.manifest() && !document.querySelector('.mr-chrome-loading:not([hidden])')", 180000);
    await this.until("!document.body.innerText.match(/Setting up the room|Loading/)", 180000).catch(() => {});
    await sleep(settle);
  }
  async mouse(type, x, y, button = "none", buttons = 0, extra = {}) {
    await this.cdp("Input.dispatchMouseEvent", { type, x, y, button, buttons, clickCount: type === "mouseMoved" ? 0 : 1, ...extra });
  }
  async moveTo(x, y, ms = 700, buttons = 0, button = "none") {
    const n = Math.max(2, Math.round(ms / 16));
    const [x0, y0] = [this.x, this.y];
    for (let i = 1; i <= n; i++) {
      const u = i / n;
      const s = u * u * (3 - 2 * u);
      await this.mouse("mouseMoved", x0 + (x - x0) * s, y0 + (y - y0) * s, button, buttons);
      await sleep(14);
    }
    this.x = x;
    this.y = y;
  }
  async click(x, y, ms = 700) {
    await this.moveTo(x, y, ms);
    await this.mouse("mousePressed", x, y, "left", 1);
    await sleep(70);
    await this.mouse("mouseReleased", x, y, "left", 0);
    await sleep(150);
  }
  async drag(x0, y0, x1, y1, ms = 1800) {
    await this.moveTo(x0, y0);
    await sleep(250);
    await this.mouse("mousePressed", x0, y0, "left", 1);
    await sleep(120);
    await this.moveTo(x1, y1, ms, 1, "left");
    await sleep(120);
    await this.mouse("mouseReleased", x1, y1, "left", 0);
    await sleep(200);
  }
  async key(code, key, mods = 0) {
    const vk = { Enter: 13, Delete: 46, KeyZ: 90, KeyT: 84, KeyG: 71, KeyQ: 81, KeyE: 69, Escape: 27, Space: 32 }[code];
    const base = { code, key, modifiers: mods, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk };
    await this.cdp("Input.dispatchKeyEvent", { type: "rawKeyDown", ...base });
    await sleep(60);
    await this.cdp("Input.dispatchKeyEvent", { type: "keyUp", ...base });
  }
  async type(text, ms = 45) {
    for (const c of text) {
      await this.cdp("Input.insertText", { text: c });
      await sleep(ms * (0.6 + Math.random() * 0.8));
    }
  }
  // Sends a typed command and waits for the agent's reply line.
  async command(text) {
    const before = await this.js("document.querySelectorAll('.mr-ui-log > *').length");
    const p = await this.js("(()=>{const r=document.querySelector('.mr-ui-input').getBoundingClientRect();return [r.x+r.width/3,r.y+r.height/2]})()");
    await this.click(p[0], p[1]);
    await this.type(text);
    await sleep(350);
    await this.key("Enter", "Enter");
    const t = Date.now();
    await sleep(1000);
    await this.until("!document.querySelector('.mr-ui-bar').dataset.busy", 300000);
    return { seconds: (Date.now() - t) / 1000, before };
  }
  // Tab capture of this tab at a high bitrate. Stop writes OUT/<name>.webm.
  async record(name) {
    const path = join(OUT, name + ".webm");
    rmSync(path, { force: true });
    sink = createWriteStream(path);
    await this.js(`(async () => {
      const s = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: 60, max: 60 }, width: { ideal: innerWidth }, height: { ideal: innerHeight }, cursor: "never" },
        audio: false, preferCurrentTab: true, selfBrowserSurface: "include",
      });
      const r = new MediaRecorder(s, { mimeType: "video/webm;codecs=h264", videoBitsPerSecond: 40000000 });
      let q = Promise.resolve();
      r.ondataavailable = (e) => { q = q.then(() => fetch("http://127.0.0.1:${SINK}/", { method: "POST", mode: "no-cors", body: e.data })); };
      await s.getVideoTracks()[0].applyConstraints({ width: { min: innerWidth, max: innerWidth }, height: { min: innerHeight, max: innerHeight }, frameRate: { min: 30, max: 60 } }).catch(() => {});
      window.__rec = { s, r, done: new Promise((ok) => (r.onstop = () => q.then(ok))) };
      r.start(1000);
    })()`, true);
    await sleep(400);
    return async () => {
      await this.js("(async () => { __rec.r.stop(); await __rec.done; __rec.s.getTracks().forEach((t) => t.stop()); })()");
      await sleep(300);
      sink.end();
      sink = null;
      return path;
    };
  }
  async shot(file) {
    const r = await this.cdp("Page.captureScreenshot", { format: "jpeg", quality: 70 });
    const { writeFileSync } = await import("node:fs");
    writeFileSync(join(OUT, file), Buffer.from(r.data, "base64"));
  }
}

// Injected after each navigation: visible cursor, camera moves and world-to-screen.
export const HELPERS = `(() => {
  if (window.__mrHelpers) return; window.__mrHelpers = 1;
  const c = document.createElement("div");
  c.style.cssText = "position:fixed;left:0;top:0;width:22px;height:22px;z-index:2147483647;pointer-events:none;transform:translate(960px,540px)";
  c.innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22"><path d="M4 2l15 12-7 1 4 7-3 1-4-7-5 5z" fill="#fff" stroke="#000" stroke-width="1.5"/></svg>';
  const add = () => document.documentElement.appendChild(c);
  add();
  addEventListener("pointermove", (e) => { c.style.transform = "translate(" + e.clientX + "px," + e.clientY + "px)"; }, true);
  const v = { sub: (a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]], dot:(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2], norm:(a)=>{const l=Math.hypot(...a);return a.map(x=>x/l)} };
  const FOV = 50;
  const basis = (eye,target) => { const f=v.norm(v.sub(target,eye)); const r=v.norm(v.cross(f,[0,1,0])); const u=v.cross(r,f); return {f,r,u}; };
  window.__pose = (eye,target) => { const {f,r,u}=basis(eye,target); window.__cam={eye,target}; mrCamera([[r[0],u[0],-f[0],eye[0]],[r[1],u[1],-f[1],eye[1]],[r[2],u[2],-f[2],eye[2]],[0,0,0,1]], FOV); };
  window.__glide = (eye,target,ms) => new Promise((ok) => { const a=window.__cam, t0=performance.now();
    const lerp=(p,q,s)=>p.map((x,i)=>x+(q[i]-x)*s);
    const step=(t)=>{ const u=Math.min(1,(t-t0)/ms), s=u*u*(3-2*u); __pose(lerp(a.eye,eye,s),lerp(a.target,target,s)); u<1?requestAnimationFrame(step):ok(); };
    requestAnimationFrame(step); });
  window.__screen = (p) => { const {eye,target}=window.__cam; const {f,r,u}=basis(eye,target); const d=v.sub(p,eye); const z=v.dot(d,f);
    const k=1/Math.tan(FOV*Math.PI/360); const asp=innerWidth/innerHeight; return [(v.dot(d,r)/z*k/asp*.5+.5)*innerWidth,(.5-v.dot(d,u)/z*k*.5)*innerHeight]; };
  window.__obj = (id) => mrUi.manifest().objects.find((o)=>o.id===id);
  window.__busy0 = 0;
  window.__mrBusy = () => { const s=[...document.querySelectorAll('.mr-ui-log > *')].at(-1); return s && /…|\\.\\.\\.$/.test(s.textContent) && !/\\.\\s*$/.test(s.textContent.replace(/…$/,'x.')) ; };
})()`;

export async function launch(extra = []) {
  rmSync(join(HERE, "profile"), { recursive: true, force: true });
  const proc = spawn(CHROME, [
    `--remote-debugging-port=${PORT}`, `--user-data-dir=${join(HERE, "profile")}`,
    "--no-first-run", "--no-default-browser-check", "--start-fullscreen",
    "--auto-accept-this-tab-capture", "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding", "--disable-background-timer-throttling",
    "--disable-features=CalculateNativeWinOcclusion", "--hide-crash-restore-bubble", ...extra, "about:blank",
  ], { stdio: "ignore" });
  for (let i = 0; ; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json();
      const page = list.find((t) => t.type === "page");
      const ws = new WebSocket(page.webSocketDebuggerUrl);
      await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = no; });
      const tab = new Tab(ws);
      await tab.cdp("Page.enable");
      await tab.cdp("Runtime.enable");
      // real window size
      return { tab, close: () => { ws.close(); proc.kill(); } };
    } catch (e) {
      if (i > 60) throw e;
      await sleep(500);
    }
  }
}
