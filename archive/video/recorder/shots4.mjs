// Select, move and turn the armchair in one take, from the side where its splats are clean.
import { launch, sleep } from "./lib.mjs";
const J = JSON.stringify;
const { tab, close } = await launch();
await sleep(3000);
await tab.cdp("Network.enable");
await tab.cdp("Network.clearBrowserCookies");
await tab.goto("http://localhost:5173/?scene=blender07&onboarding=0&hud=0&budget=0");
await tab.ready(14000);
tab.x = 1500; tab.y = 700;
const scr = (p) => tab.js(`__screen(${J(p)})`);
const center = (id) => tab.js(`(()=>{const o=__obj(${J(id)});const t=o.transform||{};const d=t.translation||[0,0,0];return o.bbox_min.map((m,i)=>(m+o.bbox_max[i])/2+(d[i]||0))})()`);
const hscale = (p) => tab.js(`(()=>{const e=__cam.eye,p=${J(p)};return Math.hypot(e[0]-p[0],e[1]-p[1],e[2]-p[2])*1.9*Math.tan(25*Math.PI/180)*0.9/4})()`);
await tab.js(`__pose([2.15,1.5,-0.3],[0.2,0.72,-1.95])`);
await sleep(5000);
await tab.js(`mrCamera()`); // hands the desktop controls back, or clicks do nothing
await sleep(800);
const stop = await tab.record("edit_armchair");
await sleep(500);
let c = await center("armchair#1");
const a = await scr(c);
await tab.click(a[0], a[1], 500);
await sleep(600);
// move along the z axis, sideways on screen
let s = await hscale(c);
const h = await scr([c[0], c[1], c[2] - 0.45 * s]);
const g = await scr([c[0], c[1], c[2] - 0.45 * s - 0.6]);
await tab.drag(h[0], h[1], g[0], g[1], 1300);
await sleep(500);
// rotate, turning the clean side towards the camera
c = await center("armchair#1");
const btn = await tab.js(`(()=>{const b=[...document.querySelectorAll('.mr-ui-gizmo button')].find(x=>x.textContent==='Rotate').getBoundingClientRect();return [b.x+b.width/2,b.y+b.height/2]})()`);
await tab.click(btn[0], btn[1], 450);
await sleep(350);
s = await hscale(c);
const r = 0.5 * s;
const at = (deg) => scr([c[0] + r * Math.cos(deg * Math.PI / 180), c[1], c[2] + r * Math.sin(deg * Math.PI / 180)]);
const p0 = await at(70);
await tab.moveTo(p0[0], p0[1], 450);
await sleep(150);
await tab.mouse("mousePressed", p0[0], p0[1], "left", 1);
for (let d = 70; d >= 25; d -= 3) { const p = await at(d); await tab.moveTo(p[0], p[1], 70, 1, "left"); }
const pe = await at(25);
await tab.mouse("mouseReleased", pe[0], pe[1], "left", 0);
await sleep(700);
await tab.shot("edit_armchair_sel.jpg");
await tab.key("Escape", "Escape");
await sleep(1200);
await tab.shot("edit_armchair.jpg");
await stop();
console.log(await tab.js(`JSON.stringify(__obj("armchair#1").transform)`));
close();
process.exit(0);
