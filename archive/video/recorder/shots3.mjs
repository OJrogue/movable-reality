// New landing page shots from the live site, 5 Nov.
import { launch, sleep } from "./lib.mjs";

const URL_LIVE = "http://localhost:5173/";
const { tab, close } = await launch();
await sleep(4000);
await tab.cdp("Network.enable");
await tab.cdp("Network.clearBrowserCookies");
await tab.goto(URL_LIVE);
await sleep(5000);
await tab.js(`window.__scrollTo = (y, ms, el) => new Promise((ok) => {
  const t = el || document.scrollingElement, a = t.scrollTop, t0 = performance.now();
  const step = (n) => { const u = Math.min(1, (n - t0) / ms), s = u * u * (3 - 2 * u);
    t.scrollTop = a + (y - a) * s; u < 1 ? requestAnimationFrame(step) : ok(); };
  requestAnimationFrame(step); });
  window.__top = (q) => document.querySelector(q).getBoundingClientRect().top + document.scrollingElement.scrollTop;
  document.documentElement.style.scrollBehavior = "auto";`);
console.log("size", await tab.js("[innerWidth, innerHeight]"));

const rec = async (name, fn) => {
  const stop = await tab.record(name);
  await sleep(500);
  await fn();
  await sleep(500);
  await tab.shot(name + ".jpg");
  await stop();
  console.log("done", name);
};

await rec("landing_hero", async () => { await sleep(10000); });

await tab.js(`__scrollTo(__top('#try') - 70, 1)`);
await sleep(1500);
await rec("landing_card", async () => {
  await sleep(1200);
  const p = await tab.js(`(()=>{const r=document.querySelector('.lp-scene[data-scene="blender07"]').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()`);
  await tab.click(p[0], p[1], 1100);
  await sleep(5500);
  await tab.js(`(()=>{const d=document.querySelector('.lp-room');return __scrollTo(d.scrollHeight, 5000, d)})()`);
  await sleep(2500);
});
await tab.js(`document.querySelector('.lp-room').close()`);
await sleep(600);

await tab.js(`__scrollTo(__top('#pipeline') - 70, 1)`);
await sleep(1500);
await rec("landing_pipeline", async () => {
  await sleep(1200);
  await tab.js(`__scrollTo(__top('#models') - 500, 14000)`);
  await sleep(800);
});

await tab.js(`__scrollTo(__top('#models') - 70, 1)`);
await sleep(1500);
await rec("landing_models", async () => { await sleep(16000); });

await tab.js(`__scrollTo(__top('#build') - 180, 1)`);
await sleep(1500);
await rec("landing_build", async () => { await sleep(14000); });

close();
process.exit(0);
