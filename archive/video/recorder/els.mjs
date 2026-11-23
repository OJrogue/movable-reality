// Element screenshots from the landing page, at 2x, without the page around them.
import { launch, sleep } from "./lib.mjs";
import { writeFileSync } from "node:fs";
const { tab, close } = await launch();
await sleep(2000);
await tab.cdp("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 2, mobile: false });
await tab.cdp("Page.navigate", { url: "http://localhost:5173/" });
await sleep(5000);
await tab.js(`(()=>{const s=document.createElement("style");s.textContent="header,nav,.lp-top,.lp-bar{display:none!important}";document.head.append(s)})()`);
const rect = await tab.js(`(()=>{const e=document.querySelector('.lp-models');e.scrollIntoView();const r=e.getBoundingClientRect();
  const y=scrollY||document.scrollingElement.scrollTop;
  return {x:r.x,y:r.y+y,w:r.width,h:r.height,cells:[...e.children].map(c=>{const q=c.getBoundingClientRect();return [q.x-r.x,q.y-r.y,q.width,q.height]})}})()`);
await sleep(1500);
const s = await tab.cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
  clip: { x: rect.x, y: rect.y, width: rect.w, height: rect.h, scale: 2 } });
writeFileSync("../_cut/el_models.png", Buffer.from(s.data, "base64"));
writeFileSync("../_cut/el_models.json", JSON.stringify(rect));
console.log(rect);
close();
process.exit(0);
