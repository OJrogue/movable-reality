// Stills of the armchair from several angles and object styles, to pick the cleanest view.
import { launch, sleep } from "./lib.mjs";
const J = JSON.stringify;
const { tab, close } = await launch();
await sleep(3000);
const VIEWS = {
  a: [[1.9, 1.3, -0.9], [0.1, 0.5, -1.8]],
  b: [[1.9, 1.2, -3.2], [0.1, 0.5, -1.8]],
  c: [[2.4, 1.0, -1.9], [0.1, 0.55, -1.8]],
  d: [[0.9, 1.5, -3.4], [0.1, 0.5, -1.7]],
  e: [[1.2, 1.4, -0.3], [0.1, 0.5, -1.8]],
};
for (const style of (process.argv[2] || "auto,completed,mesh").split(",")) {
  await tab.cdp("Network.enable");
  await tab.cdp("Network.clearBrowserCookies");
  await tab.goto(`http://localhost:5173/?scene=blender07&onboarding=0&hud=0&budget=0&objects=${style}`);
  await tab.ready(12000);
  for (const [k, v] of Object.entries(VIEWS)) {
    await tab.js(`__pose(${J(v[0])},${J(v[1])})`);
    await sleep(3500);
    await tab.shot(`look_${style}_${k}.jpg`);
    console.log(style, k);
  }
}
close();
process.exit(0);
