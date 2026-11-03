// Page entry. / without a scene is the landing page (landing/); any viewer parameter
// (?scene, ?ply, ?bench, ...) opens the viewer (main.ts), so every existing link still works.
// Dynamic imports keep three.js and Spark out of the landing page.
const VIEWER_PARAMS = ["scene", "ply", "bench", "menu"];
const params = new URLSearchParams(location.search);

if (VIEWER_PARAMS.some((p) => params.has(p))) void import("./main");
else void import("./landing").then((m) => m.installLanding());
