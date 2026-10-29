// Run menu for device sessions, served by the dev server at /r0 (vite.config.ts): lists
// every run of the R0 checklist as a big link, ticks the runs whose logs already exist
// (GET /__bench), and links to the next one. Typing long URLs in a headset browser is
// slow; this needs one short URL per device.
// Options: ?duration=<s> (default 120), ?path=<name> (default walk), ?device=<label>.
import { deviceLabel, isLogOf, modeTag } from "../perf/logName";

export const R0_SCENES = [50_000, 150_000, 300_000, 600_000, 1_000_000].map((n) => `bench_${n}`);
export const R0_MODES = ["naive", "lod"] as const;

export interface MenuRun {
  scene: string;
  mode: (typeof R0_MODES)[number];
  url: string;
}

export function r0Runs(params: URLSearchParams): MenuRun[] {
  const duration = params.get("duration") ?? "120";
  const path = params.get("path") ?? "walk";
  const device = params.get("device");
  const runs: MenuRun[] = [];
  for (const scene of R0_SCENES) {
    for (const mode of R0_MODES) {
      const q = new URLSearchParams({ bench: "1", scene, mode, duration, path });
      if (device) q.set("device", device);
      runs.push({ scene, mode, url: `/?${q}` });
    }
  }
  return runs;
}

export async function showMenu(params: URLSearchParams): Promise<void> {
  const label = deviceLabel(navigator.userAgent, navigator.webdriver === true, params.get("device"));
  let files: string[] = [];
  let endpoint = true;
  try {
    const r = await fetch("/__bench", { cache: "no-store" });
    files = ((await r.json()) as { raw: string[] }).raw;
  } catch {
    endpoint = false;
  }
  const runs = r0Runs(params);
  const done = (run: MenuRun) => files.some((f) => isLogOf(f, label, run.scene, modeTag(run.mode, false)));
  const next = runs.find((r) => !done(r));

  const root = document.createElement("div");
  root.style.cssText =
    "font:22px/1.4 system-ui,sans-serif;color:#eee;padding:24px;max-width:900px;margin:auto";
  const h = (tag: string, text: string, css = "") => {
    const el = document.createElement(tag);
    el.textContent = text;
    el.style.cssText = css;
    root.appendChild(el);
    return el;
  };
  h("h1", "R0 device baseline", "font-size:32px;margin:0 0 8px");
  h(
    "p",
    `device: ${label}   ${runs.filter(done).length} / ${runs.length} runs logged` +
      (endpoint ? "" : "   (no /__bench endpoint: start the dev server, see docs/DEVICES.md)"),
    "margin:0 0 16px;color:#bbb",
  );
  if (next) {
    const a = h("a", `next: ${next.scene} ${next.mode}`, linkCss("#2d6a3e", 28)) as HTMLAnchorElement;
    a.href = next.url;
  } else {
    h("p", "all runs logged. Run scripts/bench_report.py on the laptop.", "color:#9fe3a8");
  }
  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px";
  root.appendChild(grid);
  for (const run of runs) {
    const a = document.createElement("a");
    a.href = run.url;
    a.textContent = `${done(run) ? "✓ " : ""}${run.scene.replace("bench_", "")} ${run.mode}`;
    a.style.cssText = linkCss(done(run) ? "#333" : "#244a6b", 24);
    grid.appendChild(a);
  }
  h(
    "p",
    "Each run: open it, press Enter VR, stand still, let the room move you. The log saves " +
      "itself when the run ends; then press Exit VR and go Back to this page.",
    "margin-top:20px;color:#bbb",
  );
  document.body.appendChild(root);
}

function linkCss(bg: string, size: number): string {
  return (
    `display:block;padding:18px 20px;border-radius:12px;background:${bg};color:#fff;` +
    `text-decoration:none;font-size:${size}px;text-align:center`
  );
}

if (typeof document !== "undefined" && document.title === "R0 runs") {
  void showMenu(new URLSearchParams(location.search));
}
