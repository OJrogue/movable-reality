// Lane renderer: HUD, frame-time log, budget controller, naive/LOD toggle and bench mode.
// main.ts calls installPerf(ctx) after the other lanes' installs.
//
// URL parameters (all optional):
//   ?mode=lod|naive        LOD (default) or naive: LOD off, full splats (the proof toggle)
//   ?budget=1|0            budget controller (default on in LOD mode, off in bench mode)
//   ?target=<ms>           frame-time target (default: from the XR frame rate, else 13.9)
//   ?hz=<rate>             ask the XR session for this display rate (if supported)
//   ?lodSplatCount=<n>     starting splat budget (the controller owns it when active)
//   ?lodRenderScale=<f>    starting minimum splat size in pixels (1 to 5)
//   ?hud=0                 start with the HUD hidden
//   ?device=<label>        device name for log file names (default: from the user agent)
//   ?keepFull=0            LOD mode: do not keep the full splat set for the instant toggle
//   ?bench=1&scene=<id>&mode=lod|naive&duration=<s>&path=<name>   bench mode (bench/)
// Keys and controller buttons: see input.ts.
import * as THREE from "three";
import type { SplatMeshOptions } from "@sparkjsdev/spark";
import type { ViewerContext } from "../context";
import { applyPose, BenchRunner, parseBench } from "../bench";
import { BudgetController, DEFAULT_BUDGET, levelForSplatCount, settingsForLevel } from "./budget";
import {
  deviceInfo,
  downloadLog,
  FrameRecorder,
  postLog,
  type DisplayInfo,
  type FrameLogJson,
  type FrameLogMeta,
  type RenderMode,
} from "./frameLog";
import { Hud, type HudData } from "./hud";
import { PerfInput, type PerfAction } from "./input";
import {
  hasFullSplats,
  hasLodTree,
  lodSettings,
  lodSplatCount,
  plyVertexCount,
  sparkCounters,
  splatMeshes,
  splatsReady,
} from "./sparkInfo";
import { estimateHz, percentile, RollingWindow, summarize, targetForHz } from "./stats";

/** Fixed LOD budget for bench runs without the controller, so devices are comparable. */
export const BENCH_LOD_SPLATS = 500_000;
const DEFAULT_TARGET_MS = 13.9;
/** Free (non-bench) recordings stop growing after 30 min at 72 Hz. */
const MAX_FREE_FRAMES = 72 * 60 * 30;
const HUD_INTERVAL_MS = 250;
const ESTIMATE_AFTER_MS = 2000;

export interface PerfHandle {
  mode(): RenderMode;
  setMode(m: RenderMode): void;
  budget(): { active: boolean; level: number; phase: string; splatCount: number };
  bench(): ReturnType<BenchRunner["status"]> | null;
  /** Current recording as a log (null before the scene is ready). */
  log(): FrameLogJson | null;
  /** POST the current recording to /__bench (download if there is no endpoint). */
  save(): Promise<string | null>;
  readonly target: { ms: number; source: string; hz: number | null };
}

export function installPerf(ctx: ViewerContext): PerfHandle {
  const p = ctx.params;
  const spark = ctx.spark;
  const gl = ctx.renderer.getContext();

  // ---- bench configuration
  let bench: BenchRunner | null = null;
  let benchError: string | null = null;
  try {
    const cfg = parseBench(p);
    if (cfg) bench = new BenchRunner(cfg);
  } catch (e) {
    benchError = `bench: ${e instanceof Error ? e.message : String(e)}`;
  }

  // ---- mode and mesh options
  let mode: RenderMode = bench?.cfg.mode ?? (p.get("mode") === "naive" ? "naive" : "lod");
  const keepFull = !bench && mode === "lod" && p.get("keepFull") !== "0";
  if (mode === "lod") {
    // Lane lod sets ctx.meshOptions in installLod (runs before this). Until it is in
    // main.ts, fall back to Spark's load-time LOD tree (tiny-lod in a worker).
    const inner = ctx.meshOptions;
    ctx.meshOptions = (ply, id, m) => {
      const o: Partial<SplatMeshOptions> = { ...(inner ? inner(ply, id, m) : { lod: true }) };
      // Keep the full splat set next to the tree so the naive toggle is instant. A .RAD
      // tree has no full set; toggling then reloads the page with ?mode=naive.
      if (keepFull && o.lod && !o.paged) o.nonLod = true;
      return o;
    };
  } else {
    spark.enableLod = false;
  }

  // ---- LOD settings and budget controller
  const urlCount = Number(p.get("lodSplatCount"));
  const hasUrlCount = Number.isInteger(urlCount) && urlCount > 0;
  const budgetParam = p.get("budget");
  let budgetWanted = budgetParam === "1" || (budgetParam === null && !bench);
  if (hasUrlCount) spark.lodSplatCount = urlCount;
  else if (bench && mode === "lod" && !budgetWanted) spark.lodSplatCount = BENCH_LOD_SPLATS;
  const urlRenderScale = Number(p.get("lodRenderScale"));
  if (urlRenderScale >= 1 && urlRenderScale <= 5) spark.lodRenderScale = urlRenderScale;

  const explicitTarget = Number(p.get("target"));
  const target = {
    ms: explicitTarget > 0 ? explicitTarget : DEFAULT_TARGET_MS,
    source: explicitTarget > 0 ? "url" : "default",
    hz: null as number | null,
  };
  const startLevel = hasUrlCount
    ? levelForSplatCount(urlCount, DEFAULT_BUDGET)
    : levelForSplatCount(lodSplatCount(spark).count, DEFAULT_BUDGET);
  const controller = new BudgetController({ targetMs: target.ms, startLevel });
  const budgetActive = () => budgetWanted && mode === "lod";
  const applyBudget = () => {
    const s = controller.settings();
    if (spark.lodSplatCount !== s.splatCount) spark.lodSplatCount = s.splatCount;
    if (spark.lodRenderScale !== s.renderScale) spark.lodRenderScale = s.renderScale;
  };

  // ---- HUD, input, download button
  const hud = new Hud(ctx.camera, p.get("hud") !== "0");
  const input = new PerfInput(ctx.renderer);
  const button = document.createElement("button");
  button.textContent = "save log";
  Object.assign(button.style, {
    position: "fixed",
    right: "12px",
    top: "12px",
    padding: "4px 10px",
    font: "13px system-ui, sans-serif",
    background: "rgba(0,0,0,0.55)",
    color: "#eee",
    border: "1px solid #666",
    borderRadius: "6px",
    cursor: "pointer",
    zIndex: "1000",
  });
  button.addEventListener("click", () => {
    const log = recorder?.finish();
    if (log && log.frames.t.length > 0) downloadLog(log);
  });
  document.body.appendChild(button);

  // ---- total splats from the PLY headers (the original count, however meshes loaded)
  const plyCounts = new Map<string, number | null>();
  let totalSplats: number | null = null;
  const countPly = (url: string) => {
    if (plyCounts.has(url)) return;
    plyCounts.set(url, null);
    plyVertexCount(url)
      .then((n) => plyCounts.set(url, n))
      .catch(() => plyCounts.set(url, null))
      .finally(updateTotal);
  };
  const updateTotal = () => {
    let sum = 0;
    for (const n of plyCounts.values()) {
      if (n === null) return; // still loading or unknown
      sum += n;
    }
    totalSplats = sum;
    if (budgetActive()) {
      let ceiling = levelForSplatCount(sum, DEFAULT_BUDGET);
      if (settingsForLevel(ceiling, DEFAULT_BUDGET).splatCount < sum) ceiling++;
      controller.setMaxLevel(ceiling);
    }
  };
  ctx.onManifest((m) => {
    if (!ctx.sceneBase) return;
    countPly(`${ctx.sceneBase}/${m.background_ply}`);
    for (const o of m.objects) countPly(`${ctx.sceneBase}/${o.ply}`);
  });
  const plyParam = p.get("ply");
  if (plyParam) countPly(plyParam);

  // ---- XR session: frame rate and target
  const xrScale = ctx.xrFramebufferScale;
  const wantHz = Number(p.get("hz"));
  let estimateFrom: number | null = null;
  const setTarget = (ms: number, source: string, hz: number | null) => {
    target.hz = hz;
    if (target.source === "url") return;
    target.ms = ms;
    target.source = source;
    controller.setTarget(ms);
  };
  ctx.renderer.xr.addEventListener("sessionstart", () => {
    const session = ctx.renderer.xr.getSession();
    if (!session) return;
    const supported = session.supportedFrameRates;
    if (wantHz > 0 && supported && Array.from(supported).includes(wantHz)) {
      session.updateTargetFrameRate(wantHz).catch(() => undefined);
    }
    if (session.frameRate) setTarget(targetForHz(session.frameRate), "xr", session.frameRate);
    else estimateFrom = null;
    session.addEventListener("frameratechange", () => {
      if (session.frameRate) setTarget(targetForHz(session.frameRate), "xr", session.frameRate);
    });
    resetWindows("xr-enter");
  });
  ctx.renderer.xr.addEventListener("sessionend", () => {
    if (target.source === "xr") setTarget(DEFAULT_TARGET_MS, "default", null);
    resetWindows("xr-exit");
  });

  // ---- recording
  const device = deviceInfo(gl);
  const headless = device.label === "headless";
  let recorder: FrameRecorder | null = null;
  let flash = "";
  let flashUntil = 0;
  const say = (msg: string, ms = 4000) => {
    flash = msg;
    flashUntil = performance.now() + ms;
    console.info(`[perf] ${msg}`);
  };
  if (benchError) say(benchError, 1e9);

  const displayInfo = (): DisplayInfo => {
    const session = ctx.renderer.xr.getSession();
    const layer = session?.renderState.baseLayer;
    const size = ctx.renderer.getDrawingBufferSize(new THREE.Vector2());
    return {
      framebuffer: layer ? [layer.framebufferWidth, layer.framebufferHeight] : [size.x, size.y],
      pixel_ratio: ctx.renderer.getPixelRatio(),
      xr_framebuffer_scale: ctx.renderer.xr.isPresenting ? xrScale : null,
      xr_frame_rate: session?.frameRate ?? null,
      xr_supported_frame_rates: session?.supportedFrameRates
        ? Array.from(session.supportedFrameRates)
        : null,
    };
  };

  const meta = (): FrameLogMeta => ({
    scene: bench?.cfg.scene ?? p.get("scene") ?? (plyParam ? "ply" : "none"),
    mode,
    budget: budgetActive(),
    path: bench?.cfg.path.name ?? "free",
    xr: ctx.renderer.xr.isPresenting,
    headless,
    device,
    target_ms: target.ms,
    target_source: target.source,
    url: location.href,
    total_splats: totalSplats,
    display: displayInfo(),
    settings: lodSettings(spark, budgetActive()),
    bench: bench ? { duration_s: bench.cfg.durationS, warmup_s: bench.cfg.warmupS } : null,
  });

  const startRecording = (time: number, why: string) => {
    recorder = new FrameRecorder(meta());
    recorder.event(time, "start", { why });
  };

  const save = async (): Promise<string | null> => {
    const rec = recorder;
    if (!rec || rec.frames === 0) return null;
    const log = rec.finish();
    try {
      const path = await postLog(log);
      say(`saved ${path}`);
      return path;
    } catch {
      downloadLog(log); // no dev server (static hosting): hand the file to the browser
      say("downloaded log (no /__bench endpoint)");
      return "download";
    }
  };

  // ---- mode toggle and budget toggle
  let savedCleanupMs = spark.lodCleanupTimeoutMs;
  const reloadWithMode = (m: RenderMode) => {
    const u = new URL(location.href);
    u.searchParams.set("mode", m);
    location.assign(u.toString());
  };
  const setMode = (next: RenderMode) => {
    if (next === mode) return;
    if (bench) return say("mode is fixed in bench mode");
    const meshes = splatMeshes(ctx.scene).filter(hasLodTree);
    if (next === "naive") {
      if (meshes.length === 0 || !meshes.every(hasFullSplats)) return reloadWithMode("naive");
      // Keep the LOD trees in Spark's worker while they are unused, so switching back is
      // instant (Spark otherwise drops them after lodCleanupTimeoutMs).
      savedCleanupMs = spark.lodCleanupTimeoutMs;
      spark.lodCleanupTimeoutMs = Infinity;
      spark.enableLod = false;
    } else {
      if (meshes.length === 0) return reloadWithMode("lod");
      spark.enableLod = true;
      spark.lodCleanupTimeoutMs = savedCleanupMs;
      if (budgetActive()) applyBudget();
    }
    mode = next;
    say(`mode ${mode}`);
    resetWindows(`mode-${mode}`);
    recorder?.event(lastTime ?? 0, "mode", { mode });
  };
  const toggleBudget = () => {
    if (mode !== "lod") return say("budget controller works in LOD mode only");
    budgetWanted = !budgetWanted;
    if (budgetWanted) {
      controller.restart(levelForSplatCount(lodSplatCount(spark).count, DEFAULT_BUDGET));
      applyBudget();
    }
    recorder?.event(lastTime ?? 0, "budget", { active: budgetWanted });
    say(`budget controller ${budgetWanted ? "on" : "off (settings frozen)"}`);
  };
  const act = (a: PerfAction) => {
    if (a === "mode") setMode(mode === "lod" ? "naive" : "lod");
    else if (a === "hud") hud.toggle();
    else if (a === "budget") toggleBudget();
    else if (a === "save") void save();
  };

  // ---- per-frame work
  const window2s = new RollingWindow(2000);
  let lastTime: number | null = null;
  let lastDt = 0;
  let ready = false;
  let hudAt = -Infinity;
  const resetWindows = (why: string) => {
    window2s.clear();
    controller.reset();
    estimateFrom = null;
    recorder?.event(lastTime ?? 0, "reset", { why });
  };

  ctx.onFrame((time) => {
    for (const a of input.poll()) act(a);
    if (!bench) return;
    const presenting = ctx.renderer.xr.isPresenting;
    const signal = bench.tick(time, ready, presenting);
    if (signal === "start-recording") {
      startRecording(time, "bench");
      if (budgetActive()) controller.restart(controller.settings().level);
    } else if (signal === "finish" && recorder) {
      const log = recorder.finish();
      recorder = null;
      postLog(log)
        .then((path) => bench!.done(path, pickSummary(log)))
        .catch((e: unknown) => {
          downloadLog(log);
          bench!.fail(`POST /__bench failed (${String(e)}); downloaded instead`);
        });
    } else if (signal === "abort") {
      recorder = null;
    }
    ctx.controlsEnabled = !bench.drivesRig;
    if (bench.drivesRig) applyPose(ctx.rig, bench.pose(time));
  });

  ctx.onFrameEnd((time, _frame, cpuMs) => {
    const dt = lastTime === null ? 0 : time - lastTime;
    lastTime = time;
    if (dt > 0) lastDt = dt;
    if (!ready) {
      ready = splatsReady(ctx.scene, spark) && spark.activeSplats > 0;
      if (ready) {
        if (!bench) startRecording(time, "ready");
        if (budgetActive()) applyBudget();
      }
    }
    const counters = sparkCounters(spark);
    if (ready && dt > 0) {
      window2s.push(dt);
      // Display rate from steady frames once the scene is up (desktop, or XR without
      // XRSession.frameRate). An overloaded start gives no estimate; the default stays.
      if (target.source === "default") {
        estimateFrom ??= time;
        if (time - estimateFrom >= ESTIMATE_AFTER_MS) {
          // p10: overload only lengthens intervals, so the short ones show the display.
          const hz = estimateHz(percentile(window2s.snapshot(), 10));
          if (hz) setTarget(targetForHz(hz), "estimate", hz);
          else target.source = "default-checked";
        }
      }
      if (budgetActive() && target.source !== "default") {
        const step = controller.update(dt);
        if (step.reason) {
          applyBudget();
          recorder?.event(time, "budget", { ...step });
        }
      }
      if (recorder && (bench || recorder.frames < MAX_FREE_FRAMES)) {
        const lod = lodSplatCount(spark);
        recorder.push(
          time,
          dt,
          cpuMs,
          counters.drawn,
          counters.cut,
          lod.count,
          spark.lodRenderScale,
          mode === "lod",
        );
      }
    }

    const now = performance.now();
    if (now - hudAt < HUD_INTERVAL_MS) return;
    hudAt = now;
    if (!ready) {
      hud.showStatus([ctx.status, bench?.label() ?? "", flash].filter(Boolean).join("\n"));
      return;
    }
    const s = window2s.summary(target.ms);
    const lod = lodSplatCount(spark);
    const b = controller.state();
    const data: HudData = {
      mode,
      xr: ctx.renderer.xr.isPresenting,
      targetMs: target.ms,
      hz: target.hz,
      lastMs: lastDt,
      p50: s.p50,
      p99: s.p99,
      fps: s.fps,
      missedShare: s.missedShare,
      drawn: counters.drawn,
      cut: counters.cut,
      total: totalSplats,
      lodSplatCount: lod.count,
      lodExplicit: lod.explicit,
      lodSplatScale: spark.lodSplatScale,
      lodRenderScale: spark.lodRenderScale,
      budget: budgetActive() ? `L${b.level} ${b.phase}` : "off",
      bench: bench?.label() ?? null,
      status: [ctx.status, now < flashUntil ? flash : ""].filter(Boolean).join("  "),
    };
    hud.update(data, now);
    button.style.display = data.xr || bench ? "none" : "";
  });

  const handle: PerfHandle = {
    mode: () => mode,
    setMode,
    budget: () => ({ active: budgetActive(), ...controller.state() }),
    bench: () => bench?.status() ?? null,
    log: () => recorder?.finish() ?? null,
    save,
    target,
  };
  (window as unknown as { mrPerf: PerfHandle }).mrPerf = handle;
  console.info(
    `[perf] installed: mode ${mode}, budget ${budgetActive() ? "on" : "off"}, ` +
      `lodSplatCount ${lodSplatCount(spark).count}${bench ? `, bench ${bench.cfg.path.name}` : ""}`,
  );
  return handle;
}

function pickSummary(log: FrameLogJson): Record<string, number> {
  const s = summarize(log.frames.dt, log.target_ms);
  return { frames: s.frames, p50: s.p50, p99: s.p99, fps: s.fps, missedShare: s.missedShare };
}
