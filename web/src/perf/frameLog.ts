// Frame-time log: one row per animation-loop frame, stored as columns, exported as JSON.
// In dev the viewer POSTs it to the Vite endpoint /__bench (vite.config.ts), which writes
// benchmarks/raw/<timestamp>_<device>_<scene>_<mode>.json (headless runs go to
// benchmarks/headless/). scripts/bench_report.py reads these files.
import { summarize, type FrameSummary } from "./stats";
import { deviceLabel, logStem } from "./logName";

export type RenderMode = "lod" | "naive";

export interface DeviceInfo {
  label: string;
  user_agent: string;
  platform: string;
  hardware_concurrency: number | null;
  device_memory_gb: number | null;
  gpu_vendor: string | null;
  gpu_renderer: string | null;
}

export interface DisplayInfo {
  /** Drawing buffer of the canvas, or the XR framebuffer while presenting. */
  framebuffer: [number, number];
  pixel_ratio: number;
  /** renderer.xr framebuffer scale requested from the session (Spark's default is 0.5). */
  xr_framebuffer_scale: number | null;
  xr_frame_rate: number | null;
  xr_supported_frame_rates: number[] | null;
}

export interface LodSettings {
  enable_lod: boolean;
  /** Explicit lodSplatCount, or Spark's platform default when unset. */
  lod_splat_count: number;
  lod_splat_count_explicit: boolean;
  lod_splat_scale: number;
  lod_render_scale: number;
  lod_traverse_mode: string;
  budget_controller: boolean;
  /** Spark's fixed foveation (renderer options). */
  cone_fov0: number;
  cone_fov: number;
  cone_foveate: number;
  behind_foveate: number;
}

export interface FrameLogMeta {
  scene: string;
  /** Mode at the start; "mixed" in the file if the recording spans a toggle. */
  mode: RenderMode | "mixed";
  /** Budget controller active at the start (see events for changes). */
  budget: boolean;
  /** Bench path name, or "free" for a manual recording. */
  path: string;
  xr: boolean;
  headless: boolean;
  device: DeviceInfo;
  target_ms: number;
  /** "xr" (XRSession.frameRate), "estimate" (steady frame intervals), "url" or "default". */
  target_source: string;
  url: string;
  total_splats: number | null;
  display: DisplayInfo;
  settings: LodSettings;
  bench: { duration_s: number; warmup_s: number } | null;
}

export interface FrameLogEvent {
  t: number;
  kind: string;
  detail: Record<string, unknown>;
}

export interface FrameLogJson extends FrameLogMeta {
  schema: 1;
  kind: "mr-frame-log";
  created: string;
  events: FrameLogEvent[];
  frames: {
    /** Frame timestamp (XR frame time in XR, rAF time on desktop), ms since the first row. */
    t: number[];
    /** Interval to the previous frame, ms. */
    dt: number[];
    /** Main-thread time of the frame callback (lane hooks + render submit), ms. */
    cpu_ms: number[];
    /** Splats Spark drew: SparkRenderer.activeSplats, the sort's count after culling. */
    drawn: number[];
    /** Splats in the current cut (sum over the generator mapping: LOD cut or full meshes). */
    cut: number[];
    lod_splat_count: number[];
    lod_render_scale: number[];
    /** 1 when the frame rendered with LOD, 0 in naive mode. */
    lod: number[];
  };
  summary: FrameSummary & { duration_s: number };
}

const round2 = (v: number) => Math.round(v * 100) / 100;

export class FrameRecorder {
  readonly created = new Date().toISOString();
  private t0: number | null = null;
  private readonly cols: FrameLogJson["frames"] = {
    t: [],
    dt: [],
    cpu_ms: [],
    drawn: [],
    cut: [],
    lod_splat_count: [],
    lod_render_scale: [],
    lod: [],
  };
  private readonly events: FrameLogEvent[] = [];

  constructor(readonly meta: FrameLogMeta) {}

  get frames(): number {
    return this.cols.t.length;
  }

  /** Seconds since the first recorded frame. */
  elapsedS(time: number): number {
    return this.t0 === null ? 0 : (time - this.t0) / 1000;
  }

  push(
    time: number,
    dt: number,
    cpuMs: number,
    drawn: number,
    cut: number,
    splatCount: number,
    renderScale: number,
    lod: boolean,
  ): void {
    if (this.t0 === null) this.t0 = time;
    const c = this.cols;
    c.t.push(round2(time - this.t0));
    c.dt.push(round2(dt));
    c.cpu_ms.push(round2(cpuMs));
    c.drawn.push(drawn);
    c.cut.push(cut);
    c.lod_splat_count.push(splatCount);
    c.lod_render_scale.push(round2(renderScale));
    c.lod.push(lod ? 1 : 0);
  }

  event(time: number, kind: string, detail: Record<string, unknown> = {}): void {
    this.events.push({ t: this.t0 === null ? 0 : round2(time - this.t0), kind, detail });
  }

  finish(): FrameLogJson {
    const s = summarize(this.cols.dt, this.meta.target_ms);
    const duration = this.cols.dt.reduce((a, b) => a + b, 0) / 1000;
    const lodFrames = this.cols.lod.reduce((a, b) => a + b, 0);
    const mixed = lodFrames > 0 && lodFrames < this.cols.lod.length;
    return {
      schema: 1,
      kind: "mr-frame-log",
      created: this.created,
      ...this.meta,
      mode: mixed ? "mixed" : this.meta.mode,
      events: this.events,
      frames: this.cols,
      summary: { ...s, duration_s: round2(duration) },
    };
  }
}

export function deviceInfo(gl: WebGL2RenderingContext | WebGLRenderingContext): DeviceInfo {
  const nav = navigator as Navigator & { deviceMemory?: number };
  const params = new URLSearchParams(location.search);
  let vendor: string | null = null;
  let renderer: string | null = null;
  try {
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    vendor = String(gl.getParameter(ext ? ext.UNMASKED_VENDOR_WEBGL : gl.VENDOR));
    renderer = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  } catch {
    // Some browsers refuse the extension; the user agent still identifies the device.
  }
  return {
    label: deviceLabel(nav.userAgent, nav.webdriver === true, params.get("device")),
    user_agent: nav.userAgent,
    platform: nav.platform,
    hardware_concurrency: nav.hardwareConcurrency ?? null,
    device_memory_gb: nav.deviceMemory ?? null,
    gpu_vendor: vendor,
    gpu_renderer: renderer,
  };
}

/** POST to the dev server; resolves to the path it wrote, rejects if there is no endpoint. */
export async function postLog(log: FrameLogJson): Promise<string> {
  const r = await fetch("/__bench", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(log),
  });
  if (!r.ok) throw new Error(`/__bench answered ${r.status}`);
  const body = (await r.json()) as { path?: string };
  return body.path ?? "(no path)";
}

export function downloadLog(log: FrameLogJson): void {
  const blob = new Blob([JSON.stringify(log)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${logStem(log)}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}
