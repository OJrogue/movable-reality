// Bench mode: ?bench=1&scene=<id>&mode=lod|naive&duration=<s>&path=<name>
// Optional: &warmup=<s> (default 3), &xr=auto|1|0 (default auto: wait for Enter VR when
// the browser supports immersive-vr, else run on the desktop at once).
//
// Phases: loading (until every splat mesh has loaded and, with LOD, has its first cut)
// -> waiting-xr (XR only, until the user presses Enter VR) -> warmup (standing at the
// path's start) -> recording (the rig follows the path; perf records every frame)
// -> saving (perf POSTs the log to /__bench) -> done | failed.
// The runner only decides phases and rig poses; perf/index.ts records and saves.
import * as THREE from "three";
import type { RenderMode } from "../perf/frameLog";
import { DEFAULT_PATH, getPath, PATHS, type BenchPath, type RigPose } from "./paths";

export interface BenchConfig {
  scene: string;
  mode: RenderMode;
  durationS: number;
  warmupS: number;
  path: BenchPath;
  xr: "auto" | "on" | "off";
}

export type BenchPhase =
  | "loading"
  | "waiting-xr"
  | "warmup"
  | "recording"
  | "saving"
  | "done"
  | "failed";

export type BenchSignal = "start-recording" | "finish" | "abort" | "timeout";

export interface BenchStatus {
  phase: BenchPhase;
  scene: string;
  mode: RenderMode;
  path: string;
  durationS: number;
  /** Seconds recorded so far (recording phase). */
  recordedS: number;
  savedPath: string | null;
  error: string | null;
  summary: Record<string, number> | null;
}

const LOAD_TIMEOUT_S = 300;

/** Parse bench parameters; null without ?bench=1. Throws on invalid values. */
export function parseBench(params: URLSearchParams): BenchConfig | null {
  if (params.get("bench") !== "1") return null;
  const scene = params.get("scene");
  if (!scene) throw new Error("bench needs &scene=<id>");
  const modeParam = params.get("mode") ?? "lod";
  if (modeParam !== "lod" && modeParam !== "naive") throw new Error(`bad mode ${modeParam}`);
  const durationS = Number(params.get("duration") ?? 120);
  const warmupS = Number(params.get("warmup") ?? 3);
  if (!(durationS > 0) || !(warmupS >= 0)) throw new Error("bad duration or warmup");
  const name = params.get("path") ?? DEFAULT_PATH;
  const path = getPath(name);
  if (!path) throw new Error(`unknown path ${name}; known: ${Object.keys(PATHS).join(", ")}`);
  const xrParam = params.get("xr") ?? "auto";
  const xr = xrParam === "1" ? "on" : xrParam === "0" ? "off" : "auto";
  return { scene, mode: modeParam, durationS, warmupS, path, xr };
}

const UP = new THREE.Vector3(0, 1, 0);

export function applyPose(rig: THREE.Object3D, p: RigPose): void {
  rig.position.set(p.x, 0, p.z);
  rig.quaternion.setFromAxisAngle(UP, p.yaw);
}

export class BenchRunner {
  phase: BenchPhase = "loading";
  savedPath: string | null = null;
  error: string | null = null;
  summary: Record<string, number> | null = null;
  private xrWanted: boolean | null;
  private phaseStart: number | null = null;
  private recordStart = 0;
  private lastT = 0;

  constructor(readonly cfg: BenchConfig) {
    this.xrWanted = cfg.xr === "on" ? true : cfg.xr === "off" ? false : null;
    if (this.xrWanted === null) {
      const xr = (navigator as Navigator & { xr?: XRSystem }).xr;
      if (!xr) this.xrWanted = false;
      else {
        xr.isSessionSupported("immersive-vr").then(
          (ok) => (this.xrWanted = ok),
          () => (this.xrWanted = false),
        );
      }
    }
  }

  /** Advance the phase for a frame at time tMs; returns what perf must do now. */
  tick(tMs: number, ready: boolean, presenting: boolean): BenchSignal | null {
    this.lastT = tMs;
    this.phaseStart ??= tMs;
    const inPhase = (tMs - this.phaseStart) / 1000;
    switch (this.phase) {
      case "loading":
        if (ready && this.xrWanted !== null) {
          this.enter(this.xrWanted && !presenting ? "waiting-xr" : "warmup", tMs);
        } else if (inPhase > LOAD_TIMEOUT_S) {
          this.fail(`scene not ready after ${LOAD_TIMEOUT_S} s`);
          return "timeout";
        }
        return null;
      case "waiting-xr":
        if (presenting) this.enter("warmup", tMs);
        return null;
      case "warmup":
        if (this.xrWanted && !presenting) {
          this.enter("waiting-xr", tMs);
          return null;
        }
        if (inPhase >= this.cfg.warmupS) {
          this.enter("recording", tMs);
          this.recordStart = tMs;
          return "start-recording";
        }
        return null;
      case "recording":
        if (this.xrWanted && !presenting) {
          this.fail("left XR during the recording");
          return "abort";
        }
        if ((tMs - this.recordStart) / 1000 >= this.cfg.durationS) {
          this.enter("saving", tMs);
          return "finish";
        }
        return null;
      default:
        return null;
    }
  }

  /** Rig pose for this frame: the path's start until recording, then the path. */
  pose(tMs: number): RigPose {
    const t = this.phase === "recording" ? (tMs - this.recordStart) / 1000 : 0;
    return this.cfg.path.pose(t);
  }

  /** Whether the rig is driven by the bench (desktop controls off). */
  get drivesRig(): boolean {
    return this.phase !== "done" && this.phase !== "failed";
  }

  done(path: string, summary: Record<string, number>): void {
    this.savedPath = path;
    this.summary = summary;
    this.phase = "done";
  }

  fail(error: string): void {
    this.error = error;
    this.phase = "failed";
  }

  status(): BenchStatus {
    const recordedS =
      this.phase === "recording" ? (this.lastT - this.recordStart) / 1000 : this.phase === "saving" || this.phase === "done" ? this.cfg.durationS : 0;
    return {
      phase: this.phase,
      scene: this.cfg.scene,
      mode: this.cfg.mode,
      path: this.cfg.path.name,
      durationS: this.cfg.durationS,
      recordedS: Math.max(0, recordedS),
      savedPath: this.savedPath,
      error: this.error,
      summary: this.summary,
    };
  }

  /** One HUD line. */
  label(): string {
    const s = this.status();
    const head = `bench ${s.path} ${s.durationS}s`;
    switch (s.phase) {
      case "loading":
        return `${head}: loading scene`;
      case "waiting-xr":
        return `${head}: press Enter VR to start`;
      case "warmup":
        return `${head}: warm-up`;
      case "recording":
        return `${head}: recording ${s.recordedS.toFixed(0)} / ${s.durationS} s`;
      case "saving":
        return `${head}: saving log`;
      case "done":
        return `${head}: saved ${s.savedPath ?? ""}`;
      case "failed":
        return `${head}: FAILED ${s.error ?? ""}`;
    }
  }

  private enter(phase: BenchPhase, tMs: number): void {
    this.phase = phase;
    this.phaseStart = tMs;
  }
}
