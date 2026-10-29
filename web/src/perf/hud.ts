// Performance HUD. Desktop: text in the #hud element. WebXR: the same lines drawn into a
// canvas texture on a quad attached to the camera, readable in the headset.
//
// Shows the last frame time, rolling p50/p99 over ~2 s, FPS, missed-frame share, splats
// drawn (Spark's sort output), splats in the current cut, total splats in the scene, the
// LOD settings and the budget-controller state. Toggle: `H` on the desktop, the left
// controller's X button (or, with one controller, its thumbstick press) in XR, and
// `?hud=0` to start hidden.
import * as THREE from "three";

export interface HudData {
  mode: string;
  xr: boolean;
  targetMs: number;
  hz: number | null;
  lastMs: number;
  p50: number;
  p99: number;
  fps: number;
  missedShare: number;
  drawn: number;
  cut: number;
  total: number | null;
  lodSplatCount: number;
  lodExplicit: boolean;
  lodSplatScale: number;
  lodRenderScale: number;
  budget: string;
  bench: string | null;
  status: string;
}

export function fmtCount(n: number | null): string {
  if (n === null || !Number.isFinite(n)) return "?";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}k`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

const ms = (v: number) => (Number.isFinite(v) ? v.toFixed(1) : "-");

export function hudLines(d: HudData): string[] {
  const hz = d.hz ? `${d.hz} Hz` : "? Hz";
  const over = d.p99 > d.targetMs * 1.2 ? "  OVER" : "";
  const lines = [
    `${d.mode.toUpperCase()}${d.xr ? " XR" : ""}  ${hz}  target ${d.targetMs.toFixed(1)} ms`,
    `frame ${ms(d.lastMs)}  p50 ${ms(d.p50)}  p99 ${ms(d.p99)} ms${over}`,
    `fps ${d.fps.toFixed(1)}  missed ${(d.missedShare * 100).toFixed(1)}%`,
    `splats drawn ${fmtCount(d.drawn)}  cut ${fmtCount(d.cut)}  total ${fmtCount(d.total)}`,
    d.mode === "naive"
      ? "LOD off (full splats)"
      : `LOD count ${fmtCount(d.lodSplatCount)}${d.lodExplicit ? "" : " (default)"}` +
        ` x${d.lodSplatScale}  px ${d.lodRenderScale.toFixed(2)}`,
    `budget ${d.budget}`,
  ];
  if (d.bench) lines.push(d.bench);
  if (d.status) lines.push(d.status);
  return lines;
}

const PANEL_PX = { w: 768, h: 320 };
/** Panel size in metres and placement in front of the left eye, below centre. */
const PANEL_M = { w: 0.32, h: 0.1333 };
const PANEL_POS = new THREE.Vector3(-0.09, -0.12, -0.6);

export class Hud {
  private visible: boolean;
  private readonly el: HTMLElement | null;
  private readonly canvas = document.createElement("canvas");
  private readonly g: CanvasRenderingContext2D | null;
  private readonly texture: THREE.CanvasTexture;
  readonly panel: THREE.Mesh;
  private lastXrDraw = -Infinity;
  private lastText = "";

  constructor(camera: THREE.Camera, visible: boolean) {
    this.visible = visible;
    this.el = document.getElementById("hud");
    this.canvas.width = PANEL_PX.w;
    this.canvas.height = PANEL_PX.h;
    this.g = this.canvas.getContext("2d");
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.generateMipmaps = false;
    const mat = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    this.panel = new THREE.Mesh(new THREE.PlaneGeometry(PANEL_M.w, PANEL_M.h), mat);
    this.panel.position.copy(PANEL_POS);
    this.panel.renderOrder = 10_000; // after the splats, which are transparent too
    this.panel.frustumCulled = false;
    this.panel.visible = false;
    camera.add(this.panel);
    if (this.el && !visible) this.el.style.display = "none";
  }

  get shown(): boolean {
    return this.visible;
  }

  toggle(): void {
    this.setVisible(!this.visible);
  }

  setVisible(v: boolean): void {
    this.visible = v;
    if (this.el) this.el.style.display = v ? "" : "none";
    if (!v) this.panel.visible = false;
  }

  /** Called once per frame; redraws the XR panel at most 4 times a second. */
  update(d: HudData, now: number): void {
    const lines = hudLines(d);
    if (!d.xr) {
      this.panel.visible = false;
      if (this.el && this.visible) {
        const text = lines.join("\n");
        if (text !== this.lastText) this.el.textContent = this.lastText = text;
      }
      return;
    }
    if (this.el) this.el.style.display = "none";
    this.panel.visible = this.visible;
    if (!this.visible || now - this.lastXrDraw < 250) return;
    this.lastXrDraw = now;
    this.draw(lines, d.p99 > d.targetMs * 1.2);
  }

  /** Desktop text only; for the loading phase before the perf loop has numbers. */
  showStatus(text: string): void {
    if (this.el && this.visible && text !== this.lastText) this.el.textContent = this.lastText = text;
  }

  private draw(lines: string[], over: boolean): void {
    const g = this.g;
    if (!g) return;
    const { w, h } = PANEL_PX;
    g.clearRect(0, 0, w, h);
    g.fillStyle = "rgba(0, 0, 0, 0.72)";
    if (typeof g.roundRect === "function") {
      g.beginPath();
      g.roundRect(0, 0, w, h, 18);
      g.fill();
    } else g.fillRect(0, 0, w, h); // older headset browsers
    g.font = "600 30px ui-monospace, Menlo, Consolas, monospace";
    g.textBaseline = "top";
    lines.slice(0, 8).forEach((line, i) => {
      g.fillStyle = i === 1 && over ? "#ff7a6b" : i === 0 ? "#9fe3a8" : "#f2f2f2";
      g.fillText(line, 22, 16 + i * 37, w - 44);
    });
    this.texture.needsUpdate = true;
  }
}
