// Lane renderer: manual detail control. A slider sets the LOD splat count by hand (log
// scale, up to every splat in the scene); "auto" hands it back to the budget controller.

/** Slider positions run 0..SLIDER_STEPS. */
export const SLIDER_STEPS = 1000;
/** Lowest splat count the slider offers. */
export const DETAIL_MIN = 50_000;

/** Splat count for a slider position, geometric between min and total. */
export function countForSlider(
  pos: number,
  total: number,
  min = DETAIL_MIN,
): number {
  const lo = Math.min(min, total);
  const q = Math.min(Math.max(pos / SLIDER_STEPS, 0), 1);
  if (total <= lo) return total;
  return Math.round(lo * (total / lo) ** q);
}

/** Slider position for a splat count (inverse of countForSlider, clamped). */
export function sliderForCount(
  count: number,
  total: number,
  min = DETAIL_MIN,
): number {
  const lo = Math.min(min, total);
  if (!(total > lo) || !(count > 0)) return SLIDER_STEPS;
  const q =
    Math.log(Math.min(Math.max(count, lo), total) / lo) / Math.log(total / lo);
  return Math.round(q * SLIDER_STEPS);
}

/** "1.25M", "740k", "900". */
export function formatSplats(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(Math.round(n));
}

export interface DetailState {
  auto: boolean;
  /** Splat count in use (the LOD budget). */
  count: number;
  /** Splats in the scene (PLY headers), null while loading. */
  total: number | null;
}

/** Panel with an "auto" checkbox and a splat-count slider, top right under the buttons. */
export class DetailControl {
  readonly el = document.createElement("div");
  private readonly auto = document.createElement("input");
  private readonly slider = document.createElement("input");
  private readonly label = document.createElement("span");
  private dragging = false;
  private total: number | null = null;

  constructor(onManual: (count: number) => void, onAuto: () => void) {
    Object.assign(this.el.style, {
      position: "fixed",
      right: "12px",
      top: "100px",
      width: "210px",
      padding: "6px 10px",
      font: "12px system-ui, sans-serif",
      background: "rgba(0,0,0,0.55)",
      color: "#eee",
      border: "1px solid #666",
      borderRadius: "6px",
      zIndex: "1000",
    });
    const head = document.createElement("label");
    Object.assign(head.style, {
      display: "flex",
      alignItems: "center",
      gap: "6px",
    });
    this.auto.type = "checkbox";
    this.auto.title = "Let the frame-rate budget choose the detail";
    head.append("Detail", this.auto, "auto");
    this.label.style.marginLeft = "auto";
    head.append(this.label);
    this.slider.type = "range";
    this.slider.min = "0";
    this.slider.max = String(SLIDER_STEPS);
    this.slider.title = "Splats drawn (right end: every splat in the scene)";
    this.slider.setAttribute("aria-label", "Splats drawn");
    Object.assign(this.slider.style, { width: "100%", margin: "4px 0 0" });
    this.el.append(head, this.slider);
    // Keys typed on the slider (arrows) must not reach the viewer's shortcuts.
    this.el.addEventListener("keydown", (e) => e.stopPropagation());

    this.slider.addEventListener("pointerdown", () => (this.dragging = true));
    window.addEventListener("pointerup", () => (this.dragging = false));
    this.slider.addEventListener("input", () => {
      if (this.total === null) return;
      const n = countForSlider(Number(this.slider.value), this.total);
      this.auto.checked = false;
      this.label.textContent = formatSplats(n);
      onManual(n);
    });
    this.auto.addEventListener("change", () => {
      if (this.auto.checked) onAuto();
      else if (this.total !== null)
        onManual(countForSlider(Number(this.slider.value), this.total));
    });
    document.body.appendChild(this.el);
  }

  update(s: DetailState): void {
    this.total = s.total;
    this.slider.disabled = s.total === null;
    this.auto.checked = s.auto; // a programmatic set fires no change event
    const shown = s.total === null ? s.count : Math.min(s.count, s.total);
    this.label.textContent =
      s.total === null
        ? "loading"
        : `${formatSplats(shown)} / ${formatSplats(s.total)}`;
    if (!this.dragging && s.total !== null) {
      this.slider.value = String(sliderForCount(shown, s.total));
    }
  }

  setVisible(v: boolean): void {
    this.el.style.display = v ? "" : "none";
  }
}
