// Landing page styles. Colour roles follow Material 3 (dark scheme from one seed, tonal
// surfaces, state layers); type, spacing, translucency and 44 px targets follow Apple's
// Human Interface Guidelines. Every class starts with lp- so the viewer is unaffected.
export const LANDING_CSS = `
:root {
  --lp-primary: #b9c3ff; --lp-on-primary: #1a2a6c; --lp-primary-container: #33418a;
  --lp-on-primary-container: #dde1ff; --lp-secondary-container: #3a3f5c;
  --lp-on-secondary-container: #dfe1fb; --lp-tertiary: #ffb59a;
  --lp-surface: #0f1015; --lp-surface-1: #17181f; --lp-surface-2: #1d1e26; --lp-surface-3: #23242d;
  --lp-on-surface: #e4e1e9; --lp-on-surface-variant: #c6c5d0; --lp-outline: #464651;
  --lp-scrim: rgba(8, 9, 14, 0.62);
  --lp-font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI Variable Display",
    "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --lp-ease: cubic-bezier(0.2, 0, 0, 1); /* M3 emphasized */
  color-scheme: dark;
}
html.lp, html.lp body {
  height: auto; overflow: auto; background: var(--lp-surface); color: var(--lp-on-surface);
  font: 17px/1.5 var(--lp-font); -webkit-font-smoothing: antialiased; scroll-behavior: smooth;
}
@media (prefers-reduced-motion: reduce) { html.lp { scroll-behavior: auto; } }
html.lp #hud { display: none; }
.lp a { color: inherit; }
.lp-wrap { width: min(1120px, calc(100% - 40px)); margin: 0 auto; }

/* ---- top bar: translucent material over the video (HIG), small app bar (M3) */
.lp-nav {
  position: fixed; inset: 0 0 auto 0; z-index: 10; height: 60px; display: flex; align-items: center;
  transition: background 240ms var(--lp-ease), border-color 240ms var(--lp-ease);
  border-bottom: 1px solid transparent; padding-top: env(safe-area-inset-top, 0px);
}
.lp-nav[data-solid] {
  background: rgba(15, 16, 21, 0.72); border-bottom-color: rgba(255, 255, 255, 0.08);
  backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px);
}
.lp-nav .lp-wrap { display: flex; align-items: center; gap: 24px; }
.lp-brand { white-space: nowrap; display: flex; align-items: center; gap: 10px; font-weight: 600; letter-spacing: -0.01em; text-decoration: none; }
.lp-mark {
  width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center;
  background: linear-gradient(135deg, #8e9bff, #ffb59a); color: #10122a; font-size: 14px; font-weight: 800;
}
.lp-links { display: flex; gap: 4px; margin-left: auto; }
.lp-links a {
  text-decoration: none; font-size: 15px; color: var(--lp-on-surface-variant); padding: 8px 12px;
  border-radius: 20px; transition: background 160ms, color 160ms;
}
.lp-links a:hover { color: var(--lp-on-surface); background: rgba(228, 225, 233, 0.08); }
@media (max-width: 720px) { .lp-links a:not(.lp-keep) { display: none; } .lp-stats { gap: 20px 28px; } }

/* ---- buttons: M3 filled / tonal / text, 44 px minimum target */
.lp-btn {
  position: relative; display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  min-height: 48px; padding: 0 24px; border-radius: 24px; border: 0; cursor: pointer;
  font: 600 16px/1 var(--lp-font); letter-spacing: 0.01em; text-decoration: none; overflow: hidden; white-space: nowrap;
  transition: box-shadow 200ms var(--lp-ease), transform 200ms var(--lp-ease);
}
.lp-btn::after { /* state layer */
  content: ""; position: absolute; inset: 0; background: currentColor; opacity: 0;
  transition: opacity 160ms;
}
.lp-btn:hover::after { opacity: 0.08; }
.lp-btn:active::after { opacity: 0.12; }
.lp-btn:focus-visible { outline: 3px solid var(--lp-primary); outline-offset: 3px; }
.lp-btn-filled { background: var(--lp-primary); color: var(--lp-on-primary); }
.lp-btn-filled:hover { box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35), 0 8px 24px rgba(142, 155, 255, 0.25); }
.lp-btn-tonal { background: rgba(58, 63, 92, 0.72); color: var(--lp-on-secondary-container); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); }
.lp-btn-small { min-height: 40px; padding: 0 18px; font-size: 15px; }
.lp-btn svg { width: 20px; height: 20px; flex: none; }

/* ---- hero with the showcase video */
.lp-hero { position: relative; min-height: 100svh; display: flex; align-items: flex-end; overflow: hidden; isolation: isolate; }
.lp-hero-media { position: absolute; inset: 0; z-index: -2; background: radial-gradient(120% 90% at 70% 20%, #2a2f63 0%, #14152a 55%, #0b0b10 100%); }
.lp-hero-media video, .lp-hero-media img { width: 100%; height: 100%; object-fit: cover; display: block; }
.lp-hero-media img { animation: lp-drift 30s ease-in-out infinite alternate; }
@keyframes lp-drift { from { transform: scale(1.04) translate(0, 0); } to { transform: scale(1.12) translate(-2%, -1%); } }
@media (prefers-reduced-motion: reduce) { .lp-hero-media img { animation: none; } }
.lp-hero::before {
  content: ""; position: absolute; inset: 0; z-index: -1;
  background:
    linear-gradient(90deg, rgba(8, 9, 14, 0.7) 0%, rgba(8, 9, 14, 0.25) 55%, rgba(8, 9, 14, 0) 100%),
    linear-gradient(180deg, rgba(8, 9, 14, 0.55) 0%, rgba(8, 9, 14, 0.1) 30%, rgba(8, 9, 14, 0.35) 60%, var(--lp-surface) 100%);
}
.lp-hero-body { padding: 120px 0 clamp(48px, 10vh, 112px); }
.lp-eyebrow {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 8px;
  font-size: 13px; font-weight: 600; letter-spacing: 0.02em; color: var(--lp-on-primary-container);
  background: rgba(51, 65, 138, 0.6); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}
.lp-hero h1 {
  margin: 20px 0 16px; max-width: 14ch; font-size: clamp(44px, 7.4vw, 92px); line-height: 1.02;
  font-weight: 700; letter-spacing: -0.035em; text-wrap: balance;
}
.lp-hero h1 em { font-style: normal; background: linear-gradient(90deg, #c3cbff, #ffc2a8); -webkit-background-clip: text; background-clip: text; color: transparent; }
.lp-lede { max-width: 60ch; margin: 0 0 32px; font-size: clamp(17px, 1.6vw, 21px); color: #d9d7e2; text-wrap: pretty; }
.lp-ctas { display: flex; flex-wrap: wrap; gap: 12px; }
.lp-stats { display: flex; flex-wrap: wrap; gap: 32px; margin-top: 48px; color: var(--lp-on-surface-variant); font-size: 14px; }
.lp-stats b { display: block; color: var(--lp-on-surface); font-size: 28px; font-weight: 650; letter-spacing: -0.02em; line-height: 1.2; }
.lp-motion {
  position: absolute; right: 16px; bottom: 16px; width: 44px; height: 44px; border-radius: 22px; border: 0;
  display: grid; place-items: center; cursor: pointer; color: #fff; background: rgba(20, 21, 30, 0.55);
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
}
.lp-motion svg { width: 18px; height: 18px; }
.lp-motion[hidden] { display: none; }

/* ---- sections */
.lp-section { padding: clamp(64px, 11vw, 128px) 0; }
.lp-section + .lp-section { padding-top: 0; }
.lp-kicker { font-size: 14px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--lp-primary); margin: 0 0 12px; }
.lp-h2 { margin: 0 0 16px; font-size: clamp(32px, 4.4vw, 52px); line-height: 1.08; font-weight: 700; letter-spacing: -0.028em; max-width: 22ch; text-wrap: balance; }
.lp-sub { margin: 0 0 48px; max-width: 64ch; color: var(--lp-on-surface-variant); font-size: 19px; text-wrap: pretty; }

.lp-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.lp-card {
  padding: 28px; border-radius: 24px; background: var(--lp-surface-1);
  border: 1px solid rgba(255, 255, 255, 0.05);
}
.lp-card h3 { margin: 16px 0 8px; font-size: 21px; font-weight: 650; letter-spacing: -0.015em; }
.lp-card p { margin: 0; color: var(--lp-on-surface-variant); font-size: 16px; }
.lp-icon {
  width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center;
  background: var(--lp-primary-container); color: var(--lp-on-primary-container);
}
.lp-icon svg { width: 24px; height: 24px; }

/* ---- pipeline stepper */
.lp-steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 12px; counter-reset: step; }
.lp-step {
  display: grid; grid-template-columns: 56px 1fr; gap: 20px; padding: 24px 28px; border-radius: 24px;
  background: var(--lp-surface-1); border: 1px solid rgba(255, 255, 255, 0.05);
}
.lp-step-n {
  counter-increment: step; width: 44px; height: 44px; border-radius: 22px; display: grid; place-items: center;
  background: var(--lp-secondary-container); color: var(--lp-on-secondary-container); font-weight: 700; font-size: 17px;
}
.lp-step-n::before { content: counter(step); }
.lp-step h3 { margin: 8px 0 6px; font-size: 20px; font-weight: 650; letter-spacing: -0.012em; }
.lp-step p { margin: 0 0 12px; color: var(--lp-on-surface-variant); font-size: 16px; max-width: 72ch; }
.lp-flow { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 13px; }
.lp-flow span {
  padding: 4px 10px; border-radius: 8px; background: var(--lp-surface-3); color: var(--lp-on-surface);
  border: 1px solid rgba(255, 255, 255, 0.06); white-space: nowrap;
}
.lp-flow span.lp-model { background: rgba(51, 65, 138, 0.55); color: var(--lp-on-primary-container); border-color: transparent; font-weight: 600; }
.lp-flow i { font-style: normal; color: var(--lp-outline); }
@media (max-width: 560px) {
  .lp-step { grid-template-columns: 1fr; gap: 8px; padding: 20px; }
  .lp-card { padding: 22px; }
}

/* ---- models */
.lp-models { display: grid; gap: 1px; border-radius: 24px; overflow: hidden; background: rgba(255, 255, 255, 0.06); grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.lp-model-card { padding: 22px 24px; background: var(--lp-surface-1); }
.lp-model-card b { display: block; font-size: 17px; font-weight: 650; }
.lp-model-card small { display: block; margin: 2px 0 8px; font: 12px/1.4 ui-monospace, "SF Mono", Consolas, monospace; color: var(--lp-primary); }
.lp-model-card span { color: var(--lp-on-surface-variant); font-size: 15px; }
.lp-note { margin: 20px 0 0; color: var(--lp-on-surface-variant); font-size: 14px; }

/* ---- scene cards (page and dialog) */
.lp-scenes { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.lp-scene {
  position: relative; display: flex; flex-direction: column; text-align: left; padding: 0; border: 0;
  border-radius: 24px; overflow: hidden; cursor: pointer; color: inherit; font: inherit;
  background: var(--lp-surface-2); text-decoration: none; isolation: isolate;
  transition: transform 260ms var(--lp-ease), box-shadow 260ms var(--lp-ease);
}
.lp-scene:hover { transform: translateY(-3px); box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45); }
.lp-scene:focus-visible { outline: 3px solid var(--lp-primary); outline-offset: 3px; }
.lp-scene[aria-disabled="true"] { cursor: default; }
.lp-scene[aria-disabled="true"]:hover { transform: none; box-shadow: none; }
.lp-captures { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-top: 32px; }
.lp-capture { margin: 0; border-radius: 20px; overflow: hidden; background: var(--lp-surface-1); border: 1px solid rgba(255, 255, 255, 0.08); }
.lp-capture video, .lp-capture-empty { display: block; width: 100%; aspect-ratio: 16 / 9; background: #000; object-fit: cover; }
.lp-capture-empty { display: grid; place-items: center; color: rgba(255, 255, 255, 0.55); font-size: 14px; background: linear-gradient(135deg, #2b3170, #523a4a); }
.lp-capture figcaption { display: grid; gap: 4px; padding: 14px 18px 18px; font-size: 14px; line-height: 1.45; }
.lp-capture figcaption span { opacity: 0.7; }
.lp-scene-img { aspect-ratio: 16 / 10; background: linear-gradient(135deg, #2b3170, #523a4a); position: relative; }
.lp-scene-img img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 600ms var(--lp-ease); }
.lp-scene:hover .lp-scene-img img { transform: scale(1.04); }
.lp-scene[aria-disabled="true"] .lp-scene-img img { filter: grayscale(0.7) brightness(0.6); }
.lp-scene-body { padding: 18px 20px 20px; }
.lp-scene-body b { display: block; font-size: 19px; font-weight: 650; letter-spacing: -0.012em; }
.lp-scene-body span { display: block; margin-top: 4px; color: var(--lp-on-surface-variant); font-size: 15px; }
.lp-chip {
  position: absolute; left: 12px; top: 12px; padding: 4px 10px; border-radius: 8px; font-size: 12px; font-weight: 600;
  background: rgba(15, 16, 21, 0.7); color: var(--lp-on-surface); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}
.lp-chip[data-kind="real"] { background: rgba(255, 181, 154, 0.88); color: #3a1406; }
.lp-chip-state { left: auto; right: 12px; }
.lp-chip-state[data-state="processing"], .lp-chip-state[data-state="updating"] { background: rgba(255, 214, 102, 0.9); color: #2d2200; }
.lp-chip-state[data-state="missing"] { background: rgba(70, 70, 81, 0.9); }
.lp-chip-state[data-state="ready"] { display: none; }

/* ---- dialog: M3 dialog on wide screens, full-height sheet on phones (HIG) */
.lp-dialog {
  width: min(960px, calc(100vw - 32px)); max-height: calc(100svh - 48px); padding: 0; border: 0; border-radius: 28px;
  background: var(--lp-surface-2); color: var(--lp-on-surface); box-shadow: 0 24px 64px rgba(0, 0, 0, 0.6);
}
.lp-dialog::backdrop { background: var(--lp-scrim); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); }
.lp-dialog[open] { animation: lp-pop 280ms var(--lp-ease); }
@keyframes lp-pop { from { opacity: 0; transform: translateY(16px) scale(0.98); } }
@media (prefers-reduced-motion: reduce) { .lp-dialog[open] { animation: none; } }
.lp-dialog-inner { padding: 28px; }
.lp-dialog-head { display: flex; align-items: flex-start; gap: 16px; margin-bottom: 24px; }
.lp-dialog-head h2 { margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.02em; }
.lp-dialog-head p { margin: 6px 0 0; color: var(--lp-on-surface-variant); font-size: 15px; }
.lp-close {
  margin-left: auto; flex: none; width: 44px; height: 44px; border-radius: 22px; border: 0; cursor: pointer;
  display: grid; place-items: center; color: var(--lp-on-surface-variant); background: transparent;
}
.lp-close:hover { background: rgba(228, 225, 233, 0.08); }
.lp-close svg { width: 22px; height: 22px; }
.lp-devices { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 24px; color: var(--lp-on-surface-variant); font-size: 14px; }
@media (max-width: 600px) {
  .lp-dialog { width: 100vw; max-width: 100vw; max-height: 100svh; height: 100svh; margin: 0; border-radius: 0; }
  .lp-dialog-inner { padding: 20px 16px calc(20px + env(safe-area-inset-bottom, 0px)); }
}

.lp-footer { padding: 40px 0 56px; border-top: 1px solid rgba(255, 255, 255, 0.06); color: var(--lp-on-surface-variant); font-size: 14px; }
.lp-footer .lp-wrap { display: flex; flex-wrap: wrap; gap: 12px 32px; justify-content: space-between; }
.lp-footer a { color: var(--lp-primary); text-decoration: none; }
`;
