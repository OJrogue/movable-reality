// Lane ui: styles for the DOM overlays (app chrome, selection label, command bar, onboarding
// panel) and the developer-info rules. Colour roles follow Material 3 (dark scheme, tonal
// surfaces, state layers, switches, menus); translucent materials, type and 44 px targets
// follow Apple's Human Interface Guidelines. The touch rules come from lane devices (mobile
// profile, command bar data-touch). Injected once by installUi; classes start with mr-ui-,
// mr-chrome- or mr-debug so nothing else is affected.
export const UI_CSS = `
:root {
  --mr-primary: #b9c3ff; --mr-on-primary: #1a2a6c; --mr-primary-container: #33418a;
  --mr-on-primary-container: #dde1ff; --mr-surface-3: rgba(35, 36, 45, 0.96);
  --mr-on-surface: #e4e1e9; --mr-on-surface-variant: #c6c5d0; --mr-outline: rgba(198, 197, 208, 0.28);
  --mr-glass: rgba(22, 23, 30, 0.62); --mr-glass-border: rgba(255, 255, 255, 0.1);
  --mr-error: #ffb4ab; --mr-error-container: rgba(147, 0, 10, 0.78); --mr-good: #7fdba0;
  --mr-warn: #ffd166;
  --mr-font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI Variable Text", "Segoe UI",
    Roboto, "Helvetica Neue", Arial, sans-serif;
  --mr-ease: cubic-bezier(0.2, 0, 0, 1);
}
.mr-glass, .mr-chrome-group, .mr-ui-row, .mr-ui-onboarding, .mr-ui-line, .mr-chrome-loading-card {
  background: var(--mr-glass); border: 1px solid var(--mr-glass-border);
  backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px);
}
.mr-chrome-top button, .mr-chrome-top a, .mr-chrome-menu button, .mr-chrome-menu a,
.mr-ui-onboarding button, .mr-ui-bar button {
  -webkit-tap-highlight-color: transparent;
}
:is(.mr-chrome-icon, .mr-chrome-title, .mr-chrome-vr, .mr-chrome-item, .mr-ui-chip, .mr-ui-talk):focus-visible {
  outline: 2px solid var(--mr-primary); outline-offset: 2px;
}

/* ---- developer info: hidden until body[data-debug] (ui/chrome.ts) */
body:not([data-debug]) .mr-debug, body:not([data-debug]) #hud { display: none !important; }
body[data-debug] #hud {
  top: calc(76px + env(safe-area-inset-top, 0px)) !important; left: 12px !important;
  font: 12px/1.45 ui-monospace, "SF Mono", Consolas, monospace !important;
  background: rgba(10, 10, 14, 0.72) !important; border: 1px solid var(--mr-glass-border); border-radius: 12px !important;
  padding: 8px 12px !important;
}
.mr-debug-savelog, .mr-debug-detail, .mr-debug-objects {
  right: 12px !important; border-radius: 12px !important; border-color: var(--mr-glass-border) !important;
  background: rgba(10, 10, 14, 0.72) !important; font-family: var(--mr-font) !important;
}
.mr-debug-savelog { top: calc(76px + env(safe-area-inset-top, 0px)) !important; }
.mr-debug-detail { top: calc(116px + env(safe-area-inset-top, 0px)) !important; }
.mr-debug-objects { top: calc(176px + env(safe-area-inset-top, 0px)) !important; }

/* ---- top app bar */
.mr-chrome-top {
  position: fixed; z-index: 1100; left: 12px; right: 12px; top: calc(12px + env(safe-area-inset-top, 0px));
  display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; pointer-events: none;
  font-family: var(--mr-font); color: var(--mr-on-surface);
}
body[data-chrome="0"] .mr-chrome-top, body[data-chrome="0"] .mr-chrome-menu { display: none; }
.mr-chrome-group { display: flex; align-items: center; gap: 2px; padding: 4px; border-radius: 28px; pointer-events: auto; min-width: 0; }
.mr-chrome-icon {
  flex: none; width: 44px; height: 44px; border-radius: 22px; border: 0; padding: 0; display: grid; place-items: center;
  background: transparent; color: var(--mr-on-surface); cursor: pointer; text-decoration: none;
  transition: background 160ms;
}
.mr-chrome-icon:hover, .mr-chrome-title:hover { background: rgba(228, 225, 233, 0.08); }
.mr-chrome-icon:active, .mr-chrome-title:active { background: rgba(228, 225, 233, 0.12); }
.mr-chrome-icon[aria-expanded="true"] { background: rgba(185, 195, 255, 0.16); color: var(--mr-primary); }
.mr-chrome-title {
  display: flex; align-items: center; gap: 6px; min-width: 0; height: 44px; padding: 0 10px 0 12px; border: 0;
  border-radius: 22px; background: transparent; color: inherit; cursor: pointer; font: inherit; text-align: left;
}
.mr-chrome-title-stack { display: flex; flex-direction: column; min-width: 0; line-height: 1.2; }
.mr-chrome-title-text { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mr-chrome-title-sub { font-size: 12px; color: var(--mr-on-surface-variant); white-space: nowrap; }
.mr-chrome-chevron { display: grid; color: var(--mr-on-surface-variant); transition: transform 200ms var(--mr-ease); }
.mr-chrome-title[aria-expanded="true"] .mr-chrome-chevron { transform: rotate(180deg); }
.mr-chrome-vr {
  display: inline-flex; align-items: center; gap: 8px; height: 40px; margin: 0 4px; padding: 0 16px 0 12px;
  border: 0; border-radius: 20px; cursor: pointer; background: var(--mr-primary); color: var(--mr-on-primary);
  font: 600 14px var(--mr-font);
}
.mr-chrome-vr:hover { box-shadow: 0 0 0 4px rgba(185, 195, 255, 0.18); }

/* ---- menus (M3 menu on the HIG translucent material) */
.mr-chrome-menu {
  position: fixed; z-index: 1200; top: calc(70px + env(safe-area-inset-top, 0px)); width: min(340px, calc(100vw - 24px));
  max-height: calc(100vh - 180px); overflow: auto; padding: 8px 0; border-radius: 20px;
  background: var(--mr-surface-3); border: 1px solid var(--mr-glass-border); color: var(--mr-on-surface);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5); font-family: var(--mr-font);
  animation: mr-menu-in 180ms var(--mr-ease); transform-origin: top;
}
.mr-chrome-menu[data-align="left"] { left: 12px; }
.mr-chrome-menu[data-align="right"] { right: 12px; transform-origin: top right; }
.mr-chrome-menu[hidden] { display: none; }
@keyframes mr-menu-in { from { opacity: 0; transform: scale(0.96) translateY(-6px); } }
@media (prefers-reduced-motion: reduce) { .mr-chrome-menu { animation: none; } }
.mr-chrome-section { padding: 12px 20px 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; color: var(--mr-primary); }
.mr-chrome-divider { height: 1px; margin: 8px 0; background: var(--mr-glass-border); }
.mr-chrome-item {
  display: flex; align-items: center; gap: 16px; width: 100%; min-height: 52px; padding: 8px 20px; border: 0;
  background: transparent; color: inherit; font: inherit; text-align: left; cursor: pointer; text-decoration: none;
  box-sizing: border-box;
}
.mr-chrome-item:hover:not(:disabled) { background: rgba(228, 225, 233, 0.08); }
.mr-chrome-item:disabled { cursor: default; opacity: 0.45; }
.mr-chrome-item-text { display: flex; flex-direction: column; flex: 1; min-width: 0; }
.mr-chrome-item-label { font-size: 15px; }
.mr-chrome-item-caption { font-size: 12.5px; color: var(--mr-on-surface-variant); margin-top: 1px; }
.mr-chrome-check { flex: none; width: 22px; height: 22px; color: var(--mr-primary); }
.mr-chrome-switch {
  flex: none; position: relative; width: 52px; height: 32px; border-radius: 16px; box-sizing: border-box;
  border: 2px solid #918f9a; background: #35343c; transition: background 200ms, border-color 200ms;
}
.mr-chrome-switch::after {
  content: ""; position: absolute; top: 6px; left: 6px; width: 16px; height: 16px; border-radius: 50%;
  background: #918f9a; transition: all 200ms var(--mr-ease);
}
.mr-chrome-switch[data-on="1"] { background: var(--mr-primary); border-color: var(--mr-primary); }
.mr-chrome-switch[data-on="1"]::after { left: 22px; top: 2px; width: 24px; height: 24px; background: var(--mr-on-primary); }

/* ---- loading card */
.mr-chrome-loading {
  position: fixed; inset: 0; z-index: 1050; display: grid; place-items: center; pointer-events: none;
  background: radial-gradient(80% 70% at 50% 40%, rgba(40, 46, 96, 0.55), rgba(10, 10, 14, 0.94));
  transition: opacity 400ms var(--mr-ease); font-family: var(--mr-font); color: var(--mr-on-surface);
}
.mr-chrome-loading[data-out] { opacity: 0; }
.mr-chrome-loading-card { pointer-events: auto; width: min(320px, calc(100vw - 48px)); padding: 24px; border-radius: 24px; text-align: center; }
.mr-chrome-loading-title { font-size: 22px; font-weight: 650; letter-spacing: -0.015em; }
.mr-chrome-loading-sub { margin-top: 4px; font-size: 14px; color: var(--mr-on-surface-variant); }
.mr-chrome-progress { position: relative; height: 4px; margin-top: 20px; border-radius: 2px; overflow: hidden; background: rgba(185, 195, 255, 0.18); }
.mr-chrome-progress span {
  position: absolute; top: 0; bottom: 0; width: 40%; border-radius: 2px; background: var(--mr-primary);
  animation: mr-indeterminate 1.4s var(--mr-ease) infinite;
}
@keyframes mr-indeterminate { from { left: -40%; } to { left: 100%; } }
.mr-chrome-loading-back {
  display: inline-flex; align-items: center; min-height: 44px; margin-top: 16px; padding: 0 20px; border-radius: 22px;
  background: var(--mr-primary); color: var(--mr-on-primary); font-weight: 600; font-size: 15px; text-decoration: none;
}

/* ---- selection label */
.mr-ui-label {
  position: fixed; transform: translate(-50%, calc(-100% - 8px)); pointer-events: none;
  padding: 4px 12px; border-radius: 10px; background: var(--mr-primary-container); color: var(--mr-on-primary-container);
  font: 600 13px/1.4 var(--mr-font); white-space: nowrap; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}

/* ---- command bar (M3 search bar shape) */
.mr-ui-bar {
  position: fixed; left: 50%; bottom: calc(20px + env(safe-area-inset-bottom, 0px)); transform: translateX(-50%);
  width: min(640px, calc(100vw - 32px)); font: 14px/1.45 var(--mr-font); color: var(--mr-on-surface); z-index: 1000;
}
.mr-ui-log { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; pointer-events: none; }
.mr-ui-line {
  align-self: flex-start; max-width: 88%; padding: 7px 14px; border-radius: 18px; overflow-wrap: anywhere;
  animation: mr-line-in 220ms var(--mr-ease);
}
@keyframes mr-line-in { from { opacity: 0; transform: translateY(6px); } }
@media (prefers-reduced-motion: reduce) { .mr-ui-line { animation: none; } }
.mr-ui-you { align-self: flex-end; background: rgba(51, 65, 138, 0.86); color: var(--mr-on-primary-container); border-color: transparent; }
.mr-ui-agent { color: var(--mr-on-surface-variant); font-style: italic; }
.mr-ui-status { font-style: normal; display: flex; align-items: center; gap: 8px; }
.mr-ui-status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--mr-primary, #b9c3ff); animation: mr-pulse 1.2s ease-in-out infinite; flex: none; }
@keyframes mr-pulse { 50% { opacity: 0.3; transform: scale(0.7); } }
@media (prefers-reduced-motion: reduce) { .mr-ui-status::before { animation: none; } }
.mr-ui-reply { background: rgba(20, 90, 60, 0.82); border-color: transparent; }
.mr-ui-info { color: var(--mr-on-surface-variant); }
.mr-ui-error { background: var(--mr-error-container); color: #ffdad6; border-color: transparent; }
.mr-ui-row { display: flex; align-items: center; gap: 8px; min-height: 52px; padding: 4px 6px 4px 16px; border-radius: 28px; box-sizing: border-box; transition: border-color 200ms; }
.mr-ui-row:focus-within { border-color: rgba(185, 195, 255, 0.55); }
.mr-ui-bar[data-busy] .mr-ui-row { border-color: rgba(255, 209, 102, 0.75); }
.mr-ui-mic { display: grid; color: var(--mr-on-surface-variant); }
.mr-ui-bar[data-talking] .mr-ui-mic { color: #ff6b6b; }
.mr-ui-bar[data-talking] .mr-ui-row { border-color: #ff6b6b; box-shadow: 0 0 0 4px rgba(255, 107, 107, 0.18); }
.mr-ui-input {
  flex: 1; min-width: 0; padding: 8px 4px; border: 0; outline: 0; background: transparent;
  color: #fff; font: inherit; font-size: 15px;
}
.mr-ui-input::placeholder { color: rgba(198, 197, 208, 0.7); }
.mr-ui-badge {
  flex: none; display: inline-flex; align-items: center; gap: 6px; margin-right: 6px; padding: 4px 10px; border-radius: 14px;
  font-size: 12px; font-weight: 500; color: var(--mr-on-surface-variant); background: rgba(255, 255, 255, 0.06);
}
.mr-ui-badge::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #918f9a; }
.mr-ui-badge[data-state="online"]::before { background: var(--mr-good); box-shadow: 0 0 8px var(--mr-good); }
.mr-ui-badge[data-state="offline"]::before { background: var(--mr-warn); }

/* ---- onboarding card */
.mr-ui-onboarding {
  position: fixed; z-index: 1000; right: 12px; top: calc(76px + env(safe-area-inset-top, 0px)); width: 340px;
  max-height: calc(100vh - 220px); overflow: auto; padding: 18px 20px 20px; border-radius: 24px; box-sizing: border-box;
  color: var(--mr-on-surface); font: 13.5px/1.5 var(--mr-font); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}
body[data-debug] .mr-ui-onboarding { top: calc(236px + env(safe-area-inset-top, 0px)); max-height: calc(100vh - 380px); }
.mr-ui-onboarding[hidden], .mr-ui-help[hidden], .mr-ui-label[hidden], .mr-ui-gizmo[hidden] { display: none; }

/* ---- move / rotate gizmo modes (gizmo.ts), top centre while an object is selected */
.mr-ui-gizmo {
  position: fixed; top: calc(20px + env(safe-area-inset-top, 0px)); left: 50%; transform: translateX(-50%);
  display: flex; gap: 4px; padding: 4px; border-radius: 12px; z-index: 1000;
  background: var(--mr-surface-3); border: 1px solid var(--mr-glass-border); box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}
.mr-ui-gizmo button {
  font: 600 13px/1.4 var(--mr-font); color: var(--mr-on-surface); background: transparent; cursor: pointer;
  border: 0; border-radius: 8px; padding: 6px 14px;
}
.mr-ui-gizmo button.on { background: var(--mr-primary-container); color: var(--mr-on-primary-container); }
.mr-ui-gizmo button.mr-ui-gizmo-remove { color: var(--mr-error); }
.mr-ui-gizmo button.mr-ui-gizmo-remove:hover { background: rgba(255, 180, 171, 0.14); }

/* ---- furniture panel (libraryPanel.ts) */
.mr-ui-lib {
  position: fixed; z-index: 1000; left: 12px; top: calc(76px + env(safe-area-inset-top, 0px)); width: 340px;
  max-height: calc(100vh - 180px); display: flex; flex-direction: column; border-radius: 24px; box-sizing: border-box;
  background: var(--mr-surface-3); border: 1px solid var(--mr-glass-border); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
  color: var(--mr-on-surface); font: 13.5px/1.4 var(--mr-font); overflow: hidden;
}
.mr-ui-lib[hidden] { display: none; }
body[data-debug] .mr-ui-lib { top: calc(236px + env(safe-area-inset-top, 0px)); max-height: calc(100vh - 340px); }
.mr-ui-lib-tabs { flex: none; display: flex; gap: 4px; padding: 8px; border-bottom: 1px solid var(--mr-glass-border); }
.mr-ui-lib-tabs button {
  flex: 1; min-height: 40px; border: 0; border-radius: 16px; background: transparent; color: var(--mr-on-surface-variant);
  font: 600 13.5px var(--mr-font); cursor: pointer;
}
.mr-ui-lib-tabs button:hover { background: rgba(228, 225, 233, 0.08); }
.mr-ui-lib-tabs button.on { background: var(--mr-primary-container); color: var(--mr-on-primary-container); }
.mr-ui-lib-body { overflow: auto; padding: 4px 12px 12px; overscroll-behavior: contain; }
.mr-ui-lib-heading { margin: 12px 4px 8px; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; color: var(--mr-primary); }
.mr-ui-lib-note { padding: 20px 8px; text-align: center; color: var(--mr-on-surface-variant); }
.mr-ui-lib-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.mr-ui-lib-card {
  display: flex; flex-direction: column; align-items: center; min-width: 0; padding: 6px; border: 0; border-radius: 14px;
  background: rgba(228, 225, 233, 0.05); color: inherit; font: inherit; text-align: center; cursor: pointer;
  transition: background 160ms;
}
.mr-ui-lib-card:hover { background: rgba(185, 195, 255, 0.16); }
.mr-ui-lib-card img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; }
.mr-ui-lib-name { font-size: 12.5px; line-height: 1.25; overflow-wrap: anywhere; }
.mr-ui-lib-size { font-size: 11.5px; color: var(--mr-on-surface-variant); }
.mr-ui-lib-row { display: flex; align-items: center; gap: 8px; border-radius: 12px; }
.mr-ui-lib-row[data-removed="1"] .mr-ui-lib-name { opacity: 0.55; text-decoration: line-through; }
.mr-ui-lib-rowname {
  flex: 1; min-width: 0; min-height: 44px; padding: 4px 8px; border: 0; border-radius: 12px; background: transparent;
  color: inherit; font: inherit; text-align: left; cursor: pointer;
}
.mr-ui-lib-rowname:hover:not(:disabled) { background: rgba(228, 225, 233, 0.08); }
.mr-ui-lib-rowname:disabled { cursor: default; }
.mr-ui-lib-rowname .mr-ui-lib-name { font-size: 14px; }
.mr-ui-lib-action {
  flex: none; min-height: 36px; padding: 0 12px; border: 1px solid var(--mr-outline); border-radius: 18px; background: transparent;
  color: var(--mr-on-surface); font: 500 12.5px var(--mr-font); cursor: pointer;
}
.mr-ui-lib-action:hover { background: rgba(228, 225, 233, 0.08); }
:is(.mr-ui-lib-card, .mr-ui-lib-rowname, .mr-ui-lib-action, .mr-ui-lib-tabs button, .mr-ui-gizmo button):focus-visible {
  outline: 2px solid var(--mr-primary); outline-offset: 1px;
}
.mr-ui-ob-head { display: flex; justify-content: space-between; align-items: center; font-size: 17px; font-weight: 650; letter-spacing: -0.01em; }
.mr-ui-ob-intro { margin: 4px 0 12px; color: var(--mr-on-surface-variant); }
.mr-ui-ob-keys { border-collapse: collapse; width: 100%; }
.mr-ui-ob-keys td { padding: 4px 0; vertical-align: top; border-top: 1px solid rgba(255, 255, 255, 0.06); }
.mr-ui-ob-keys td:first-child { color: var(--mr-primary); padding-right: 12px; white-space: nowrap; font-weight: 500; }
.mr-ui-ob-sub { margin: 16px 0 8px; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; color: var(--mr-primary); }
.mr-ui-ob-samples, .mr-ui-ob-toggles { display: flex; flex-wrap: wrap; gap: 8px; }
.mr-ui-ob-toggles { margin-top: 12px; }
.mr-ui-onboarding button, .mr-ui-help {
  font: inherit; color: var(--mr-on-surface); background: transparent; cursor: pointer;
  border: 1px solid var(--mr-outline); border-radius: 8px; padding: 6px 12px; text-align: left;
  transition: background 160ms;
}
.mr-ui-onboarding button:hover, .mr-ui-help:hover { background: rgba(228, 225, 233, 0.08); }
.mr-ui-toggle[data-on="1"] { background: rgba(185, 195, 255, 0.2) !important; border-color: transparent !important; }
.mr-ui-close {
  width: 36px; height: 36px; padding: 0 !important; display: grid; place-items: center; text-align: center !important;
  border: 0 !important; border-radius: 18px !important; font-size: 20px !important; color: var(--mr-on-surface-variant) !important;
}
.mr-ui-help {
  position: fixed; right: 12px; top: 48px; width: 32px; height: 32px; padding: 0;
  border-radius: 50%; background: var(--mr-glass); font: 600 14px var(--mr-font); text-align: center;
}
.mr-ui-talk { display: none; }

/* ---- narrow screens */
@media (max-width: 640px) {
  .mr-chrome-title-sub { display: none; }
  .mr-chrome-vr span { display: none; }
  .mr-chrome-vr { padding: 0 10px; }
}
/* Touch layout (lane devices, mobile profile): narrow portrait screens. */
.mr-ui-bar[data-touch] {
  left: 8px; right: 8px; width: auto; transform: none;
  bottom: calc(8px + env(safe-area-inset-bottom, 0px));
}
.mr-ui-bar[data-touch] .mr-ui-mic { display: none; }
.mr-ui-bar[data-touch] .mr-ui-row { padding: 4px 4px 4px 14px; gap: 6px; }
/* 16 px stops iOS Safari from zooming the page when the box gets focus. */
.mr-ui-bar[data-touch] .mr-ui-input { font-size: 16px; padding: 8px 4px; }
/* Only the status dot on a phone; the text stays for screen readers through title. */
.mr-ui-bar[data-touch] .mr-ui-badge { font-size: 0; gap: 0; padding: 6px; margin-right: 0; }
.mr-ui-bar[data-touch] .mr-ui-line:nth-last-child(n + 4) { display: none; }
.mr-ui-bar[data-touch] .mr-ui-talk {
  display: inline-flex; align-items: center; justify-content: center; gap: 6px; flex: none; min-width: 64px; min-height: 44px;
  padding: 0 14px; border-radius: 22px; border: 0; color: var(--mr-on-primary);
  background: var(--mr-primary); font: 600 13px var(--mr-font);
  touch-action: none; user-select: none; -webkit-user-select: none; -webkit-touch-callout: none;
}
.mr-ui-bar[data-touch][data-talking] .mr-ui-talk { background: #ff6b6b; color: #2a0000; }
body[data-device="mobile"] .mr-chrome-top { left: 8px; right: 8px; top: calc(8px + env(safe-area-inset-top, 0px)); }
body[data-device="mobile"] .mr-ui-onboarding {
  left: 8px; right: 8px; width: auto; top: calc(68px + env(safe-area-inset-top, 0px)); max-height: 48vh; font-size: 14px;
}
body[data-device="mobile"] .mr-ui-onboarding button { padding: 8px 12px; }
body[data-device="mobile"] .mr-ui-lib {
  left: 8px; right: 8px; width: auto; top: calc(68px + env(safe-area-inset-top, 0px)); max-height: 56vh;
}
/* Below the top bar, which fills the phone's width. */
body[data-device="mobile"] .mr-ui-gizmo { top: calc(68px + env(safe-area-inset-top, 0px)); }
body[data-device="mobile"] .mr-ui-gizmo button { min-height: 40px; }
`;
