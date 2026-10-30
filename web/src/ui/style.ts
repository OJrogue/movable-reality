// Lane ui: styles for the DOM overlays (selection label, command bar, onboarding panel).
// The panel sits below the perf lane's "save log" button (top right).
// Injected once by installUi; every class starts with mr-ui- so nothing else is affected.
export const UI_CSS = `
.mr-ui-label {
  position: fixed; transform: translate(-50%, calc(-100% - 6px)); pointer-events: none;
  padding: 2px 8px; border-radius: 4px; background: rgba(255, 200, 61, 0.92); color: #111;
  font: 600 12px/1.5 system-ui, sans-serif; white-space: nowrap;
}
.mr-ui-bar {
  position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%);
  width: min(680px, calc(100vw - 32px)); font: 13px/1.45 system-ui, sans-serif; color: #eee;
}
.mr-ui-log { display: flex; flex-direction: column; gap: 2px; margin-bottom: 6px; pointer-events: none; }
.mr-ui-line {
  align-self: flex-start; max-width: 100%; padding: 3px 9px; border-radius: 6px;
  background: rgba(0, 0, 0, 0.6); overflow-wrap: anywhere;
}
.mr-ui-you { align-self: flex-end; background: rgba(40, 90, 160, 0.75); }
.mr-ui-agent { color: #bbb; font-style: italic; }
.mr-ui-reply { background: rgba(20, 110, 70, 0.8); }
.mr-ui-info { color: #ccc; }
.mr-ui-error { background: rgba(150, 30, 30, 0.8); }
.mr-ui-row {
  display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 10px;
  background: rgba(0, 0, 0, 0.65); border: 1px solid rgba(255, 255, 255, 0.12);
}
.mr-ui-bar[data-busy] .mr-ui-row { border-color: rgba(255, 200, 61, 0.7); }
.mr-ui-mic { color: #666; font-size: 14px; }
.mr-ui-bar[data-talking] .mr-ui-mic { color: #ff4d4d; }
.mr-ui-bar[data-talking] .mr-ui-row { border-color: #ff4d4d; }
.mr-ui-input {
  flex: 1; min-width: 0; padding: 5px 4px; border: 0; outline: 0; background: transparent;
  color: #fff; font: inherit;
}
.mr-ui-input::placeholder { color: #888; }
.mr-ui-badge {
  flex: none; padding: 1px 7px; border-radius: 9px; font-size: 11px; color: #111;
  background: #c9c9c9;
}
.mr-ui-badge[data-state="online"] { background: #5fd38d; }
.mr-ui-badge[data-state="offline"] { background: #f0b45a; }
.mr-ui-onboarding {
  position: fixed; right: 12px; top: 48px; width: 300px; max-height: calc(100vh - 176px);
  overflow: auto; padding: 10px 12px; border-radius: 10px; background: rgba(0, 0, 0, 0.72);
  color: #eee; font: 12px/1.45 system-ui, sans-serif; border: 1px solid rgba(255, 255, 255, 0.12);
}
.mr-ui-onboarding[hidden], .mr-ui-help[hidden], .mr-ui-label[hidden] { display: none; }
.mr-ui-ob-head { display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
.mr-ui-ob-intro { margin: 6px 0 8px; color: #bbb; }
.mr-ui-ob-keys { border-collapse: collapse; width: 100%; }
.mr-ui-ob-keys td { padding: 1px 0; vertical-align: top; }
.mr-ui-ob-keys td:first-child { color: #ffc83d; padding-right: 8px; white-space: nowrap; }
.mr-ui-ob-sub { margin: 10px 0 4px; color: #bbb; }
.mr-ui-ob-samples, .mr-ui-ob-toggles { display: flex; flex-wrap: wrap; gap: 4px; }
.mr-ui-ob-toggles { margin-top: 10px; }
.mr-ui-onboarding button, .mr-ui-help {
  font: inherit; color: #eee; background: rgba(255, 255, 255, 0.1); cursor: pointer;
  border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 12px; padding: 2px 9px;
}
.mr-ui-onboarding button:hover, .mr-ui-help:hover { background: rgba(255, 255, 255, 0.2); }
.mr-ui-toggle[data-on="1"] { background: rgba(255, 200, 61, 0.35); }
.mr-ui-close { border: 0 !important; background: transparent !important; font-size: 16px !important; }
.mr-ui-help {
  position: fixed; right: 12px; top: 48px; width: 28px; height: 28px; padding: 0;
  border-radius: 50%; background: rgba(0, 0, 0, 0.6); font: 600 14px system-ui, sans-serif;
}
`;
