// File naming for frame logs. No imports: vite.config.ts (the /__bench endpoint) and the
// viewer (download button) both use it, so a log has the same name either way.

/** Lower-case [a-z0-9_-], at most 40 characters; "unknown" if nothing is left. */
export function sanitizeName(s: string): string {
  const out = s
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return out || "unknown";
}

/** "2026-10-29T21:15:03.120Z" -> "20261029T211503Z". */
export function compactTimestamp(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/.exec(iso);
  return m ? `${m[1]}${m[2]}${m[3]}T${m[4]}${m[5]}${m[6]}Z` : sanitizeName(iso);
}

/** Mode part of a log name: "lod", "naive", "mixed", or "lod-budget" with the controller. */
export function modeTag(mode: string, budget: boolean): string {
  return sanitizeName(budget && mode === "lod" ? "lod-budget" : mode);
}

/** `<timestamp>_<device>_<scene>_<mode>`, without extension. */
export function logStem(log: {
  created: string;
  device: { label: string };
  scene: string;
  mode: string;
  budget?: boolean;
}): string {
  return [
    compactTimestamp(log.created),
    sanitizeName(log.device.label),
    sanitizeName(log.scene),
    modeTag(log.mode, log.budget === true),
  ].join("_");
}

/** Whether a log file name is a run of (device, scene, modeTag), e.g. from the run menu. */
export function isLogOf(file: string, device: string, scene: string, tag: string): boolean {
  const m = /^\d{8}T\d{6}Z_(.+)\.json$/.exec(file);
  if (!m) return false;
  const rest = m[1].replace(/_\d+$/, ""); // "_2" etc. when two logs share a second
  return rest === `${sanitizeName(device)}_${sanitizeName(scene)}_${tag}`;
}

/** Short device label from the user agent; `override` (?device=) wins. */
export function deviceLabel(ua: string, webdriver: boolean, override?: string | null): string {
  if (override) return sanitizeName(override);
  if (webdriver || /HeadlessChrome/.test(ua)) return "headless";
  if (/Pico/i.test(ua)) return "pico";
  if (/OculusBrowser|Quest/i.test(ua)) return "quest";
  if (/Android/.test(ua)) return "android";
  if (/Linux (aarch64|arm)/i.test(ua)) return "linux-arm64"; // Steam Frame, Chromium XR
  if (/Mac OS X/.test(ua)) return "mac";
  if (/Windows/.test(ua)) return "windows";
  if (/Linux/.test(ua)) return "linux";
  return "unknown";
}
