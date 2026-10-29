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

/** `<timestamp>_<device>_<scene>_<mode>`, without extension. */
export function logStem(log: {
  created: string;
  device: { label: string };
  scene: string;
  mode: string;
}): string {
  return [
    compactTimestamp(log.created),
    sanitizeName(log.device.label),
    sanitizeName(log.scene),
    sanitizeName(log.mode),
  ].join("_");
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
