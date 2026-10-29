import { describe, expect, it } from "vitest";
import { compactTimestamp, deviceLabel, logStem, sanitizeName } from "./logName";

const UA = {
  headless:
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/140.0.0.0 Safari/537.36",
  // Chromium XR on the Steam Frame: SteamOS, arm64.
  frame:
    "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
  pico: "Mozilla/5.0 (Linux; Android 10; A8150 Build/QKQ1.200423.001) AppleWebKit/537.36 (KHTML, like Gecko) PicoBrowser/3.3.22 Chrome/105.0.5195.68 VR Safari/537.36",
  quest:
    "Mozilla/5.0 (X11; Linux x86_64; Quest 2) AppleWebKit/537.36 (KHTML, like Gecko) OculusBrowser/35.0.0 Chrome/128.0.0.0 VR Safari/537.36",
  mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
};

describe("log naming", () => {
  it("labels devices from the user agent", () => {
    expect(deviceLabel(UA.headless, false)).toBe("headless");
    expect(deviceLabel(UA.mac, true)).toBe("headless"); // navigator.webdriver
    expect(deviceLabel(UA.frame, false)).toBe("linux-arm64");
    expect(deviceLabel(UA.pico, false)).toBe("pico");
    expect(deviceLabel(UA.quest, false)).toBe("quest");
    expect(deviceLabel(UA.mac, false)).toBe("mac");
    expect(deviceLabel(UA.frame, false, "Steam Frame")).toBe("steam-frame");
  });

  it("sanitizes names for file paths", () => {
    expect(sanitizeName("../../etc/passwd")).toBe("etc-passwd");
    expect(sanitizeName("bench_300000")).toBe("bench_300000");
    expect(sanitizeName("")).toBe("unknown");
    expect(sanitizeName("x".repeat(99))).toHaveLength(40);
  });

  it("builds <timestamp>_<device>_<scene>_<mode>", () => {
    expect(compactTimestamp("2026-10-29T21:15:03.120Z")).toBe("20261029T211503Z");
    const stem = logStem({
      created: "2026-10-29T21:15:03.120Z",
      device: { label: "pico" },
      scene: "bench_300000",
      mode: "lod",
    });
    expect(stem).toBe("20261029T211503Z_pico_bench_300000_lod");
  });
});
