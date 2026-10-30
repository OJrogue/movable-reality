import { describe, expect, it } from "vitest";
import { chunksToWav, concat, encodeWav, resample, toInt16, VOICE_RATE } from "./audio";

describe("concat", () => {
  it("joins chunks in order", () => {
    expect([...concat([new Float32Array([1, 2]), new Float32Array([3])])]).toEqual([1, 2, 3]);
  });
});

describe("resample", () => {
  it("keeps the duration when downsampling 48 kHz to 16 kHz", () => {
    const out = resample(new Float32Array(48000), 48000, 16000);
    expect(out.length).toBe(16000);
  });

  it("averages each window, so a constant stays constant", () => {
    const out = resample(new Float32Array(441).fill(0.5), 44100, 16000);
    expect(out.length).toBe(160);
    for (const s of out) expect(s).toBeCloseTo(0.5, 6);
  });

  it("suppresses a tone at the old Nyquist frequency", () => {
    // 24 kHz at 48 kHz alternates +1, -1; plain decimation by 3 would keep full amplitude.
    const tone = Float32Array.from({ length: 4800 }, (_, i) => (i % 2 ? -1 : 1));
    const out = resample(tone, 48000, 16000);
    const peak = Math.max(...out.map(Math.abs));
    expect(peak).toBeLessThanOrEqual(1 / 3 + 1e-6);
  });

  it("interpolates when upsampling", () => {
    expect([...resample(new Float32Array([0, 1]), 8000, 16000)]).toEqual([0, 0.5, 1, 1]);
  });
});

describe("toInt16", () => {
  it("maps the full range and clips", () => {
    expect([...toInt16(new Float32Array([0, 1, -1, 2, -2, 0.5]))]).toEqual([
      0, 32767, -32768, 32767, -32768, 16384,
    ]);
  });
});

describe("encodeWav", () => {
  it("writes a valid 16 kHz mono 16-bit header and the samples", () => {
    const buf = encodeWav(new Int16Array([1, -2, 300]));
    const v = new DataView(buf);
    const str = (at: number, n: number) =>
      String.fromCharCode(...new Uint8Array(buf, at, n));
    expect(buf.byteLength).toBe(44 + 6);
    expect(str(0, 4)).toBe("RIFF");
    expect(v.getUint32(4, true)).toBe(36 + 6);
    expect(str(8, 8)).toBe("WAVEfmt ");
    expect(v.getUint16(20, true)).toBe(1);
    expect(v.getUint16(22, true)).toBe(1);
    expect(v.getUint32(24, true)).toBe(VOICE_RATE);
    expect(v.getUint32(28, true)).toBe(VOICE_RATE * 2);
    expect(v.getUint16(32, true)).toBe(2);
    expect(v.getUint16(34, true)).toBe(16);
    expect(str(36, 4)).toBe("data");
    expect(v.getUint32(40, true)).toBe(6);
    expect([v.getInt16(44, true), v.getInt16(46, true), v.getInt16(48, true)]).toEqual([1, -2, 300]);
  });
});

describe("chunksToWav", () => {
  it("turns one second at 48 kHz into 16000 samples", () => {
    const wav = chunksToWav([new Float32Array(24000), new Float32Array(24000)], 48000);
    expect(wav.byteLength).toBe(44 + 16000 * 2);
  });
});
