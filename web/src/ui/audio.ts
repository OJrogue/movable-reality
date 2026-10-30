// Lane ui: pure audio helpers for hold-to-talk. The voice endpoint takes WAV, mono, 16-bit
// PCM at 16 kHz (docs/SERVER_API.md), so the browser resamples and encodes it itself.

export const VOICE_RATE = 16000;

/** Concatenates recorded chunks into one buffer. */
export function concat(chunks: Float32Array[]): Float32Array {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/**
 * Resamples mono float audio from `fromRate` to `toRate`. Downsampling averages the input
 * samples that fall into each output sample's window (a box filter, which suppresses most of
 * the aliasing that plain decimation would add to speech). Upsampling interpolates linearly.
 */
export function resample(input: Float32Array, fromRate: number, toRate = VOICE_RATE): Float32Array {
  if (fromRate === toRate) return input.slice();
  const ratio = fromRate / toRate;
  const n = Math.floor(input.length / ratio);
  const out = new Float32Array(n);
  if (ratio > 1) {
    for (let i = 0; i < n; i++) {
      const start = Math.floor(i * ratio);
      const end = Math.min(input.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      for (let j = start; j < end; j++) sum += input[j];
      out[i] = end > start ? sum / (end - start) : 0;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const x = i * ratio;
      const j = Math.floor(x);
      const f = x - j;
      const b = input[Math.min(j + 1, input.length - 1)];
      out[i] = input[j] * (1 - f) + b * f;
    }
  }
  return out;
}

/** Float samples in [-1, 1] to signed 16-bit, clipping out-of-range values. */
export function toInt16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return out;
}

/** A RIFF/WAVE file: 44-byte header, mono, 16-bit little-endian PCM. */
export function encodeWav(samples: Int16Array, rate = VOICE_RATE): ArrayBuffer {
  const dataBytes = samples.length * 2;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);
  const ascii = (at: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(at + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  v.setUint32(4, 36 + dataBytes, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  v.setUint32(16, 16, true); // fmt chunk size
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); // byte rate
  v.setUint16(32, 2, true); // block align
  v.setUint16(34, 16, true); // bits per sample
  ascii(36, "data");
  v.setUint32(40, dataBytes, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, samples[i], true);
  return buf;
}

/** Recorded chunks at `rate` to a 16 kHz WAV, ready to upload. */
export function chunksToWav(chunks: Float32Array[], rate: number): ArrayBuffer {
  return encodeWav(toInt16(resample(concat(chunks), rate, VOICE_RATE)), VOICE_RATE);
}
