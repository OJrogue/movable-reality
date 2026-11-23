"""Original background music, synthesised with numpy (no samples, no model).

Two beds to compare: "pad" (slow chords only) and "pulse" (chords plus a soft plucked pattern).
"""
import sys
import wave
from pathlib import Path

import numpy as np

SR = 48000
DUR = 182.0
BAR = 8.0  # seconds per chord, about 60 bpm in 4/4 x 2
rng = np.random.default_rng(26)
N = int(SR * DUR)
t = np.arange(N) / SR


def midi(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)


# A minor to F to C to G: open, unresolved, nothing that fights speech.
CHORDS = [[45, 57, 60, 64, 71], [41, 53, 60, 65, 69], [48, 55, 64, 67, 71], [43, 55, 62, 67, 71]]


def lowpass(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s = (1 - a) * v + a * s
        y[i] = s
    return y


def pad():
    out = np.zeros(N)
    n_bars = int(DUR / BAR) + 1
    for b in range(n_bars):
        start = b * BAR
        length = BAR + 3.0
        i0 = int(start * SR)
        n = min(int(length * SR), N - i0)
        if n <= 0:
            break
        tt = np.arange(n) / SR
        env = np.minimum(tt / 2.5, 1) * np.minimum((length - tt) / 3.0, 1)
        env = env * env * (3 - 2 * env)
        for k, m in enumerate(CHORDS[b % 4]):
            f = midi(m)
            amp = (0.5 if k == 0 else 0.28) / (1 + 0.15 * k)
            for det in (-0.0016, 0.0, 0.0019):
                ph = rng.uniform(0, 6.28)
                out[i0:i0 + n] += amp * env * (np.sin(2 * np.pi * f * (1 + det) * tt + ph)
                                               + 0.18 * np.sin(4 * np.pi * f * (1 + det) * tt + ph))
    return out


def pluck():
    out = np.zeros(N)
    step = BAR / 16
    pattern = [0, 2, 3, 4, 1, 3, 2, 4]
    for i in range(int(DUR / step)):
        if i % 2:
            continue
        b = int(i // 16)
        ch = CHORDS[b % 4]
        m = ch[pattern[(i // 2) % 8] % len(ch)] + 12
        i0 = int(i * step * SR)
        n = min(int(1.8 * SR), N - i0)
        if n <= 0:
            break
        tt = np.arange(n) / SR
        env = np.minimum(tt / 0.006, 1) * np.exp(-tt * 3.2)
        f = midi(m)
        out[i0:i0 + n] += (0.55 + 0.25 * rng.random()) * env * (np.sin(2 * np.pi * f * tt) + 0.25 * np.exp(-tt * 9) * np.sin(4 * np.pi * f * tt))
    return out


def reverb(x, secs=2.6, wet=0.35):
    n = int(secs * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 3.0)
    ir /= np.sqrt((ir**2).sum())
    y = np.fft.irfft(np.fft.rfft(x, len(x) + n) * np.fft.rfft(ir, len(x) + n))[: len(x)]
    return x * (1 - wet) + y * wet


def save(name, left, right):
    s = np.stack([left, right], 1)
    fade = np.minimum(np.arange(N) / (3 * SR), 1) * np.minimum((N - 1 - np.arange(N)) / (5 * SR), 1)
    s = s * fade[:, None]
    s = s / np.abs(s).max() * 0.8
    with wave.open(str(name), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((s * 32767).astype(np.int16).tobytes())
    print("saved", name)


OUT = Path("data/broll/_cut")
p = lowpass(pad(), 1500.0)
slow = 0.5 + 0.5 * np.sin(2 * np.pi * t / 37.0)
a_l = reverb(p * (0.9 + 0.1 * slow))
a_r = reverb(p * (0.9 + 0.1 * (1 - slow)))
save(OUT / "music_pad.wav", a_l, a_r)

pl = lowpass(pluck(), 2600.0)
b_l = a_l + 0.22 * reverb(pl * 0.9, wet=0.5)
b_r = a_r + 0.22 * reverb(np.roll(pl, int(0.012 * SR)), wet=0.5)
save(OUT / "music_pulse.wav", b_l, b_r)
sys.exit(0)
