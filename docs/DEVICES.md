# Devices: running the renderer benchmarks on a headset

Steps for a device session (R0 first, on 29 Oct): serve the viewer from the laptop, open it
in the headset, run the bench scenes, and collect the frame-time logs. Lane renderer.

Devices: **Pico 4** (Pico Browser, always works, the R0 device to start with) and **Steam
Frame** (needs a WebXR build of Chromium, see section 4). Target 72 Hz: p99 frame time at or
under 13.9 ms. Standalone only: never measure through PC streaming (AGENTS.md).

## 1. Prepare the laptop (once)

```bash
uv sync
uv run python scripts/bench_make_splats.py      # data/scenes/bench_{50000..2000000}, ~4 s, 280 MB
cd web && npm install
```

The scenes are synthetic rooms (6 x 5 x 2.7 m: floor, ceiling, walls, furniture boxes,
clutter, a few floaters) in one PLY each. They stay in `data/` (gitignored).

## 2. Serve the viewer to the headset

WebXR needs a secure context: `https://`, or `http://localhost`. Two ways.

### A. USB cable + `adb reverse` (Pico 4; no certificate, works offline)

1. Developer mode on the Pico 4: Settings > General > About, tap **Software Version** about
   8 times; a **Developer** section appears in Settings. In it, turn on **USB debugging**.
2. Connect the Pico to the laptop by USB-C, put the headset on and accept **Allow USB
   debugging** for this computer.
3. On the laptop:
   ```bash
   adb devices                         # the Pico is listed as "device", not "unauthorized"
   adb reverse tcp:5173 tcp:5173
   cd web && npm run dev               # http://localhost:5173
   ```
4. In the headset browser open `http://localhost:5173/r0` (the run menu, section 3).

`adb reverse` lasts until the cable is unplugged or adb restarts; run it again then.

### B. HTTPS on the LAN (Steam Frame, or a Pico without a cable)

```bash
cd web && MR_HTTPS=1 npm run dev     # prints  Network: https://<laptop-ip>:5173/
```

- `MR_HTTPS=1` adds `@vitejs/plugin-basic-ssl` (Vite team, MIT, dev only), which makes a
  self-signed certificate, and serves on all interfaces. The headset browser warns once
  ("Your connection is not private"): choose **Advanced > Proceed to <ip> (unsafe)**. The
  certificate is cached in `web/node_modules/.vite/basic-ssl/`.
- Laptop and headset on the same Wi-Fi; allow port 5173 in the laptop's firewall.
- Use the IP that Vite prints under **Network** (`https://<laptop-ip>:5173/r0`).
- On the Steam Frame, `localhost` would be the Frame itself, so HTTPS on the LAN is the
  way there (or an SSH tunnel: `ssh -R 5173:localhost:5173 frame`, then
  `http://localhost:5173/r0` on the Frame).

Every POSTed log lands on the laptop, whichever way the headset connects.

## 3. Run the benchmarks

Open **`/r0`** in the headset browser (for example `http://localhost:5173/r0`). It lists
every R0 run as a big button, ticks the runs whose logs are already on the laptop, and has
a **next** button. Options: `/r0?duration=60` (shorter runs), `/r0?device=steam-frame`
(file name label; default from the user agent: `pico`, `linux-arm64` for the Frame).

Each run:

1. Tap the run (or **next**). The scene loads; the HUD says `bench walk 120s: loading scene`,
   then `press Enter VR to start`.
2. Press **ENTER VR** (bottom of the page) and allow VR if asked. Stand still, facing
   forward, in a clear space: the bench moves you through the room. Do not walk; small head
   movements are fine (the HUD stays in view).
3. 3 s warm-up, then 120 s recording (`recording 42 / 120 s` in the HUD).
4. At the end the HUD says `saved benchmarks/raw/...json`. Press the headset's menu or
   exit button to leave VR, go **Back** to `/r0`, and take the next run.

If a run says `FAILED` (for example you left VR during the recording), run it again; no
log is written for a failed run.

The bench URL itself, if you need it by hand:

```
/?bench=1&scene=bench_300000&mode=lod&duration=120&path=walk
```

| Parameter | Values | Default |
|---|---|---|
| `scene` | `bench_50000` ... `bench_2000000`, `demo`, any scene id | required |
| `mode` | `lod` (Spark LOD tree built at load time), `naive` (LOD off, full splats) | `lod` |
| `duration` | seconds recorded | `120` |
| `path` | `walk` (loop with a slow view sweep), `orbit`, `pan` (turn on the spot), `still` | `walk` |
| `warmup` | seconds standing at the start before recording | `3` |
| `budget` | `1` runs the budget controller during the bench | off in bench mode |
| `lodSplatCount` | LOD splat budget for the run | `500000` in bench mode |
| `xr` | `1` wait for Enter VR, `0` run on the flat page, `auto` | `auto` |
| `xrScale` | WebXR framebuffer scale 0.25 to 1 | `0.5` (Spark's default) |
| `hz` | ask the session for this rate (if the browser lists it) | the browser's |
| `device` | label for the log file name | from the user agent |
| `hud` | `0` hides the HUD | shown |

Bench mode pins `lodSplatCount` at 500k (unless `lodSplatCount=` or `budget=1` is given),
because Spark's own default depends on the user agent (500k only for the Quest browser,
1M or 2.5M elsewhere; `defaultSplatTarget()` in Spark 2.4.0), and R0 must compare the
devices at the same setting.

## 4. Steam Frame: getting a browser with WebXR

The Frame's built-in browser has no immersive WebXR. Stock Chromium on Linux does not
create an OpenXR device; the builds below add it. **Options 1 and 2 are unofficial
community binaries** (not from Valve or Google). Each runs Chromium with part of its
sandbox turned off so SteamVR accepts the session, so use it for these pages only. **Check
the SHA-256 of the download before installing**:

```bash
sha256sum <downloaded file>     # must equal the value below, character for character
```

The choice is OJ's; this repo does not download or install any of them.

1. **phit/chromium-webxr-steam-frame, release v157.0.8088.0-1** (a fork of
   saphid/chromium-webxr-steam-frame, built by GitHub Actions).
   <https://github.com/phit/chromium-webxr-steam-frame/releases/tag/v157.0.8088.0-1>
   - Asset `chromium-xr-arm64.tar.xz`, SHA-256
     `ecc1fba98df3901e026b29f37f6d9d1090dffa3ae7c5f87cc74e76be72da75ab`.
   - Install with that repo's script, on the Frame (Desktop Mode terminal or SSH):
     `git clone https://github.com/phit/chromium-webxr-steam-frame && chromium-webxr-steam-frame/frame/install.sh ~/chromium-xr-arm64.tar.xz`.
     It adds **Chromium XR** to the Steam library and `~/.local/bin/chromium-xr`.
   - Its release page says it is **not yet tested on the Frame** (the last tested build is
     156.0.8071.0).
2. **yellkell "Frame Tools" Chromium XR, release chromium-xr-frame-157.0.8085.0-5**
   (pre-release test build). <https://yellkell.com/frame>, releases at
   <https://github.com/yellkell/frameplayer/releases>.
   - Asset `ChromiumXR-Frame-arm64.zip`, SHA-256
     `78b376f8609fe069a9f05fcc6c4dd2b6bfbc3f250dd5690c1346d44ac3e24a55`.
   - Installed from Desktop Mode with the page's **Install Frame Tools** link.
   - Its notes: no WebXR layers (pages use the standard WebGL layer, which is what this
     viewer uses), no camera hand tracking (controllers are needed), no DRM. Logs in
     `~/.local/state/chromium-xr-frame/`.
3. **Build from source** with saphid's recipe,
   <https://github.com/saphid/chromium-webxr-steam-frame> (`build/build.sh`), on an x86-64
   Linux machine: about 90 GB of free disk and about 9.5 h on 6 cores for the first build.
   Then `frame/install.sh` on the Frame as in option 1. Not possible before tonight.

With option 1 or 3, saphid's notes (tested on 156.0.8071.0, SteamOS 0.3.0, SteamVR
2.17.10) say: start it from the Steam library (it opens as a panel), open the page, press
**ENTER VR**, answer **Allow VR?** with *Allow*; leave VR with the Steam button. A first
**External Controller Translation** notice can be dismissed with OK. The session renders at
72 Hz with a 2880 x 1440 framebuffer (1440 x 1440 per eye) before Spark's 0.5 scale. From
an SSH shell, `chromium-xr https://<laptop-ip>:5173/r0` opens the menu directly.

If none of these is ready, run R0 on the Pico 4 only and note it in the build log.

## 5. Where the logs land

- Device runs: `benchmarks/raw/<timestamp>_<device>_<scene>_<mode>.json` on the laptop,
  written by the dev server (`POST /__bench`, `web/vite.config.ts`). `<mode>` is `lod`,
  `naive`, or `lod-budget` with the controller on.
- Headless browsers (`HeadlessChrome` or `navigator.webdriver`): `benchmarks/headless/`,
  never `raw/`, so they cannot be mistaken for device numbers.
- Without a dev server (a static build), the log downloads in the browser instead.
- Each log holds per-frame timestamps (the XR frame time in XR), frame interval, main-thread
  time, splats drawn and in the cut, the LOD settings per frame, the device's user agent and
  WebGL renderer, XR frame rate and framebuffer size, and the scene, mode and path.

Then:

```bash
uv run python scripts/bench_report.py           # benchmarks/raw/*.json -> benchmarks/RESULTS.md
git add benchmarks/raw benchmarks/RESULTS.md     # logs are evidence: commit them with the table
```

## 6. Outside bench mode: HUD, toggles, manual logs

Any viewer URL (`/?scene=bench_300000`, `/?scene=demo`) shows the HUD and records frames
from the moment the scene is ready.

| Action | Desktop | XR controller |
|---|---|---|
| naive / LOD toggle (proof shot) | `M` | **A** or **X** |
| HUD on / off | `H` | **B** or **Y** |
| budget controller on / off | `B` | (desktop only) |
| save the log (POST, else download) | `L` or the **save log** button | thumbstick press |

The HUD shows: last frame, p50 / p99 over 2 s, FPS, missed frames (over 1.2x target),
splats drawn / in the cut / total, the LOD splat count and minimum splat size, and the
budget controller's level. In XR it is a panel fixed in front of the left eye, below the
centre of view. The target follows `XRSession.frameRate` when the browser reports it
(13.9 ms at 72 Hz, 11.1 ms at 90 Hz), else the rate estimated from steady frames.

The toggle switches instantly when the meshes hold both their LOD tree and their full splat
set (the default with load-time LOD); with precomputed `.RAD` trees it reloads the page
with the other `?mode=`.

## 7. What can change at runtime (Spark 2.4.0)

Checked in `node_modules/@sparkjsdev/spark/dist/spark.module.js`:

| Setting | Runtime? | Notes |
|---|---|---|
| `SparkRenderer.lodSplatCount`, `lodSplatScale` | yes | read in every `driveLod`; a change marks the LOD dirty and Spark re-traverses (one worker round trip, then a sort) |
| `SparkRenderer.lodRenderScale` | yes | scales the pixel limit; a change re-traverses |
| `SparkRenderer.enableLod` | yes | off = full splat sets (if loaded); this is the naive toggle |
| `SparkRenderer.lodTraverseMode`, `coneFov0`, `coneFov`, `coneFoveate`, `behindFoveate` | yes, but | read on the next traversal; changing them does not mark the LOD dirty, so set `spark.lodDirty = true` |
| `SplatMesh.lodScale`, cone foveation per mesh | yes, but | same: needs `lodDirty` (lane lod does this) |
| `SplatMesh` `lod` / `nonLod` (tree build, keeping the full set) | no | load time only: rebuild the mesh (reload) |
| WebXR framebuffer scale (`?xrScale=`) | no | fixed per session by three.js; exit and re-enter VR |
| Fixed foveation of the XR layer (`renderer.xr.setFoveation`) | per session | three.js sets 1.0 (maximum) where the browser supports it; logged as `display.xr_fixed_foveation` |

The budget controller (`web/src/perf/budget.ts`) changes `lodSplatCount` and
`lodRenderScale` only. Bench mode keeps it off unless `budget=1`, so R0 measures fixed
settings.

## 8. R0 checklist (29 Oct)

2 min each, `path=walk`, bench defaults. Tick in `/r0`; numbers come from `bench_report.py`.

| Scene | Pico 4 naive | Pico 4 lod | Steam Frame naive | Steam Frame lod |
|---|---|---|---|---|
| bench_50000 | [ ] | [ ] | [ ] | [ ] |
| bench_150000 | [ ] | [ ] | [ ] | [ ] |
| bench_300000 | [ ] | [ ] | [ ] | [ ] |
| bench_600000 | [ ] | [ ] | [ ] | [ ] |
| bench_1000000 | [ ] | [ ] | [ ] | [ ] |

20 runs x about 2.5 min = about 50 min of headset time, plus loading. Order: Pico 4 first
(works for sure), small to large, naive before lod for each scene. If time is short, keep
`duration=60`. Optional afterwards: `bench_2000000` both modes, and one `budget=1` run on
the largest scene that fails at fixed settings.

Before you start:

- [ ] `uv run python scripts/bench_make_splats.py` done; `npm run dev` running (USB:
      `adb reverse tcp:5173 tcp:5173`; LAN: `MR_HTTPS=1`)
- [ ] Headset charged, guardian or boundary set, a clear standing space
- [ ] `/r0` opens in the headset browser and says `0 / 10 runs logged`
- [ ] One short test run (`/r0?duration=10`, any scene) saves a log on the laptop

After:

- [ ] `uv run python scripts/bench_report.py` and read `benchmarks/RESULTS.md`
- [ ] Commit `benchmarks/raw/*.json` and `benchmarks/RESULTS.md`
- [ ] Choose the main device; note it and the numbers in `docs/BUILD_LOG.md`

Things to write down during the session (the logs cannot capture them): which browser and
version, whether the session ran at 72 or 90 Hz (also in the log), anything that looked
wrong (flicker, sorting artefacts, the HUD unreadable), and the headset's temperature
warnings if any.
