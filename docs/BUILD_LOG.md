# Build log

One entry per work session, newest last. Format: date (Prague), what changed, what is next.
This log is the record of what was built during the hackathon (26 Oct to 4 Nov 2026).

## 2026-10-26 (Mon, evening, after kickoff)

- Created the repository scaffold: agent instructions (`AGENTS.md`, `CLAUDE.md`), docs
  (architecture, plan, hackathon reference), Python package skeleton (config, API clients,
  SAM 3.1 helpers, scene manifest schema), smoke tests for Meta Model API and Baseten,
  frame extraction script, Modal reconstruction job skeleton, Spark-based web viewer
  skeleton. Scaffold drafted with AI assistance.
- Checked: unit tests pass; frame extraction works on a test clip; viewer renders the
  synthetic `demo` scene and applies remove/move edits live (headless Chromium).
  Not yet run: Meta API and Baseten smoke tests (keys pending), Modal job.
- Credits: Baseten $100 approved; Meta API claim pending; Modal and Fireworks requested.
- Next: M0 smoke tests, film 3 rooms, first Modal reconstruction run (M1).

## 2026-10-29 (Thu, early afternoon)

- Updated the plan and agent instructions before the first push: the scaffold above was
  generated on 26 Oct but not yet committed; it enters the repository together with these
  changes in the first commit.
- New scope: coverage loop (find gaps, recapture or generate with Muse Image, tag generated
  splats) and on-device rendering on standalone headsets at 72 FPS (per-object LOD trees,
  baked floor and ceiling, budget controller). Nothing from the original plan is cut.
- New rule from Cerebral Valley (28 Oct): every model called at runtime must be a Meta
  model. MapAnything and SAM 3D Objects are listed as Meta open-source models.
- Work now runs in lanes (core, coverage, renderer, lod, bake) with folder ownership in
  `AGENTS.md`; schema sections for lod, coverage and planes added to `scene.py` and
  `scene.ts`.
- Next: R0 device baseline tonight on the Steam Frame and Pico 4; M1 first real room; lane
  sessions for renderer base, per-object LOD, coverage scoring and plane baking.

## 2026-10-29 (lane: renderer)

- Measurement tools for tonight's R0 device baseline, in `web/src/perf/` and
  `web/src/bench/` (installed from `main.ts` through `installPerf(ctx)`):
  - HUD: last frame, p50 / p99 over 2 s, FPS, missed frames, splats drawn / in the cut /
    total, LOD splat count and minimum splat size, budget state. Desktop in `#hud`; in
    WebXR a canvas-texture panel on the camera. Toggle with `H`, B/Y on a controller, or
    `?hud=0`.
  - Frame log: per-frame timestamps (the XR frame time in XR), interval, main-thread time,
    splats drawn and in the cut, LOD settings, plus device user agent, WebGL renderer, XR
    frame rate, framebuffer and foveation, scene, mode and path. In dev it POSTs to
    `/__bench` (middleware in `vite.config.ts`), which writes
    `benchmarks/raw/<timestamp>_<device>_<scene>_<mode>.json`; headless browsers go to
    `benchmarks/headless/`. Desktop **save log** button and `L` key download or POST.
  - Budget controller (`budget.ts`): pure state machine from frame intervals to
    `lodSplatCount` (150k to 1.5M, 21 geometric levels) and `lodRenderScale` (1 to 3 below
    the middle level). It steps down on missed vsyncs, shedding in proportion on a severe
    overload, then probes up with per-level backoff and a cooldown after each step down. The
    target comes from `XRSession.frameRate` (13.9 ms at 72 Hz, 11.1 ms at 90 Hz) or an
    estimate from steady frames. vitest drives it on a simulated vsynced headset: overload,
    recovery, noise over 10 min, step changes, bounds, 90 Hz.
  - Naive vs LOD toggle: `?mode=naive|lod`, `M`, A/X on a controller. Instant when meshes
    keep their full splat set next to the load-time tree, otherwise a reload.
  - Bench mode: `?bench=1&scene=&mode=&duration=&path=` waits for the scene (and for Enter
    VR when the browser has WebXR), warms up 3 s, moves the rig along a deterministic
    path (`walk`, `orbit`, `pan`, `still`; under 0.5 m/s and 30 deg/s for a standing
    user), then posts the log. `/r0` is a run menu for the headset: one button per R0
    run, ticks for runs already logged.
  - `scripts/bench_make_splats.py`: synthetic 6 x 5 x 2.7 m rooms with 50k to 2M splats
    (floor, ceiling, walls, furniture, clutter, floaters; varied scales and opacities),
    standard 3DGS PLYs in `data/scenes/bench_<N>/`, about 4 s for all six.
  - `scripts/bench_report.py`: p50 / p99, share over budget, FPS first vs last minute and
    drop, pass/fail against the AGENTS.md bar, into `benchmarks/RESULTS.md`.
  - `docs/DEVICES.md`: serving over USB (`adb reverse`) or LAN HTTPS
    (`MR_HTTPS=1`, new dev dependency `@vitejs/plugin-basic-ssl`, MIT), the run menu,
    three ways to get WebXR on the Steam Frame, where logs land, which Spark settings
    change at runtime, and the R0 checklist.
- Spark 2.4.0, read in `dist/spark.module.js`: `lodSplatCount`, `lodSplatScale`,
  `lodRenderScale` and `enableLod` are read on every LOD update and can change at runtime;
  `lodTraverseMode` and the cone foveation fields too, but they need `lodDirty`; `lod` /
  `nonLod` are load-time only; the XR framebuffer scale is fixed per session. Bench mode
  pins `lodSplatCount` at 500k because Spark's default depends on the user agent.
- Tested: vitest 38 tests (controller traces, stats, log names, paths, menu), pytest for
  the generator and the report, `npm run build`, ruff. Headless Chromium (SwiftShader, 2
  vCPU, software rendering) ran `?bench=1&scene=bench_50000&mode=lod&duration=10` and
  `mode=naive` end to end: logs in `benchmarks/headless/` and
  `benchmarks/headless/RESULTS.md`. SwiftShader renders about 1 fps, so those numbers only
  show that the pipeline works; they are not device numbers. Also checked headless: the
  HUD lines and the XR panel layout (`?hudPanel=1`), `M` toggles to the full splat set and
  back, `H`, manual save of a "mixed" log, the `/r0` menu, the 60 Hz rate estimate, and a
  trial merge with `lane/lod` (clean; 60 vitest tests pass; bench run works with
  `installLod`).
- Needs a headset: all frame times; the XR HUD panel's readability and placement; the
  controller buttons (the Chromium XR notes say A/B/X/Y and the thumbstick are untested on
  the Frame); `XRSession.frameRate` and `updateTargetFrameRate(72)` on both browsers; the
  budget controller's thresholds against real frame-time noise; whether WebXR works at all
  in the Frame builds listed in `docs/DEVICES.md`.
- Next: R0 tonight (Pico 4 for sure, the Frame if a Chromium XR build is installed), commit
  the logs and `RESULTS.md`, choose the main device; then R1 with the lod lane's per-object
  trees and R3 tuning of the controller on the walk path.
