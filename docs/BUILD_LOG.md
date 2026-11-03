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

## 2026-10-29 (lane: lod)

- Read Spark 2.4.0's LOD path end to end (npm `dist/` and the GitHub tag `v2.4.0`) and
  wrote it up in `docs/LOD.md`: load-time `lod: true` vs precomputed `.RAD`, one shared
  budget across all LOD meshes, what `lodScale` does in the joint traversal, paging, and
  what a move does (no rebuild; Spark just does not re-traverse on its own).
- Precomputed trees work: Spark's `build-lod` (Rust, MIT) builds CPU-only with cargo in
  1.5 min. `scripts/build_lod.py` sets it up in a cache outside the repo (version from
  `web/package.json`), writes `lod/<stem>.rad` per object and for the background, sets
  `SceneObject.lod` / `Scene.background_lod`, skips fresh trees unless `--force`, and
  keeps agent edits made during a build. Helpers in `src/movable_reality/lod/`: RAD
  header reader, per-object PLY split (keeps SH and `generated`), build wrapper.
- Viewer: `installLod(ctx)` in `web/src/lod/`, wired into `main.ts`. `.RAD` when the
  manifest has one, else `lod: true`; `?mode=naive` leaves LOD off. The mouse or controller
  ray (manifest bboxes, pivot frame) and the latest edit ramp `lodScale` to 2.0 over
  300 ms; it sets `spark.lodDirty` so the cut follows priority and edits.
- Tested: `uv run pytest` (new `test_lod_*`, synthetic PLYs, a fake and the real
  build-lod), vitest for priority, ramp, picking and options, `npm run build`. Headless
  Chromium (SwiftShader) on `demo`: four LOD meshes from `.RAD` and from load-time trees;
  at a 20,000-splat budget hovering the table moves its cut from 2,321 to 3,984 splats
  while the total stays 20,000; `edit_scene.py` moves and a rotate add traversals but no
  `initLodTree`, `loadPackedSplats` or LOD build; hiding an object for over 3 s drops its
  worker tree and restoring re-inits it (Spark's `lodCleanupTimeoutMs`); chunked `.RAD`
  paging streams on the dev server. build-lod timing: 1M splats with SH 3, 27.6 s
  `--quality`, 8.8 s `--quick` on 2 vCPU.
- Needs a headset or real scenes: frame time with LOD vs naive on the Steam Frame and
  Pico 4, controller-ray priority in XR, the right priority scale and per-device
  `lodSplatCount` (Spark's platform default does not recognise either headset), tree
  build times on real rooms.
- Next: run `build_lod.py` after `lift_ids` on the first real room (M2); renderer lane to
  own `lodSplatCount` and `lodCleanupTimeoutMs`; `.RAD` for `completed_ply` when M4 lands.
## 2026-10-29 (lane: bake)

- Plane baking (stage 12), CPU path end to end. `src/movable_reality/bake/`: RANSAC plane
  fit on Gaussian centres with least-squares refinement and the normal oriented into the
  room; oriented bounding rectangle from PCA (largest connected cluster, trimmed extents,
  angle sweep for near-square rooms); a numpy reference rasterizer that splats each plane
  Gaussian's covariance onto the plane orthographically, front to back as seen from inside
  the room, with gsplat's classic-mode rules and degree-0 SH colour; push-pull hole
  filling; a hand-written GLB writer (one unlit, opaque, textured quad, no new
  dependency). `scripts/bake_planes.py --scene <id>` writes `planes/<label>.glb` in the
  recon frame and `objects/background_baked.ply` next to the original, and records
  `planes` and the new `plane_bake` section in scene.json (background_ply points at the
  reduced PLY). Idempotent; `--force` bakes again from the original background, `--revert`
  goes back to splats only (for the benchmark ablation).
- Synthetic test room (`--synthetic`): 4 x 5 x 2.5 m, patterned semi-transparent floor and
  ceiling Gaussians with noise and 3% mislabelled outliers, a doorway island, walls, three
  boxes on the floor and no floor under the sofa, in the recon frame with a tilted, yawed
  world_from_recon. Bake: floor and ceiling normals within 0.2 deg, offsets within 3 mm,
  footprint within 8 cm (tests); 42,315 of 71,484 splats replaced; 2.4 s per plane at 5 mm
  texels; a denser room (308k floor and 143k ceiling splats) bakes in 9.4 s on CPU.
- Viewer: `web/src/planes/installPlanes(ctx)` loads the quads with GLTFLoader under the same
  world_from_recon group as the splats, opaque, depthWrite on, renderOrder -100. It needs no
  SceneView change: the reduced background comes in through background_ply, so the lod
  lane's meshOptions keep working. A background_lod built before the bake still holds the
  plane splats; the viewer warns, and `build_lod.py` rebuilds it.
- Key risk checked: Spark 2.4.0 splats depth-test against opaque three.js meshes.
  SparkRenderer's material is transparent, depthTest true, depthWrite false (source,
  `spark.module.js`), so it draws in three.js's transparent pass after the opaque quads.
  Headless Chromium (SwiftShader) on `src/planes/depthcheck.html`: splats behind the quad
  are hidden, splats in front are drawn, and a wall crossing the quad's depth switches at
  the intersection. On `?scene=bakedemo` the floor stays complete where the sofa stood
  after `edit_scene.py move`, where the splat-only scene shows a hole.
- GPU path: `modal_jobs/bake.py` (gsplat 1.5.3 pinned, `camera_model="ortho"`, covariances
  passed directly). Camera convention checked against gsplat's source and tested on CPU;
  the job itself was not run (no Modal credentials in this session).
- Tests: `tests/test_bake_*.py` (30) and the plane section of `test_scene.py`;
  `web/src/planes/planes.test.ts` (8).
- Needs real scenes: `lift_ids` output (`ids/gaussian_ids.npy` and an `ids/labels.json`
  that maps ids to SAM object ids, a convention this lane proposes), real floor noise and
  SH view dependence, seams at walls and under furniture. Needs a headset: frame time
  with and without planes (ablation row "+ baked planes") and how the 5 mm texture looks
  at grazing angles.
- Next: run the bake on the first real room, check the Modal job, then measure the ablation
  row on the Steam Frame.
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

## 2026-10-29 (orchestration, evening)

- Reviewed and merged PRs #1 (lod), #2 (bake) and #3 (renderer) into main; conflicts in
  `AGENTS.md`, `docs/BUILD_LOG.md` and `web/src/main.ts` resolved by keeping both sides.
  Combined checks: 70 Python and 69 viewer tests pass; headless Chromium loads the baked
  synthetic room with LOD trees, baked planes, HUD and a bench run.
- Wrote `docs/HANDOFF.md` for the local continuation: state, decisions D1-D14, tonight's
  runbook (setup, filming, R0, M1), tasks per lane, how to run lanes locally.
- Next: filming and R0 tonight on the PC and both headsets; coverage PR; core lane M2.

## 2026-10-29 (lane: coverage)

- Coverage scoring (C1) in `src/movable_reality/coverage/`, run by
  `scripts/score_coverage.py`. Per Gaussian it counts the training views that see it
  (projection plus a coarse depth test) and the angle spread between them. A 10 cm voxel
  grid carries the scores, and exposure rays from walkable standing positions decide which
  voxels a VR user can see. Poorly covered visible voxels inside the room box cluster into
  gaps (wide ones split), each with a label from the nearest object and up to three
  next-best views from walkable positions. Results go to `coverage/{gaussians,gaps}.npz`,
  `coverage/report.json` and the `coverage` section of `scene.json` (new fields
  `Gap.next_best_views`, `Coverage.gap_share` in `scene.py` and `scene.ts`).
- Synthetic capture with ground truth (`scripts/coverage_synthetic.py`, scene `cov_synth`):
  a 208k-Gaussian room, 150 views on a perimeter walk, and hidden regions behind the sofa,
  under the desk and coffee table, on top of the bookshelf and on the ceiling. The
  synthetic splats take their opacity and noise from the exact view count, and never-seen
  splats get opacity 0.3 to 0.6, under the 0.5 occluder threshold. That makes the depth
  test easier than on a real gsplat result.
- Results on `cov_synth` (`coverage/eval.json`, `coverage/report.json`), 1-voxel tolerance.
  - Against the exact geometric score (views and angle spread): precision 0.949, recall
    0.986.
  - Against the exact view counts alone: precision 0.826, recall 0.982.
  - 2,371 of 8,394 gap voxels (28%) hold no VR-visible true sample and are not scored.
  - Region recall is 1.0 behind the sofa, under the coffee table, on the bookshelf and on
    the ceiling, and 0.856 under the desk.
  - 23 gaps. `gap_share` 0.47, the share of VR-visible non-empty voxels (surfaces and
    unobserved space) inside gaps.
  - Stage timings in `report.json` add up to 3.5 s on this Mac.
- Fix after review: the voxel grid rounds up past the room bounds, so its last layer lay
  behind the walls and above the ceiling and became gap candidates. Candidates now need a
  centre within half a voxel of the room box. Before the fix there were 37 gaps and
  precision 0.907 against the score truth, and under-desk recall was 0.956 (part of it from
  voxels behind the wall).
- U-3DGS-style uncertainty job (`modal_jobs/coverage.py`): frozen scene, per-Gaussian SH
  uncertainty fitted to the L1 + DSSIM residual with gsplat. The numpy reference and CPU
  tests cover the maths. Not yet run on Modal.
- Viewer gap overlay (`web/src/coverage/`, `installCoverage` in `main.ts`): rim-glow spheres
  per open gap, orange once a recapture is requested, floor ring and arrow for the best next
  view. `?gaps=1`, key G, XR squeeze. Checked once in the desktop app's browser pane on
  `cov_synth`: spheres render, G toggles them, no console errors (screenshots not kept).
  Not yet seen on a headset.
- Repair groundwork (C3, `coverage/repair.py`, `scripts/coverage_repair.py`): renders a gap
  view with the gap painted magenta, picks the three nearest real frames as references and
  sends one Muse Image edit per gap, cached under `data/cache/muse_image/`. Next-best views
  stand so close that a 0.75 m gap filled the whole frame, so the repair camera now backs
  off horizontally at eye height, inside the room bounds and 0.3 m short of the first
  opaque splats, until the gap spans half the vertical half FOV. The first version backed
  off in 3D and put three of five ceiling-gap cameras at y = 0.15 m, one inside the sofa.
  Gaps without a next-best view are skipped. Dry run on `cov_synth`: three gap views, all
  eyes at 1.6 m. No live Muse Image call yet (no `.env` on this machine). The CPU preview
  draws splats as squares, so sparse regions show holes. A gsplat render is the better
  input.
- Tested: pytest 121 passed, 1 skipped (lod lane); vitest 82; `npm run build`; ruff clean.
  Reviewer agent verdict before the fixes: ready after fixes.
- Next: one live `--call` once `MODEL_API_KEY` is set, then score room01 after M1. C2
  (recapture loop) and C3 distillation of repaired views back into generated splats.

## 2026-10-29 (Claude Code, whole-repo review after the move from Cowork)

- Merged PR #4 (coverage). On main, pytest gives 121 passed, 1 skipped, and vitest 82.
- Two reviewer agents read all Python and viewer code on main. Fixed from their findings:
  - Reconstruction job: gsplat now builds with `--no-build-isolation` against the image's
    torch 2.9.1, with the glm submodule checked out. Both upstream repos are pinned
    (MapAnything 3d10cf7, gsplat 512d366). Changed frames force a new reconstruction, and
    old frames are removed from the volume. Still not run on Modal.
  - `extract_frames.py` always clears frames left by an earlier or crashed run.
  - `sam.py` uploads the video again once when a remembered file id is rejected.
  - Viewer: a failed splat load no longer stalls the HUD, the budget controller and bench
    runs. Polls of `scene.json` no longer overlap. A plane removed while loading is
    dropped. A bake that changes `background_ply` replaces the background mesh.
  - Renderer: the budget controller starts at its own start level (474k) instead of
    Spark's default (2.5M on the Steam Frame). The dev server answers Range requests for
    paged `.rad` files. The naive toggle no longer reloads, and so no longer ends, an XR
    session. Dead code removed.
  - `MapAnything` ID in `config.py`, `.glb`/`.rad` in `.gitignore`, argument checks in
    `edit_scene.py`, demo object PLYs named `objects/obj_<id>.ply`.
- Docs: HANDOFF updated for the local setup (Mac, missing tools listed) and the merged
  coverage lane. Cloud-session notes and a duplicated check list removed.
- Not changed, recorded for later: four copies of the 3DGS PLY writer (merge after the
  freeze), `modal_jobs/bake.py` lacks `--max-tilt` and `--format`.
- Tested: pytest, ruff, vitest, `npm run build`, `tsc --noEmit`, `make_demo_scene.py`.
  In the desktop app's browser pane, `?scene=demo` renders, and a copy of the demo with one
  object PLY deleted shows the 404 in the status line while the HUD and budget controller
  run. Not run on a headset.
- Added `.claude/launch.json` (dev server for Claude Code's browser preview).
- Decisions with OJ, recorded as D15 to D17 in `docs/HANDOFF.md`: evaluation metrics
  with no Meta equivalent (LPIPS) are allowed; the build order follows the headset days
  (`docs/PLAN.md`, "Build order"); C2 builds a full reconstruction rerun first and the
  fast fine-tune path second. The Muse Code notes in README and AGENTS.md are shorter.

## 2026-10-30 (Fri, Windows PC setup, all lanes)

- Set up the Windows PC (RTX 3060 Laptop 6 GB, 16 GB RAM): `uv sync`, `npm ci`, ffmpeg via
  winget, `build_lod.py --setup` (Spark v2.4.0 `build-lod.exe`). The keys file had been saved
  as `env`, which `.gitignore` does not cover; renamed to `.env`.
- Windows fixes (7 tests failed before, all real or test-only Windows issues):
  - `plyfile` memory-maps binary PLYs by default; on Windows a mapped file cannot be
    replaced, so `write_generated` and the bench writer failed. All reads use `mmap=False`.
  - Manifest paths were built with `str(Path)`, giving `objects\background_baked.ply` in
    `scene.json` on Windows; now `as_posix()` (bake, coverage repair references).
  - `build-lod` lookup adds `.exe` on Windows. The four lod tests whose fake `build-lod` is a
    `/bin/sh` script skip on Windows; the test with the real binary runs and passes.
  - Now: pytest 117 passed, 5 skipped; ruff clean; vitest 82 passed; `npm run build` OK.
- Live key checks (about $0.03 in total):
  - Meta Model API: 9 models listed, including every ID in `config.py`; Muse Wonder replies.
  - Muse Image `images.generate` made a living-room photo; SAM 3.1 found and masked the
    chair in it (`data/smoke/sam_chair.png`).
  - Muse Voice Transcribe: 404 on the OpenAI audio path; works on `POST /v1/asr/transcribe`
    (multipart, 16 kHz mono WAV). Transcript of a test sentence came back right. Details
    in HANDOFF section 7.
  - Baseten Muse Spark Open replies. Fireworks key valid. HF token valid; the gated
    `facebook/sam-3d-objects` checkpoints download with it (401 without).
  - Modal: no token on this PC yet.
- Steam Frame over `ssh frame`: SteamOS aarch64, SteamVR running, Chrome XR installed. A
  reverse SSH tunnel serves the PC's dev server at the Frame's `http://localhost:5173`
  (checked with curl on the Frame). `?scene=demo` with LOD trees renders in the PC browser.
  Not yet opened in the headset itself.
- Scope change (D18): the viewer becomes a full desktop-browser product too (mouse and
  keyboard editing, command bar, hold-to-talk). New lane `ui`, tasks U1-U4 in HANDOFF.
- Analysed a Blender ground-truth room as a second demo scene (recommendation in the
  session; waiting for OJ's decision, listed as open in HANDOFF).
- Next: Modal token and the first M1 run; filming; R0 on the Frame through the tunnel; U1.

## 2026-10-30 (Fri, afternoon, core: first Modal run of reconstruct.py)

- Test video (private, never published) at `data/raw/`, scene `testroom`: 89 s, 1080p30;
  `extract_frames.py` picked 200 frames (the cap). Frames show strong motion blur: the real
  capture must be slower, ideally 60 fps with locked exposure.
- Modal image for `modal_jobs/reconstruct.py` now builds (41 min once, then cached):
  - The first build compiled gsplat for four GPU architectures and showed no progress for
    2 h; stopped. Modal's image builder has **1 CPU**, so `MAX_JOBS` does not help.
    `CUDA_ARCHS` is now `8.9` (L40S only), and pip runs with `-v` so nvcc progress shows.
  - The link failed with `clang++: No such file or directory`: `add_python`'s interpreter
    was built with clang, so setuptools defaults to clang++. The step now sets
    `CC`/`CXX`/`LDSHARED`/`LDCXXSHARED` to gcc/g++, and an import check of
    `gsplat.cuda._backend._C` runs right after the build.
  - The job then stopped before running: Modal needs a payment method on the account for
    L40S functions (OJ to add). `coverage.py` and `bake.py` build gsplat the same way and
    need the same two fixes before their first run.
- Next: rerun `modal run modal_jobs/reconstruct.py --scene testroom --max-steps 7000`
  once billing is set; the image is cached, so it starts on the GPU directly.

## 2026-10-30 (lane: blender)

- New benchmark and second demo scene, labelled synthetic everywhere; the real phone
  capture stays the main demo. Blender 4.5.10 LTS installed (winget), Cycles on the RTX
  3060 Laptop through OptiX.
- `scripts/blender_assets.py` downloads 12 CC0 Poly Haven assets (10 models, floor and wall
  textures) into `data/assets/`; list with authors in `docs/ASSETS.md`.
- `scripts/blender_room.py` builds a 4 x 5 x 2.6 m living room: 7 movable pieces (sofa,
  coffee table, armchair, side table, procedural floor lamp, cabinet with its back to the
  wall, plant) and 4 small items on surfaces (books, vase, succulent, bowl). Lamp-lit, no
  mirrors or glass. Two handheld walks with shake, motion blur, sensor noise and fixed
  exposure: main 38 s (25 s perimeter loop at 1.5 m, 11 s knee-height loop) and test 20 s
  (figure eight at 1.2 m). Ground truth in `data/raw/blender01_gt/`: OpenCV cameras and K,
  depth as 16-bit mm PNG, instance masks with `objects.json`, world-frame meshes.
- Render times at 16 samples + OIDN: main 1140 frames in 2686 s (2.36 s/frame), test 600
  frames in 1410 s (2.35 s/frame), about 70 min in total with GT conversion and encoding.
  Both mp4s are H.264 1920x1080 at 30 fps.
- Checks: floor pixels back-projected with the written K, poses and depth land at
  y = 0 ± 0.2 mm. `extract_frames.py` selects 152 frames from `blender01.mp4`.
- `scripts/blender_gt_eval.py`: pose ATE after Sim(3) alignment of the COLMAP cameras, and
  per-object mask IoU with one-to-one matching. Pipeline frames map to GT frames through
  `source_time_s`, refined by comparing against the renders. Sanity run on a fake model
  (GT poses under a random Sim(3) plus 1 cm noise) gives ATE 1.7 cm and scale 2.5 as
  expected; GT masks with permuted labels give IoU 1.0. Chamfer for completions is a stub
  until M4.
- Decision D19 (OJ, 30 Oct): Cycles' denoiser (OpenImageDenoise, a learned network) is
  allowed for the Blender renders, as offline data creation, not a runtime model. The
  existing renders stay; `--no-denoise` remains as an option.
- Next: run M1 on `blender01`, download `recon/sparse` from the Modal volume, then run
  `blender_gt_eval.py`; masks once the object stage writes label PNGs.

## 2026-10-30 (lane: coverage)

All runs on `covtest`, a copy of `smoke01` (32 frames, first 8 s of the Blender walk,
gsplat 500 steps, 60k Gaussians). Spend: Modal about $0.09 (uncertainty $0.014, two
render runs $0.011, top-up rerun $0.055, distil about $0.01), Muse Image 11 calls, $0.11.

- C1 on real cameras. The first run found 0 gaps: the median phone height is 0.98 m in
  the metric frame and the room box tops out at 1.43 m, so the fixed 1.6 m eye height left
  no walkable position. The eye height now follows the median camera height, and scoring
  reads `recon_full/` cameras. Result: 25 gaps, 95% of VR-visible non-empty voxels in gaps,
  77% of voxels never observed. That fits an 8 s capture that looks one way.
- C0. `modal_jobs/coverage.py` copies the reconstruct.py image chain byte for byte
  (`tests/test_coverage_modal.py`), so the cached gsplat layer is reused (Modal log of the first run:
  "Built image ... in 5.69s"). First uncertainty run: fit 6.5 s for 3000 iterations. The spherical mean `u_mean`
  stayed at 0.88 to 1.0 (unseen directions keep the prior), which zeroed every coverage
  score. Scoring now averages u over the observed directions (median 0.23).
- C2 full rerun. `scripts/coverage_topup.py` appends top-up frames with the next indices,
  rebuilds `sam_input.mp4`, reruns reconstruct.py, fits the new frame to the old one on
  the shared cameras (Umeyama) and rescores. Live on `covtopup` (frames 0 to 19 as the
  scene, frames 20 to 31 as a 3 s top-up video): MapAnything 38 s, gsplat 2000 steps 34 s,
  alignment scale 1.0006, RMS 2.5 mm, 127 s wall in total. This checks the plumbing only.
  The top-up re-adds frames from the same walk, and the previous model was the 32-frame
  smoke01 reconstruction, so all 32 cameras were shared and the RMS compares the same
  frames solved twice. After review, `align_world` also rejects cameras on a line and
  orientations that disagree by more than 5 degrees, and the script drops repairs and
  generated splats of the old frame and reruns `lift_ids.py` (not run live).
- Viewer: the view to film first is a yellow phone frustum with a floor ring
  (`?gapViews=0` shows only it). Screenshot of `?scene=covtest&gaps=1` checked.
- C3. Gap views now come from gsplat on Modal (`coverage_repair.py --gpu`). Two findings
  on Muse Image edit: the Meta API rejects the SDK's `image[]` keys (400), so the request
  sends `image[0]`, `image[1]`, ... itself, and with three reference frames Muse Image
  redrew the room from a reference's viewpoint. With one reference it kept the camera.
  Every repair gets an alignment check (L1 outside the gap mask, at most 0.08): 0.007
  accepted, 0.150 and 0.213 rejected. `modal_jobs/coverage.py --job distil` fits new
  Gaussians only, scanned ones frozen, against the accepted repairs and all 32 training
  frames: 246 seeds, 138 kept, masked repair L1 0.049 to 0.011, training L1 0.0254 to
  0.0256, 15 s. Output `coverage/generated.ply` (every splat `generated = 1`), recorded as
  `coverage.generated_ply` in scene.json and the viewer. Key T, `?generated=0` and
  `window.mrCoverage.toggleGenerated()` hide it.
- Not verified: repair quality on a full capture (one accepted view on covtest), the top-up
  loop with a real second video, the generated mesh with the lod lane's per-object trees.
- Next: run C1 to C3 on `blender01` once M1 has produced it, with
  `modal_jobs/coverage.py` first so the uncertainty term is in.

## 2026-10-30 (lane: blender, second synthetic room `blender02`)

- `scripts/blender_room.py --scene blender02` builds an L-shaped bedroom-study (D13): 4.5 x 5
  x 2.6 m overall with the north-east 1.5 x 2 m corner cut out, so a 4.5 x 3 m study in the
  south opens along y = 0.5 into a 3 x 2 m bed alcove in the north-west. 8 movable pieces
  (day bed, nightstand, built floor lamp, desk, desk chair, bookshelf, armchair, potted
  plant) and 4 items (books and succulent on the desk, vase and horse figurine on the
  nightstand). Lamp-lit with two ceiling panels, no mirrors or glass. The default scene
  stays `blender01`. Its camera matrices for all 1740 frames are bit-identical to the
  previous version of the script, and its objects, walks and seeds are unchanged.
- Walks for `blender02`: main 40 s (25 s chest-height loop at 1.5 m along the L, 9.6 m,
  looking across at the far side of the loop, 2 s crouch, 13 s knee-height loop at 0.6 m in
  the study) and test 20 s (a loop at 1.2 m through the study and the alcove, looking at a
  point that circles the furniture ring clockwise). Paths are corner-cut (Chaikin) polygons.
  Every frame keeps at least 0.35 m from walls and furniture footprints (unit test, checked
  against the built `objects.json` bboxes when they exist, a test skipped otherwise).
  Turn rate stays under 40 deg/s (36 main, 39.8 test). No two furniture bboxes overlap.
- Outputs follow the scene id: `data/raw/blender02/`, `blender02_gt/`, `blender02.mp4`,
  `blender02_test.mp4`. `objects.json` gains `room_outline_xz` (world x, z) for the L.
  Floor and ceiling are one quad per band of the outline, with UVs from a shared origin so
  the laminate has no seam.
- `scripts/blender_capture_guide.py --scene blender02` draws the L outline from
  `room_outline_xz` (falls back to the `room_size_m` box for `blender01`) and has its own
  instructions for the new walks.
- 9 new CC0 Poly Haven models in `blender_assets.py` and `docs/ASSETS.md`.
- Check on the Mac (Blender 5.2.1, Metal): 6 frames of the main walk (every 200th) at 8
  samples took 109 s on the first run and 27 s on a rerun (Blender console output, no log
  committed). GT conversion of those frames gives masks
  with the expected indices and depth of 1.2 to 4.8 m. The full walks are not rendered yet.
- Next: render both walks in full (main session), then the capture guide videos.
## 2026-10-30 (lane: bake)

- B1 alpha mask (`src/movable_reality/bake/mask.py`, decision D11). On a 5 cm grid, a cell
  is surface when a splat covered any of its texels or, for floors, when it lies under the
  footprint of an object standing on the plane (bbox bottom within 15 cm, structure labels
  floor, wall and ceiling excluded). A 15 cm morphological closing bridges cracks, and
  uncovered regions that do not reach the quad's border (under furniture) stay opaque with
  the push-pull fill.
- A border region is cut only when it reaches more than 15 cm from the surface and walls
  run along at least half of its edge against the surface. Walls are background splats
  0.2 to 1.5 m off the plane, five or more per cell, within 10 cm of the edge. An unseen
  corner of a rectangular room therefore stays filled, and so does the strip between a
  sofa and the wall.
- A masked plane's texture is WebP with alpha (lossy colour at the JPEG quality, lossless
  alpha) through the required glTF extension `EXT_texture_webp`, and its material uses
  `alphaMode: MASK`, `alphaCutoff: 0.5`. Synthetic L floor GLB sizes are 39 KB as WebP,
  109 KB as JPEG without a mask and 585 KB as RGBA PNG. `makeOpaque` in
  `web/src/planes/planes.ts` keeps the alpha test GLTFLoader sets, and `planeKey` includes
  `transparent_fraction`, so a re-bake that only changes the mask reloads the plane.
  Planes without transparent texels stay JPEG and `OPAQUE` as before.
- New field `BakedPlane.transparent_fraction` (scene.py, scene.ts, tests). `filled_fraction`
  now counts only filled texels inside the outline. `modal_jobs/bake.py` takes
  `--no-mask` (not run).
- `make_room(shape="l")` and `bake_planes.py --synthetic --shape l`. The 1.2 x 1.3 m
  corner at +x, +z is outside the room and the walls follow the L. The unseen floor under
  the sofa runs to the far wall, so only the sofa's footprint keeps it opaque. At 5 mm
  texels the floor is 7.2 % transparent and the ceiling 7.1 %. The corner is 7.8 % of the
  4 x 5 m rectangle, and the mask extends up to one cell past the splats, behind the
  walls. The rectangular `bakedemo` has 0 % transparent texels.
- Browser check (dev server on port 5175, local copies of the bakes with the floor turned
  to face the default camera, not committed, screenshots only in the session). With the
  mask the L-shaped floor ends at the walls, the corner shows the background and the
  floor under the sofa is filled up to the wall. With `--no-mask` the floor runs past the
  walls into the corner. The default view inside `bakedemo_l` renders as before.
- Not done: no real room (room01 does not exist yet), no headset run. The cost of alpha
  test on a depth-writing full-screen quad on Adreno GPUs is unmeasured. If it shows,
  cutting the quad geometry along the mask is the alternative.
- Next: B2 (gsplat bake on Modal), B3 (real room seams).
## 2026-10-30 (lane: lod)

- L1: `SceneObject.completed_lod` in `scene.py` and `scene.ts`. `build_lod.py` builds
  `lod/obj_<id>_completed.rad` for every object with a `completed_ply` on disk and records
  it (missing completed PLYs are skipped, reruns are idempotent).
- The viewer swaps an object to its completed PLY once its transform differs from identity
  (`shownPly` in `web/src/scene.ts`, the former M4 TODO) and back on undo. Both variants
  stay in the scene graph and the unused one is hidden, so a swap back is instant and the
  swap leaks no worker tree. A completed PLY that fails to load falls back to the scan.
  `lodChoice` loads `completed_lod` for the completed PLY. Unit tests with a stub
  `SplatMesh` cover the load race, undo during a load and failed loads.
- L2 (D7): `installLod` sets `spark.lodCleanupTimeoutMs = Infinity`.
- Checked in the browser on `demo_l1` (a local copy of `demo` with a red-tinted chair as
  the fake completion, not committed). A move turns the chair red, the undo turns it
  blue and a rotate turns it red again. `mrLod.stats()` showed 5 trees and 4 traversed
  meshes throughout (manual check, no log committed).
- Known gap: With cleanup off, the old background tree stays in the worker after a live
  bake swaps the background PLY (see docs/LOD.md, open points).
- Next: L3 (priority tuning on the headset). Real completions arrive with M4.

## 2026-10-30 (lane: ui)

- New lane (D18). Everything lives in `web/src/ui/`, plus one `installUi(ctx)` call in
  `main.ts` after `installCoverage`.
- U1 desktop editing (`editor.ts`): a click picks the nearest object box with lane lod's
  `pickNearest`, draws a yellow box outline on the object's pivot and a DOM label. Dragging
  moves the object across a horizontal plane through the grab point, so the height stays
  and the object follows the cursor. Release sends one `place` edit. Q/E rotate by 15
  degrees, Delete or Backspace hides, R restores the newest hidden object, Ctrl/Cmd+Z undoes,
  Escape clears the selection. The `pointerdown` and `keydown` handlers run in the capture
  phase on `window`, so a press on an object never reaches SparkControls (no orbit), and
  Q/E move the camera again once nothing is selected. Drags on empty space orbit as before.
  Boxes that contain the camera are not pickable, so a large box around the viewer cannot
  block orbiting.
- Edit path (`index.ts`): edits go to `POST /api/scenes/{id}/edit` when `/api/health`
  answers, one at a time in order. Without a server, `edits.ts` mirrors `Scene.apply_edit`
  and `Scene.undo` (append-only history, undo entries with `index` and `of`, yaw modulo
  as in Python) and the bar shows "offline: edits are local". `?server=0` forces local
  edits. A manifest with a shorter history than the one on screen is dropped, so a late
  poll or push never undoes a newer reply. Edit requests time out after 5 s and then fall
  back to local edits. The WebSocket applies `scene` pushes, shows `agent` progress and
  answers `capture_request` with `window.mrCapture()`. With a requested camera it moves to
  that eye and target for three frames (Spark's sort catches up) before the capture. In
  XR it answers null at once.
- U2 command bar (`commandBar.ts`, `recorder.ts`, `audio.ts`): Enter posts the text to
  `/command`. Holding Space records through an AudioWorklet, resamples to 16 kHz with a box
  filter, encodes a 16-bit mono WAV and posts it to `/voice`. Keys typed in the bar no
  longer reach WASD or other shortcuts.
- U4 onboarding panel (`onboarding.ts`): controls, four sample commands that fill the bar,
  a gap overlay button (dispatches key G for the coverage lane) and a placeholder
  "show generated" toggle. `?onboarding=0` hides it, and closing it is stored in localStorage.
- Tests: 23 new vitest cases (edit mirror including undo chains, resampling, WAV header,
  onboarding visibility). `npm test` 105 passed, `npm run build` passes, `uv run pytest`
  148 passed, ruff clean.
- Manual browser check (no committed log, figures read from the page console) on a copy of `demo` (`data/scenes/uitest`), dev server on port 5174.
  Offline: select, drag (place to x 0.215, z -2.006), Q Q E, Delete, R, then seven undos
  restored the original transform and logged 12 history entries. An eighth undo reported
  "nothing to undo". Online against the core lane's work-in-progress server: the drag and
  a Q rotation were written to `scene.json`, an undo sent with curl reached the page by
  WebSocket within 300 ms, and "remove the chair" ran through the agent with progress lines
  in the bar. The verifier answered "no" each time, so the agent kept retrying. The
  viewer's capture was a non-black 2048 x 1536 JPEG. That run used the one-frame capture
  that the review replaced with the three-frame one, which was not re-run against the
  verifier.
  Not tested: hold-to-talk with a real microphone, any XR session.
- Next: a controller button for hold-to-talk in VR, "show generated" once M4 flags splats,
  the video link in the onboarding panel, U3 hosting.
## 2026-10-30 (Fri, evening, Mac, lane core: M1 to M3, M5 server side)

- Scope change from OJ: everything runs on synthetic data first and is tested in the desktop
  browser. Headset tests come later.
- Modal billing works. A T4 smoke job ran, then the first full M1 run on `smoke01` (32 frames from
  the first 8 s of the re-rendered `blender01` main walk): MapAnything 50 s, gsplat 500 steps
  23 s on the L40S. Fixes on the way: MapAnything's COLMAP export needs `open3d` (new image
  layer, the cached gsplat layer stays), and the recon download paths.
- Full-resolution training: `recon.py` inverts MapAnything's preprocessing (scale by
  max(w/W, h/H), floor, centre crop, checked against `mapanything/utils/cropping.py` at the
  pinned SHA), and the job writes `recon_full/` (frames as images, rescaled cameras).
- M2: `inventory.py` (Muse Wonder on the uploaded clip through the Responses API with an
  `input_file`) and `lift_ids.py` + `lift.py` (SAM 3.1 per noun, occlusion-aware voting,
  floor alignment with yaw, floater filter, support rule D12). On `smoke01` the inventory named
  8 objects. SAM 3.1 took 17 to 175 s per noun on the 32-frame clip (6.5 min for 11 nouns, 3 in
  parallel). Lift produced 4 objects. The 500-step splat is too rough to judge quality.
- M3: `Scene.place/undo/apply_edit` (append-only undo), agent tools (`fits`, `measure`, ...),
  Muse Wonder tool loop with the user's view direction, Muse Spark Open verifier on before and
  after screenshots with one retry, FastAPI server with WebSocket (`docs/SERVER_API.md`). Every
  chat, verifier and ASR call is cached under `data/cache/`. Live: "remove the chair and move
  the lamp half a metre to the left" on a copy of `demo` gave the right three tool calls.
- Verifier fix: Muse Spark Open is a reasoning model and used the whole 200-token limit on
  thinking, so every verdict was empty and counted as NO. Now 2000 tokens, and an empty answer
  means unverified (no retry).
- M5 server side: `voice.py` (Muse Voice Transcribe). A 16 kHz WAV of "Turn the table ninety
  degrees and bring back the chair" (macOS `say`, test only) was transcribed and executed.
- U3 server side: hosted mode with per-visitor scene copies and command caps.
- Review fixes: yaw sign in `floor_alignment` (a room rotated by a came out rotated by 2a),
  finer yaw search, NaN edits rejected, undo restores the previous visibility, depth buffer
  padding, box-local masks in `lift_ids` (memory), saved state rebroadcast after an agent error.
- Next: M1 + M2 on the full `blender01` walk, pose ATE and mask IoU against ground truth.

## 2026-10-30 (lane: core, M4)

- `modal_jobs/sam3d.py`: SAM 3D Objects (commit f91db41, gated `facebook/sam-3d-objects`,
  SAM License) as a deployed Modal class on one L40S. Its default pipeline estimates the
  pointmap with MoGe, a Microsoft model, which rule 2 forbids. The job sets
  `depth_model: null` and passes a MapAnything pointmap (`facebook/map-anything-apache`)
  through the pipeline's own `pointmap=` argument. The MoGe package stays installed for its
  geometry helpers only (no weights are loaded). DINOv2 encoders (Meta) load through
  torch.hub. HF_TOKEN reaches Modal as a secret built from `.env` at deploy time.
- Image: CUDA 12.1, Python 3.11, torch 2.5.1, the upstream requirements minus training and
  notebook packages, kaolin wheel, gsplat without its CUDA build, PyTorch3D built CPU-only
  (only transforms and structures are used, 124 s on the 1-CPU builder). First build and
  deploy took 511 s. Weights (13 GB) download once to the `movable-reality-cache` volume
  in about 60 s.
- Sources of the figures below. Timings, memory and decoder sizes come from the Modal container
  logs and the result JSONs under `data/cache/sam3d/` (gitignored), build times from the
  `modal deploy` output, the cost from `modal billing report --for today`, the splat count
  from the viewer's console, and "upright" from a viewer screenshot that is not committed.
- Measured on the L40S. Cold start 44 to 51 s to load both models (16.1 GiB), MapAnything
  pointmap 0.9 s, SAM 3D generation 11.8 to 12.4 s, peak 20.2 GiB. The dense decoder gives
  985,440 splats (67 MB) for a stool. The 4-splats-per-voxel decoder gives 123,180 splats
  (8.4 MB) and is now the default (`dense=False`).
- `src/movable_reality/insert.py`: Muse Spark gives noun and typical size (0.30 x
  0.40 x 0.30 m for the stool), Muse Image draws one object on a plain background (1600 x 1600
  returned), SAM 3.1 masks it, SAM 3D Objects lifts it. The object is turned upright
  (camera up, snapped to the nearest canonical axis within 40 degrees, and the stool's
  canonical +Z was 25.3 degrees from camera up), scaled, stood on y = 0 at (x, z), written in the
  recon frame as `objects/gen_<n>.ply` with `generated` = 1, and logged as op "insert".
  The agent tool now returns insert failures to the model as an error.
- Live test on `m4test`, a copy of `demo`. "small wooden stool" became `gen#1`, 0.34 x 0.36
  x 0.35 m, 86 s end to end with a cold container (82 s of it Modal). The viewer loads the
  PLY (Spark logs 123180 splats) and shows an upright stool on the floor between the chair
  and the table.
- `scripts/complete_objects.py` with `src/movable_reality/completion.py`: best frame per
  object (largest mask, half weight when it touches the border), SAM 3D on frame + mask,
  first guess from the pose and the frame's camera, scale from the boxes, then trimmed
  similarity ICP from 10 starts (pose guess, plus box centre, min face or max face in x and
  z), best RMS wins. Not run on a real scene yet, because no scene has `ids/masks` so far. Unit tests
  cover it on a synthetic partial scan (error under 3 cm).
- `src/movable_reality/splat_ops.py`: PLY layout, quaternion and similarity transforms of
  splats, SAM 3D pose conversion, floor placement, Umeyama and ICP. 18 CPU tests in
  `tests/test_core_m4.py`.
- Modal spend for M4 was $0.60 (billing report for 30 Oct, all build and GPU time of
  `movable-reality-sam3d`), including about ten short crash loops before two startup fixes
  (`hf_transfer` env without the package, zero-depth points in the pointmap).
- Review fixes (reviewer agent). The scan is filtered by opacity and a MAD floater test
  before registration, the box scale falls back to the metric pose guess when the two differ
  by more than 25 %, ICP ranks starts by trimmed RMS and keeps its best step (a synthetic
  scan with 7 % floaters, 3 mm noise and a 15 degree yaw error now registers within 6 cm),
  generated objects use SAM 3D's canonical +Z as up, the agent runs an insert in a worker
  thread so the server stays live, hosted mode has no insert (visitors would overwrite each
  other's `gen_<n>.ply`), and the deploy stops when HF_TOKEN is missing. The insert no
  longer touches `coverage.generated_splats`, which stays with the coverage lane.
- Next: run `complete_objects.py` once `lift_ids.py` has written masks for a real scene,
  then the viewer swap to `completed_ply` (lod lane, L1).

## 2026-10-31 (lane: blender, irregular rooms `blender03` and `blender04`)

- `scripts/blender_room.py` builds polygon rooms from zones. A zone is a convex floor
  polygon with its own ceiling plane z = a x + b y + c. `shell_faces` turns the zones into
  floor, ceiling and wall polygons. A zone edge that no other zone covers becomes a wall
  from the floor to that zone's ceiling. A covered edge where the neighbour's ceiling is
  lower becomes a vertical step face on the higher side (dormer cheeks, the header over the
  bay). Free-standing columns get four faces and their own GT index, 104. `blender01` and
  `blender02` keep their old builders, so their geometry and cameras are unchanged.
- `blender03`: living room, 5.8 x 5.0 m bounds, with an angled south-east wall, a three-sided
  bay window on the east wall (2.4 m ceiling, the rest 2.6 m) and a 0.36 m column. The
  outline has 9 vertices and is non-convex at the bay. There are 8 pieces of furniture and
  4 items, all reused CC0 assets. Main walk 40 s. Its chest loop circles the column and
  runs between the column and the bookshelf, along the bay mouth and past the angled wall.
  The knee loop stays south of the column.
- `blender04`: attic bedroom-study with 1.0 m knee walls at x = ±2.4, 45 degree slopes, a
  flat 2.6 m ceiling between x = -0.8 and 0.8, and a dormer recess (1.4 m wide, 0.6 m deep
  past the knee wall, 2.3 m flat ceiling). The bed, nightstand, armchair and plant stand
  under the slopes. Every footprint sits at least 5 cm below the ceiling (test). Main walk
  43 s. The chest loop enters the dormer mouth. The 16 s knee loop reaches under both
  slopes, to a ceiling of 1.68 m in the east and 1.79 m in the west.
- `objects.json` of the polygon scenes adds `ceiling_zones` (world outline and ceiling
  plane) and `columns_xz`. The capture guide draws the polygon, the zone edges and the
  columns, and has instructions for both new scenes. `--layout` writes
  `data/raw/<scene>_layout.png` (outline, furniture labels, ceiling heights, both walks)
  from `objects.json` or, before a build, from the scene definition. Made for `blender02`,
  `blender03` and `blender04`.
- `tests/test_blender_polygon.py` has 31 tests and 2 skips without a build. They check that
  the zones tile the outline, that floor and ceiling areas match it, that the walls run the
  perimeter once with air on their normal side, the dormer cheeks and bay header, that the
  furniture fits under the ceiling, and that every walk frame keeps 0.35 m from walls,
  furniture and the column and 0.3 m from the ceiling. They also check speed, turn rate and
  coverage of the column, bay, dormer and slopes.
- Previews at 8 samples (Blender 5.2.1, Metal, M5) are in `data/raw/preview/<scene>/`,
  every 150th frame of both walks. The first `blender03` preview had the column in the middle
  of both loops, so it filled many frames. The column moved north, off-centre, and a fourth
  ceiling light went in behind it, so its north face is no longer a dark slab. The chest and
  test walks still pass 0.44 to 0.5 m from the column, where it fills a third to a half of a
  frame. That is kept on purpose as an occlusion case.
- Full renders (16 samples) of both scenes, then the guide videos and layouts, are queued
  by `data/raw/render_loop_blender03_04.sh`. It waits for the running `blender02` render
  to finish. Not yet verified.
- Next: check the full renders and guide videos, then run M1 and `blender_gt_eval.py` on
  `blender03` and `blender04`.
## 2026-10-31 (lane: devices)

- Decision D20 (OJ, 31 Oct): the viewer must work on any device (laptop or desktop browser,
  phone or tablet browser, standalone headset), each at the best quality it can hold.
- `web/src/devices/profile.ts`: three profiles. A running WebXR session selects `xr`.
  Otherwise `?device=desktop|mobile|xr` wins, then a coarse pointer on a screen whose short
  side is at most 1024 CSS px, or a mobile user agent, selects `mobile`, and anything else
  is `desktop`. Per profile it sets the pixel ratio cap, the budget controller's splat range
  and start level, the default frame target, the range a measured display rate may set the
  target to, and Spark's sort options (Spark's defaults for now).
- Settings. Desktop has the full device pixel ratio and 400k to 6M splats, starts at the top
  level (capped by the scene's own size) and targets the display rate up to 60 Hz. Mobile
  has pixel ratio 1.5 at most and 100k to 1.5M splats, starts near 600k and targets 60 Hz.
  It drops to 30 Hz after 3 s at the lowest level with over half the frames missed, and stays
  there until the profile changes (desktop never drops). XR uses `DEFAULT_BUDGET`, 13.9 ms
  and pixel ratio 2, as before. One XR difference remains. The controller entering XR is a
  new one, so it keeps the splat count on screen (the closest XR level) but not the failure
  memory gathered in 2D. A headset browser outside XR runs the profile its user agent and
  pointer give (the Steam Frame's "Linux aarch64" Chromium detects as desktop).
- The renderer lane's `perf/index.ts` reads the profile through `ctx.device` and starts a
  new budget controller when the profile changes (XR enter or exit). Desktop and headless
  logs now target at most 60 Hz, so their missed-frame shares are not comparable with
  earlier desktop logs (`docs/DEVICES.md`, section 6). `context.ts` gained
  `device` and `onDeviceChange`. `?device=` keeps naming benchmark logs for any other value.
- Touch (`web/src/devices/touch.ts`, `gestures.ts`): one finger looks around, two fingers
  pan, a pinch moves forward or back. It replaces SparkControls' touch handling, which
  needs both fingers to land within a few milliseconds and rolls the camera on a pinch. The
  ui lane's Editor now selects on a tap and only drags the selected object by touch, so a
  look that starts over an object turns the view. The command bar gets a hold-to-talk button
  and a full-width portrait layout under the mobile profile, and onboarding lists the touch
  gestures.
- Reviewer fixes. A finger resting during a pan or pinch no longer counts as a tap (the
  Editor matches the tap to its pointer and drops it when a second finger lands), the 30 Hz
  floor is mobile only and costs nothing on other profiles, logs update `device_profile` on
  a profile change, and the XR HUD shows the framebuffer scale instead of the canvas ratio.
- The HUD's first line shows the profile and pixel ratio, for example `[desktop dpr 2.00]`.
  `window.mrDevice` exposes the profile, its settings, the source of the choice and touch
  gesture counts.
- Verification. 27 new vitest tests (profile selection, settings, the 30 Hz floor, desktop
  budget on the simulated display, gesture maths). In the desktop app's Chromium on
  `demo`, the desktop pane selected `desktop` (dpr 2, controller at level 10 = 1.55M, the
  scene's ceiling) and mouse select and drag still produced a `place` edit. The mobile
  preset (375 x 812, Android user agent) selected `mobile` (dpr 1.5). Synthetic touch
  events showed tap to select, tap on empty space to clear, a one-finger look over an
  unselected object (10 look steps, no edit), a drag of the selected table (one `place`
  edit), a second finger ignored during that drag, and two-finger pan and pinch. Frame
  rates in the emulated pane mean nothing. The browser checks were manual console scripts
  (synthetic PointerEvents, with setPointerCapture stubbed because synthetic pointers cannot
  be captured), not committed tests. Not tested: a real phone or tablet, real
  multi-touch, the microphone on a phone, an XR session switching profiles on a headset.
- Next: open `?scene=demo` on a real phone over the LAN (`MR_HTTPS=1`), check gestures
  and frame rate, and check on the Steam Frame that entering XR logs the `xr` profile.

## 2026-10-31 (night), lane core: blender01 end to end, metric scale, filming guides

- Filming guides from blender01. `data/raw/blender01_guide_main.mp4` (38 s, chest loop,
  crouch, knee loop) and `data/raw/blender01_guide_test.mp4` (20 s figure eight) show the
  phone view, the floor plan with the path and view cone, and gauges for phone height, tilt,
  walking speed and turn rate. Sent to OJ.
- MapAnything runner `src/movable_reality/ma_colmap.py` replaces `demo_colmap.py`. One shared
  camera for all frames, either given (`intrinsics.json` in the scene folder, written by
  `extract_frames.py --focal35 <mm>`) or self-calibrated from the median of a first pass.
  Pose results on blender01 against GT (152 frames, 18.9 m path).

  | Run | ATE RMSE | median rot. error | Sim3 scale |
  | --- | --- | --- | --- |
  | per-view intrinsics (demo_colmap, earlier run, eval file since overwritten) | 0.43 m | 3.3 deg | 1.97 |
  | self-calibrated shared camera (fx 1024 against GT 1387) | 0.45 m | 4.3 deg | 2.11 |
  | GT intrinsics | 0.29 m | 2.7 deg | 1.83 |

  Known intrinsics help. The scale stays about 1.8x too small, and smoke01 (real phone) shows
  the same direction (median phone height 0.98 m).
- Metric scale in `lift_ids.py`. `--scale <f>` or `--ceiling-height <m>` folds a uniform
  scale into `world_from_recon`. Coverage splits it, bake and insert invert it and the viewer
  puts it on a group. Completion needed a fix (SAM 3D's metric pose is now divided by the
  world scale). The
  ceiling fallback looks for a spike in the height histogram above the cameras and refuses
  a plain tail. blender01 has no reconstructed ceiling (the walk tilts 28 deg down, SAM found
  no "ceiling"), so the fallback refuses there. blender01 now uses `--scale 1.83`, taken
  from the GT Sim3 fit. That is GT-assisted and only valid for the synthetic test.
- `--reuse-votes` reruns floor alignment, scale, floaters and the support rule from
  `ids/assigned.npy` in about 20 s. Decoded SAM masks are now kept bit-packed next to the
  JSON cache, because meta_sam_parser's pure-Python decode took about 10 min per run.
  SAM requests retry three times, the client timeout is 30 min (the floor noun streamed for
  731 s), and floor, wall or ceiling may fail without stopping the run. The wall noun
  failed with "Connection error" 6 times out of 6 on blender01.
- Lift on blender01. 2D mask IoU against GT is 0.63 mean over objects (vase 0.95, floor
  lamp 0.93, coffee table 0.91, books 0.79, armchair 0.50, plant 0.47). The 3D boxes are
  still too large after scaling (coffee table 1.85 x 1.08 x 1.12 m against 1.20 x 0.39 x
  0.60 m). Trimming outlying splats changed little, so the splats themselves are spread,
  which fits the 0.29 m pose error. The armchair was labelled `sofa#1`.
- build_lod (10 trees), bake_planes (floor only, 21 deg tilt from recon up, 17,869 splats
  removed) and score_coverage (172 gaps) ran on blender01.
- Browser test at `?scene=blender01`. "move the coffee table half a metre towards the sofa"
  went through measure, fits and move_object (0.188 m). Muse Spark Open answered NO
  ("moved only 0.188 m not the requested 0.5 m"), the agent undid the move, moved 0.5 m, and
  the second check passed. The gap overlay draws 172 gaps over most of the room and is
  unreadable at this density (coverage lane).
- Merged PRs #14 (coverage C0 to C3), #15 (device profiles desktop, mobile, xr) and #16
  (irregular rooms blender03 and blender04). `docs/MARKETING.md` holds the marketing notes
  and the self-correcting pipeline explanation.
- Next: pose quality on synthetic renders (more frames, or keyframe spacing), a metric
  reference that works without GT (camera height prior or a measured object), gap
  clustering for the overlay, then blender02 to 04 through the same pipeline.

## 2026-10-31 (early morning), lane core: blender02 end to end

- blender02 (L-shaped bedroom-study, 160 frames, GT intrinsics through `--focal35 26` before
  the 4:3-diagonal fix, which matches Blender's convention). MapAnything 123 s, gsplat 143 s
  on the L40S. Pose ATE 0.50 m after Sim3, median rotation error 3.2 deg, Sim3 scale 2.07.
- MapAnything's COLMAP export at the pinned SHA uses `pts3d` and `camera_poses`, which
  `infer` documents as already metric. The 2x under-scale is the model on these renders, not
  the export.
- SAM on blender02. 14 nouns are cached, floor succeeded on its second attempt after 1200 s,
  and wall failed on every attempt with "Connection error". `lift_ids.py --skip wall` now
  leaves nouns out. Lift with `--scale 2.07` (GT-assisted) gave 10 objects, then LOD (11
  trees), bake (floor plane, 16,016 splats removed) and coverage (206 gaps). Object sizes are
  only roughly right (bookshelf 2.32 x 2.70 x 2.66 m against 0.58 x 2.06 x 1.37 m), in line
  with the pose error.
- Browser check at `?scene=blender02`. The room loads under the `desktop` profile (dpr 2,
  219k splats drawn). The ceiling is black because the walk never looked up.
- Filming guides `data/raw/blender02_guide_main.mp4` (40 s) and `_test.mp4` (20 s) were made
  and sent to OJ.
- Next: section 3.00 of docs/HANDOFF.md.

## 2026-10-31 (day), lanes core, renderer, devices, blender: photoreal reconstruction

- OJ found blender01 not presentable. Three causes were measured.
  - The first blender01 splat trained for only 7,000 steps (the quick-look preset) and had
    180k Gaussians. Held-out PSNR was 25.5 dB.
  - MapAnything's poses were 0.29 m off after a Sim3 fit.
  - MapAnything's COLMAP export writes its own predicted intrinsics even when known ones are
    given: fx 1050 px against the true 1387 px. Every earlier run trained with it.
- New SfM stage `src/movable_reality/sfm.py`. pycolmap 4.2.1 runs SIFT, sequential matching
  and incremental mapping with fixed intrinsics in its own CPU image on Modal. pycolmap 3.13,
  which gsplat's requirements install, registered none of the 152 frames. One Sim3 fits the
  model onto MapAnything's metric frame and writes `recon/` in MapAnything's convention, so
  later stages read it unchanged. Frames it cannot register are dropped (50% minimum). On
  blender01 it registered all 152 frames with a 0.59 px reprojection error, 2.9 mm ATE
  against GT (3.3 mm once fitted into MapAnything's frame) and a 0.08 deg median rotation
  error. The fit now drops pairs over 3x the median residual and refits once.
- Training presets in `modal_jobs/reconstruct.py`. `quality` (the default) runs 30k steps
  with absolute-gradient densification (grow_grad2d 0.0003) and antialiasing. `mcmc` relocated
  about 99% of 3M Gaussians every 100 steps on blender01 and ran out of memory at step 22k.
  `fast` is the old 7k run.
- Novel-view check `scripts/blender_view_eval.py` plus `--eval-walk test`. It renders the
  test walk, which training never saw, from GT cameras moved into the reconstruction frame,
  and scores PSNR, SSIM and LPIPS against the Blender renders.

| scene | run | Gaussians | held-out PSNR | test-walk PSNR | test-walk LPIPS |
| --- | --- | --- | --- | --- | --- |
| blender01 | 7k, MapAnything poses | 180k | 25.5 | not run | not run |
| blender01 | 30k, SfM, grow 0.0008 | 187k | 40.8 | 32.6 | 0.194 |
| blender01 | 30k, SfM, grow 0.0003 | 763k | 41.4 | 33.1 | 0.179 |
| blender01 | + upward loop (224 frames, 178 registered) | 848k | 40.9 | 34.3 | 0.179 |
| blender02 | 30k, SfM, grow 0.0003 | 1.25M | 40.5 | 31.1 | 0.140 |

The raw metrics for the last two rows are in `benchmarks/recon/<scene>/` (gsplat val stats,
test-walk metrics, `sfm.json`, GT pose ATE). Every row predates the lighting change below.

- Ceiling. The capture walks pitch 22 deg down, so the ceiling and the upper walls trained
  black. A turn on the spot at 20 to 65 deg up was tried first. Without parallax, SfM placed
  none of its frames, and MapAnything's poses for them were up to 178 deg off. blender01's
  main walk (`scripts/blender_room.py`) now ends with a second 16 s perimeter loop facing
  inwards, 15 deg up (56 s in all). SfM registers 26 of its 72 frames. The rest look at
  plain paint and are dropped.
- Viewer. The desktop profile now targets 33.3 ms, because OJ prefers every splat at 30 fps.
  The old 60 Hz target had cut blender02 to 458k of 920k splats. A detail panel adds an
  auto checkbox and a log-scale splat-count slider from 50k to every splat, so OJ can set
  the detail by hand.
- New blender01 downstream. Lift `--skip wall --scale 1.657` (GT-assisted) gave 11 objects.
  The ceiling plane fit came out 25 deg off level over a 2.83 x 0.91 m strip, so the bake
  uses `--labels floor` only. In the browser the room draws all 594k splats at about 60 fps
  under the desktop profile (read off the HUD by eye, no frame-time log), with the ceiling and upper walls present. Smudges remain where the
  upward loop was dropped (plain wall).
- Lighting (OJ, 31 Oct, "they leave a halo on the wall"). `build_room` in
  `scripts/blender_room.py` now lights every room with one ceiling-sized area light (420 W,
  warm white) instead of the 1 m ceiling panels, and the floor lamp has no bulb and no
  glowing shade. This changes `build_scene` for all four rooms. blender01 is re-rendering.
  The scores above are from the old lighting.
- Next: the upward loop in blender02 to blender04 and in the real-room filming guide, the
  splats left behind when an object moves, and object box sizes (armchair#0 is 0.07 m wide).

## 2026-10-31 (evening), lanes core, bake, blender: per-object training, ambient lighting

- PR 18 (SfM, quality preset, view evaluation, detail panel, upward loop) merged into main
  after the reviewer's 11 findings were fixed: trimmed Sim(3) refit, separate MapAnything
  and SfM stamps, a clear error without a checkpoint, case-insensitive `--skip`, metrics
  under `benchmarks/recon/`.
- Moving objects (OJ, "it leaves so many artifacts behind"). New stage
  `src/movable_reality/objtrain.py`, run with `modal_jobs/reconstruct.py --objects` after
  the lift (L40S, 6.5 min wall clock with the image start, about $0.25). It re-cuts the joint splat with the SAM masks and
  trains every part alone.
  - Lost SAM ids join the scene object of the same noun (blender01: `coffee table#1` was
    the same table, so the lift had left the table body in the background).
  - A background splat that lands on object k in half the views where SAM found k moves
    to k when it lies within 15 cm of k's splats (set in metres via world_from_recon).
  - Each object trains on black against frame x mask with an alpha term, so splats outside
    the object fade and are pruned (floor lamp 24.1k to 10.0k on the new lighting).
  - The background trains without object pixels, then colour only (SH degree 0) against the
    frame with the objects inpainted by OpenCV Telea. Geometry stays frozen in that pass:
    with geometry on, faint splats became floor-coloured floaters.
  - Background splats in front of the trained objects' rendered depth, and leftovers inside
    an object's box, are removed. Trained objects whose boxes overlap 60% are one thing
    under two nouns (armchair as `sofa#1` and `armchair#0`), and the smaller is hidden.
- Bake (`bake/pipeline.py`). The floor quad now also removes background splats on or under
  it by position, because a retrained background no longer shares the joint splat's
  centres, but only behind the quad or in its band under an object, so a rug stays (test
  `test_retrained_background_goes_by_position`). Texels under objects are blanked before
  the push-pull fill: the joint splat had blue and yellow under the sofa.
- Tried and left off: depth inpainting to add splats on the never-seen wall behind the
  sofa (`--holes 1`). The back-projected points floated in the room as speckles.
- Lighting (OJ, "remove lights like this, they leave a halo"). One ceiling-sized area light,
  no lamp bulb. blender01 was re-rendered (2280 frames, 1677 s for the test walk) and
  rebuilt: 177 of 224 frames registered, 787k Gaussians, held-out PSNR 41.0 dB (LPIPS
  0.135), test walk 31.2 dB (LPIPS 0.198). The test walk was 34.3 dB on the old lighting.
  Pose ATE median 5.6 mm, but 3 frames of the upward loop (183, 199, 201) are 0.2 to 0.66 m
  off, and the upper walls show dark streaks. Neither the MapAnything fit residual nor
  temporal smoothness separates those frames from good ones. The old-light scene is kept
  as `data/scenes/blender01_oldlight`. Metrics: `benchmarks/recon/blender01/` (new) and
  `benchmarks/recon/blender01_oldlight/`.
- Checked in the browser (blender01, both lightings): the sofa, armchair and coffee table
  move and hide without a ghost, and the floor and wall behind them stay clean
  (`benchmarks/recon/blender01/moved_sofa_table_armchair.jpg`). `books#0` still claims
  20.5k background splats against 7.4k of its own, which looks like over-claiming. Not yet
  fixed: the bowl on the table is no scene object, so it stays where the table was. The
  wall behind the sofa is thin where no view saw it. LOD and coverage were not rebuilt
  for the new objects, so the viewer reads the PLYs.
- Next: drop or re-register the bad upward-loop poses, keep small items that rest on an
  object (the bowl) with it, rebuild LOD, and the new lighting for blender02 to blender04.

## 2026-10-31 (night), lane core: SAM 3D Objects meshes next to the splats

- OJ asked for objects as 3D models instead of splats, or both, kept alongside the scan so
  they can be compared. SAM 3D Objects (Meta, already deployed on Modal) also has a mesh
  decoder. `modal_jobs/sam3d.py` now takes `mesh=True`, adds "mesh" to the pipeline's
  `decode_formats` for that call and returns the GLB that its `to_glb` builds, with vertex
  colours. Texture baking stays off because it renders 100 views through PyTorch3D, which
  is a CPU build in this image.
- `scripts/complete_objects.py --mesh` writes `objects/obj_<id>_mesh.glb` beside
  `obj_<id>_completed.ply` and sets the new optional `SceneObject.mesh` (scene.py and
  scene.ts). The mesh uses the same registration matrix as the completed splats. The
  rotation from the GLB's y-up frame back to the Gaussians' frame defaults to the inverse
  of `to_glb`'s z-up to y-up turn, and `meshes.to_gaussian_frame` logs the nearest-point
  distance of every candidate. The default fitted best on all 10 objects, but only by
  0.0295 against 0.0299 on `books#0`, a near-box, so another candidate must win by 2x to
  replace it. The raw meshes have 85k to 1.02M faces. Quadric decimation
  (fast-simplification, MIT, no learned weights) cuts each one to at most 100k faces,
  2.05 to 2.40 MB per object and 23.66 MB for the ten (`mesh_bytes` in the report).
- Run on blender01. SAM 3D's generate step summed to 92.6 s over the 10 objects, plus 48.5 s
  of model load, on one L40S (under $0.20). ICP RMS after registration was 7 to 47 mm. Per
  object, the mesh's world bounds and the completed splats' 1st to 99th percentile bounds
  differ by 1.2 to 16 cm, median about 4.5 cm. The largest gaps are on `sofa#0` (16 cm)
  and `armchair#0` (11 cm), where the percentile trim cuts more of the splats' extent
  (`completed_bounds_world`, `mesh_bounds_world` in `benchmarks/recon/blender01/completion.json`).
- A completed PLY and its mesh always come from one SAM 3D run. A run without `--mesh`
  reuses the cached mesh run when there is one, and clears `mesh` when there is none.
- Viewer. `?objects=auto|scan|completed|mesh` and key V (or the label under the detail
  panel) switch the style. auto is the old behaviour (scan, SAM 3D splats once moved).
  scan pins the scan, completed pins SAM 3D's splats, and mesh draws the GLBs (unlit,
  opaque, under each object's pivot, so edits move them) and hides that object's splats.
  The scan and the per-object training are unchanged.
- Checked in the browser on blender01 with the 15 edits that were in scene.json. The
  scanned cabinet shows its unseen back as streaks. The SAM 3D splats and mesh show a
  closed cabinet with a plausible back (`objects_scan.jpg`, `objects_completed.jpg`,
  `objects_mesh.jpg` in `benchmarks/recon/blender01/`). Frame times in the hidden browser
  pane are throttled (1 fps), so mesh mode was not timed. It is not tested on a headset.
- Next: textured meshes (texture baking with a GPU PyTorch3D or nvdiffrast build), the
  generated look of SAM 3D colours against the scan's, and timing the mesh mode on the
  Steam Frame.


## 2026-10-31 (late), lane core and blender: mesh-only view, room as a textured shell, blender05

- Request (OJ, 31 Oct). Make the modes fully separate, with a mesh-only mode in which the room is a mesh too and no Gaussian is visible, and test it on a room with windows (closed blinds allowed) and doors.
- Viewer. `?objects=meshonly` (key V cycles to it after `mesh`) hides every `SplatMesh` in the scene each frame, hides the baked planes, draws each object's SAM 3D mesh and loads the room from `scene.room_planes`. Leaving the style shows the splats it hid. `web/src/objects/index.ts`.
- Room shell (`src/movable_reality/roomshell.py`, `scripts/room_shell.py`). The SAM floor-mask pixels of every frame are cast onto the plane y = 0. A 5 cm cell counts as floor when 3 or more frames hit it, and object footprints are added. Closing, hole filling, the largest region and an orthogonally convex fill give a region, whose traced border is simplified (Douglas-Peucker, 8 cm) and rectified to the main wall direction. The height is the densest splat layer above the cameras (`lift.ceiling_from_histogram`), or the 99th percentile of splat heights when that layer has under 100 splats. Walls are vertical quads along the outline, and floor and ceiling are quads with an alpha mask. Every texel is coloured with the median of its 3 best frame samples (cosine over distance), skipping pixels SAM labelled as furniture. Windows, doors, blinds, curtains, pictures and mirrors are not skipped (`ON_WALL`), so they texture the wall. Unseen texels are filled by push-pull. No learned model is involved beyond the SAM masks already computed.
- Rectify fix. A test found that a slightly turned rectangle kept a 0.46 m diagonal at one corner (5 corners). Diagonal edges shorter than twice the 0.4 m step are now dropped, while a 1.4 m angled wall is kept (`tests/test_core_roomshell.py`, 9 tests).
- TSDF room mesh (`src/movable_reality/roommesh.py`, `reconstruct.py --room-mesh`). gsplat colour and depth from every camera, a multi-view depth agreement filter (3 of 12 neighbours within 1%) and open3d TSDF fusion at 1.5 cm. On blender01 it kept 69.7% of pixels and gave 673k faces, but the ceiling and the upper walls stayed open, so the shell is the default and the TSDF mesh is the fallback when `room_planes` is empty.
- blender01. Shell outline 4 corners, walls 4.06 and 5.03 m (ground truth 4.0 x 5.0 m), height 2.60 m from the densest layer. Screenshot `benchmarks/recon/blender01/objects_meshonly.jpg`, reports `benchmarks/recon/blender01/shell.json` and `room_mesh_report.json`.
- blender05 (blender lane, `scripts/blender_room.py`). blender01's room with two 1.2 x 1.4 m windows with closed venetian blinds (north, east) and two 0.85 x 2.05 m doors (west, south). Render at 16 samples took about 1 h 40 min for both walks. Pipeline (reports in `benchmarks/recon/blender05/`): 200 frames, reconstruct (MapAnything 161 s, SfM 345 s, gsplat 530 s), Muse Wonder inventory with 13 nouns (no window or door), lift `--skip wall --ceiling-height 2.6`, per-object training, SAM 3D completions with meshes (10 objects, ICP RMS 0.008 to 0.041 m after registration).
- blender05 scale. The ceiling rule gave x2.715 and the shell came out 4.77 x 5.90 x 3.05 m. Pose evaluation against ground truth gives a Sim(3) scale of 2.390 (ATE median 3.3 cm, RMSE 15.2 cm), so `world_from_recon` was scaled by 2.390 / 2.715 (GT-assisted, as for blender01). The rebuilt shell has walls of 4.01 and 5.03 m. Its height of 2.69 m against 2.60 m comes from the 99th-percentile fallback, because the densest-layer rule found too few ceiling splats.
- blender05 result. In mesh-only mode both windows with their blinds and the west door appear on the walls in the right place (`benchmarks/recon/blender05/meshonly_*.jpg`, with `splats_north_window.jpg` from the same camera). Defects seen: a horizontal seam across the north window where neighbouring frames disagree, dark blotches on the floor around the coffee table, and the vase and the succulent missing, because the lift merged them into the side table and the coffee table and SAM 3D completed only the larger object. SAM found no bowl. The window and door placement was judged by eye only.
- Review (reviewer agent). No geometry errors. Fixed: wall lengths instead of the bounding box in the report, `--force` and a clear exit without floor pixels in `room_shell.py`, the near-depth cut in metres, shell textures freed on reload, tests for the schema fields, `roomFiles` and the world-recon composition. Open: frames with bad poses (ATE max 1.46 m on blender05) still paint the shell, leaving meshonly while generated splats are toggled off shows them again, and `rectify` drops both corners between two parallel walls more than 0.4 m apart.
- Next: dropping badly posed frames from the projection, an occlusion test in the shell projection (walls hidden by other walls in an L-shaped room), seams across views, keeping merged small items as separate meshes, and the ceiling-rule scale error on blender05.

## 2026-11-01 (Sun), lanes core, ui, coverage: one-command pipeline, metric scale without a tape

- `scripts/run_pipeline.py video.mp4 --scene <id> --focal35 <mm>` runs every stage in order
  and resumes. SAM runs next to the GPU reconstruction. A stage is skipped when its outputs
  are newer than its inputs, rerun with `--force` when an input changed, and every later
  stage follows a rerun. It also has `--from`, `--only`, `--to` and `--dry-run`. Logs go to
  `data/scenes/<id>/logs/<stage>.log` and the state to `pipeline.json`. A Modal stage whose
  client loses its connection is retried once: the heartbeat dropped twice in a row here,
  after 46 s and 75 s. The bake now runs before LOD, because the bake changes the
  background and the viewer warned about a stale background tree.
- Metric scale from typical object heights (`src/movable_reality/metric.py`, the default in
  `lift_ids.py`). Muse Spark gives a typical height, a range and "stands on the floor" for
  each noun (one cached call per noun). Every floor-standing object votes with typical /
  measured top above the floor, weighted by 1/log(max/min)^2. A weighted median gives the
  consensus, objects outside their range at that scale are dropped, and the weighted mean
  of the log ratios is the scale. On blender01 it gave ×2.101 against a ground-truth Sim(3)
  of ×2.004, **+4.9%**; the raw scale was 50% too small. The weighted median alone gave
  +9.5%: the measured tops are accurate (coffee table 0.40 m against 0.39 m at the GT scale),
  so the remaining error is the priors, and the mean averages them. Votes, GT scale and box
  sizes are in `benchmarks/metric/blender01_auto_scale.json`.
- With that scale and after per-object training, object boxes are close to ground truth
  without any GT input. Armchair 0.78 x 1.05 x 0.86 m against 0.77 x 1.07 x 0.85, floor lamp
  0.46 x 1.72 x 0.43 against 0.46 x 1.65 x 0.46, plant 1.35 m tall against 1.35, vase 0.41
  against 0.40. The coffee table is 0.57 m tall against 0.39, because the items on it
  joined it. The room shell is 4.23 x 5.25 m against 4.0 x 5.0 (the same +5%).
- SAM `wall` and `ceiling` run on every third frame (`sam_input_every3.mp4`, 51 frames).
  Wall had failed on every long clip; now it gave 7 walls and 113 masks in 312 s.
  `lift_ids.py --sam-only` fills the SAM cache without the reconstruction.
- Room shell height: when the measured height is not a plausible ceiling (2.1 to 4.5 m), the
  shell uses the 99th percentile, then a typical 2.5 m. This walk never looks up, and its
  "densest layer above the cameras" was at 1.80 m.
- Viewer. `/scenes/` lists every local scene with links to the views it supports, and shows
  a live stage strip from `pipeline.json` while a run is going. Alt+click twice is a tape
  measure (`web/src/ui/measure.ts`): from the sofa's backrest to the floor it read 70 cm
  "up", against 0.71 m in GT. The gap overlay draws the 12 most important gaps
  (score x radius^2, `?gapLimit=0` for all), with spheres capped at 0.6 m, so the room
  stays visible.
- Run on this PC (blender01, 30 Oct render with the old lighting, 152 frames,
  `--focal35 25`): MapAnything 97 s, SfM 173 s (152 of 152 registered, 0.68 px), gsplat 509 s
  (quality preset), SAM 13 nouns in about 35 min, alongside. From lift to coverage: 25.9 min
  on the first pass, then 20.0 min for the rerun from the failed stage. Pose ATE 1.8 mm,
  median rotation error 0.03 deg.
- Checked in the browser (desktop pane): scan view, mesh-only (11 of 11 meshes and the room
  shell, no splats), the gap overlay, the tape measure, and one agent command. "How tall is
  the sofa? Then move the coffee table half a metre towards the sofa" gave "0.636 m" (the
  box height). The agent moved the table 0.348 m (the free space), the verifier said no
  ("not 0.5 m"), and the agent then moved it 0.5 m, into the sofa. The verifier then flagged
  the moved table as translucent, and the screenshot agrees: the top of the trained table
  shows the floor through it. The move was undone afterwards.
- Agent fixes after that run. `measure` now reports `top_above_floor_m` (sofa 0.74 m
  against 0.71 m GT; the box said 0.64 m). The agent never pushes into another object: it
  moves as far as fits (floor gap minus 6 cm, checked once with `fits`) and says so. The
  verifier sees the agent's reply and counts "stopped at the sofa" as done; its token budget
  went to 4000 (2000 ran out while it reasoned). The same command then verified on the
  first try.
- Reviewer pass on the branch, fixed: sparse SAM nouns (wall, ceiling, every third frame)
  were outvoted in the lift because their unqueried frames counted as "seen, not this
  label"; `vote` now counts views per label. The tape measure skips the gap overlay and
  hidden objects. The Modal retry matches only connection errors. The runner refuses a
  frames rerun without a video. The shell's height fallbacks apply only to a metric scene
  (`ids/scale.json` method not "none"). Instances of one noun share one scale weight, so six
  chairs do not outvote the sofa. The plan logic moved to `decide()` with unit tests.
- Lift rerun on blender01 with the per-label vote (SAM and Muse Spark from the cache, 141 s):
  wall Gaussians went from 5 to 16,661 of 800k, unlabelled from 130k to 114k, and wall
  splats no longer join the furniture. Sofa 1.78 x 0.71 x 0.98 m (was 1.30 m tall, GT
  0.71), vase 0.44 m tall (was 0.84, GT 0.40). The auto scale stays x2.101. Numbers from
  `ids/assigned.npy` and `logs/lift.log`, before and after.
  Then `run_pipeline.py --from objects` retrained and rebuilt everything (18 min). After
  per-object training the boxes match the first run within a few cm (sofa 1.75 x 0.64 x
  0.82 m, top 0.74), because training already trimmed the wall splats; the gain is the
  untrained split and the wall labels. The bake failed twice on Windows with
  `PermissionError` on the `background_baked.ply` rename: the Vite dev server held the old
  file open. Stopping the dev server, then `--from bake --to lod`, fixed it.
- Timings above come from the local stage logs (`data/scenes/blender01/logs/`, not
  committed); the scale evidence is committed under `benchmarks/metric/`.
- Tests: `pytest` 327 passed, 8 skipped; vitest 169 passed; ruff clean; build passes.
- Next: the translucent tops of trained objects (thin surfaces under items), objects
  standing on furniture kept separate (the bowl, the succulent), and running the pipeline
  on the new-lighting renders and on a real phone video.

## 2026-11-02 (Mon), lanes core, ui: known bugs on blender01 (branch lane/fixes)

- Black and translucent items on trained objects. The bowl on the coffee table has no
  inventory noun, so SAM never labels it. Object training forced the table transparent on
  its unlabelled pixels against a black target: 5,974 bowl splats became 523, median
  brightness 0.66 to -0.02. Unlabelled pixels inside the object's projected 3D box are now
  ignored; outside it they still count, because wall is labelled on every third frame
  only. After retraining the bowl keeps 6,000 splats at 0.66 (`objtrain.py`).
- Items left behind when furniture moves. The support rule needed an item's bottom within
  0.12 m of the parent's top, but SAM misses the succulent's pot and the vase's foot, so
  their splats start 0.14 and 0.19 m above the surface. The rule now accepts a bottom from
  0.1 m below to 0.3 m above the parent's top (a chair tucked under a table, whose seat
  starts well below the top, stays separate; test added). The succulent and the vase merge
  into their tables (D12), 9 objects. Moving the table in the viewer now takes its
  succulent and bowl and leaves no copy. A second rule in object training lets in-box
  background splats that land on neither objects nor floor and wall join the smallest box
  holding them: 584 splats on blender01. A first version also took the splats the junk
  rule removes (13,786 joined, junk 768 instead of about 14,000); the reviewer caught it.
- Ceiling. The shell's ceiling was 0% seen and came out black; an unseen surface (seen
  share inside the outline under 2%) now takes the median colour of the upper half of the
  walls. The viewer's splat styles draw a backdrop from the shell: every wall moved 5 cm
  out along its own normal (normals are now in `room/shell.json`), an unseen ceiling, and
  the shell floor 1 cm under the baked floor, each stretched 8 cm past its edges. The scan
  is a closed room instead of black above about 2 m (`web/src/objects/index.ts`).
- Streaks at the wall tops. Long thin background splats (over 15 cm, 6x their middle axis)
  more than 0.2 m above the highest camera are removed after background training: 35 on
  blender01. The walk sees the wall tops only at a grazing angle. Fainter streaks lower
  down remain; a lower cut would start taking real wall splats.
- Windows file lock. The `/scenes` middleware piped file streams without closing them when
  the viewer aborted a PLY fetch, so Vite kept `background_baked.ply` open and the bake's
  rename failed. `stream.pipeline` closes them; the bake then ran with the viewer open.
  PLY, GLB and scene.json replaces retry for 4.5 s and name the dev server if the file
  stays held (`fsutil.replace_file`).
- The 500 at viewer start was `/api/health` through Vite's proxy before the API server was
  up; the proxy now answers `{"ok": false}`, and other `/api` calls get a 503.
- Rebuilt blender01 from the objects stage five times (about 17 min each, one L40S) and from
  the lift twice. Checked in the browser: scan view with a moved table, mesh-only view.
- Reviewer: needs changes, no blockers; all nine findings fixed (above), and the lane
  notes are in the PR.
- Tests: `pytest` 333 passed, 8 skipped; vitest 172 passed; ruff clean; build passes.
- Still open: the never-filmed floor where a moved table stood shows a smear, and faint
  streaks below the needle cut.

## 2 Nov (later), core: SAM 3D meshes upright and on the floor

- Bug: SAM 3D meshes (and the completed splats, which share their matrix) stood tilted;
  blender01's coffee table leaned 16 degrees and dipped into the floor. The mesh frame was
  right (the mesh matches the completed splats under every axis permutation); the
  registration was not. SAM 3D's single-view pose was 2 to 15 degrees off level, and the
  free ICP fitted the model to a partial scan by tilting it further (coffee table 7 -> 16,
  potted plant#2 2 -> 20, books#0 upside down at 144).
- Fix in `completion.register` (stage 6, so every new scene gets it): the pose guess is
  levelled so SAM 3D's local +Z (its canonical up) lies on world +-Y
  (`splat_ops.level_rotation`), and ICP fits only a rotation about world Y
  (`splat_ops.umeyama_yaw`, `icp(upright=True)`). An object whose opaque scan reaches
  within 5 cm of the floor (0.2 % quantile) then rests on y = 0; the shelf items are not
  moved. `completion.json` reports `levelled_deg` and `floor_shift_m`.
- Review fixes: the pose is always turned to local +Z up (books#0 was 176 degrees, upside
  down); after the floor shift a last ICP with y and scale locked refits yaw and x, z;
  `icp_rms_after_m` is the final pose, `icp_rms_free_m` the pose before the floor rule;
  stale report entries (merged objects) are dropped.
- blender01 rerun from the SAM 3D cache (no Modal calls) and LOD rebuilt. Levelling per
  object (degrees, from `completion.json`): armchair 14.5, sofa 9.5, coffee table 7.2,
  cabinet 5.4, books 176.4, others 1.9 to 3.7. Seven floor objects sit at y = 0 within
  2.3 cm (floor lamp moved down 11 cm, cabinet 17 cm). Final ICP RMS against the free fit:
  coffee table 0.037 (0.032), cabinet 0.078 (0.048), potted plant#1 0.071 (0.055), the
  others within 2 mm. The RMS before this change was overwritten in `completion.json`
  (0.041 for the coffee table, measured in this session, not kept). Viewer check with
  `?objects=mesh`: screenshot not saved.
- Tests: `pytest` test_core_m4 27 passed (5 new: levelling, yaw-only fit, tilted guess,
  floor contact, free-rotation path). ruff clean.
- Still open: the cabinet and potted plant#1 models hit the 1.25 scale clamp and are
  shorter than their scans, so resting them on the floor costs 2 to 3 cm RMS; potted
  plant#2 (on the cabinet) comes out much taller than its scan (219 opaque splats).

## 2 Nov (afternoon), blender: blender06, a realistic bedroom as the main test room

- New `scripts/blender_bedroom.py` builds `blender06`, a 3.6 x 4.2 m single bedroom, 2.6 m
  high, meant to look like a real room so it can be filmed and fed to the pipeline. A
  casement window (white frame, glass, reveal, sill) on the north wall looks out on a park
  HDRI that also lights the room through a light portal, with a soft ceiling-sized fill.
  No lamp is lit by default (no hotspots baked into the splats); `--warm-lamps` lights the
  pendant and the desk lamp.
- Furniture and objects (38 GT objects): a 120 x 200 cm oak bed with an upholstered
  headboard, fitted sheet, pillow; duvet and throw draped by Blender's cloth
  solver (duvet head edge pinned, floor as a collider; a sheet that leaves the room fails
  the build). Nightstand with alarm clock and books, potted plant, desk with laptop (lit
  code editor on screen), mouse, mug, notebook and pen, succulent, desk lamp; chair with a sweater over it, backpack, waste bin; wall shelf with
  an encyclopedia set and a photo frame; chest of drawers with a tin robot, astronaut and hero figures,
  a wooden elephant and books; wall clock, laundry basket, ukulele, wool rug, linen
  curtains on a rod, radiator, skirting, switch, three double sockets, a flush door. Two
  framed public-domain prints: Hokusai's Great Wave above the bed, NASA's Earthrise
  above the desk.
- Assets: 9 new CC0 Poly Haven models, 7 textures and the HDRI, plus the two posters from
  Wikimedia Commons; `scripts/blender_assets.py` now also downloads HDRIs and posters and
  checks that Commons lists each poster as public domain. Credits in `docs/ASSETS.md`.
- Outputs: `data/raw/blender06/blender06.blend`, `blender06_gt/objects.json` and per-object
  PLY meshes in the blender_room format (pass_index = GT index), preview stills in
  `data/raw/blender06/preview/` (6 views, 26 mm phone camera). Build ~45 s with the cloth
  sims, four 1080p stills at 256 samples ~95 s on the RTX 3060.
- Checked by eye on the preview stills (the walks, GT passes and video came later the
  same day, below). ruff clean.
- OJ's review: the leaning cushion and the laptop charging cable removed. The white patch
  at the top of the window was a reflection in the glass (the ceiling-sized fill light and
  the pendant lamp), not the sky. The glass is now a 97 % transparent surface with 3 %
  gloss, and the fill light is hidden from glossy rays.
- Second review: the elephant clipped into the book stack on the chest of drawers, the
  leaning shelf book clipped into the encyclopedias, and the photo frame faced the wall.
  The built shelf books are gone (the encyclopedias stay), items are spaced, and the frame
  is turned (the model's front is its +x side). Its glass faces are dropped: the glTF JPG
  has no alpha, so the glass was opaque and hid the photo. The build now fails when two
  rigid items on the same support overlap in their bottom 15 cm (`check_overlaps`).
- blender06 is a scene in `scripts/blender_room.py` now (`builder: blender_bedroom`), so the
  walks, GT passes, mp4 and `modal_jobs/blender.py` work for it as for the other rooms.
  The desk chair closes the corridor between the bed foot and the desk, so the walker's
  floor is an L (the band south of the bed and the east side up to the chair). Main walk
  54 s: a 24 s chest loop in the L aiming at a ring round the walls (once round), crouch,
  a 12 s knee loop in the south band aiming at the bed side and under the desk, stand up,
  14 s along the chest loop facing the room 15 deg up (ceiling, upper walls). Test walk
  20 s at 1.2 m in the L, looking at a ring round the furniture.
  `tests/test_blender_bedroom.py` (12 tests): 25 cm inside the walls, the eye 10 cm outside
  every footprint at any height (never over the bed or the chair) and 15 cm below its top,
  under 1.6 m/s, under 75 deg/s of turn, the three parts, and `BEDROOM_OBSTACLES` checked
  against the built objects.json. A local check rendered 9 main-walk frames (8 samples)
  and converted their GT.
- Reviewer (high effort) before the PR: the first walk aimed the chest loop at its own far
  side, which swings past the eye on a folded loop (560 deg/s at t = 7 s), and went over
  the bed and the chair; now at most 47 deg/s (main) and 59 deg/s (test). The GT
  `openings` are in the `world_openings` format now, so `blender_capture_guide.py` draws
  them. The cloth sims are deterministic across builds (same vertices with 1 thread and
  all threads), so the Modal chunks agree.
- Rendered on Modal (`modal_jobs/blender.py --scene blender06`, 16 samples, 74 chunks of 30
  frames, 95 to 165 s each): `data/raw/blender06.mp4` (main walk, 54 s, 1620 frames, the
  pipeline input) and `blender06_test.mp4` (20 s, 600 frames), with GT depth, masks and
  cameras in `data/raw/blender06_gt/`. Checked on a contact sheet of every 6th second.
  Next: run the pipeline on it (`run_pipeline.py data/raw/blender06.mp4 --scene blender06
  --focal35 26`).

## 2 Nov (afternoon), lane core: wall and floor artifacts

- OJ: no artifacts on walls and floor. Two scorers measure them on blender01 at the 30 GT
  test-walk poses (every 20th frame) in the real viewer:
  - `scripts/blender_viewer_eval.py` maps the GT cameras into the viewer world
    (`views`), the dev server saves one PNG per pose (`mrEvalRun`, `web/src/bench/evalViews.ts`,
    `POST /__eval`), and `score` counts the pixels on the GT floor, wall and ceiling masks
    (eroded 4 px) whose blurred sRGB error is over 0.12. `sheet` draws a contact sheet.
  - `scripts/artifact_judge.py` asks a Meta vision model (Muse Spark by default; Muse Wonder
    or Muse Spark Open with `--model`) to list surface artifacts with a kind, severity and
    box. It needs no GT, so it also works on real rooms. Responses are cached.
- Causes found: furniture painted onto the shell textures (no depth test), the furniture's
  shadows baked into the floor (stains once it moves), holes under furniture filled with
  smeared colour, and the background splats of walls, floor and ceiling (blur, haze, streaks
  at the wall tops) drawn over the shell.
- Fixes, all in the pipeline:
  - `room_shell.py`: every frame gets an occluder depth buffer (background splats more than
    8 cm from a surface plus every object splat), so texels behind furniture are not
    projected. The floor under and around each footprint (1.2 m reach, 10 cm margin) and
    darker-than-local texels near it (`shadow_mask`) are regenerated, by default by copying
    the surface's own texels at the best-matching offset (`copy_fill`, planks continue in
    line) with a clamped low-frequency seam correction; `--fill muse` keeps the Muse Image
    edit. The floor alpha now reaches 2 texels past the outline: the cut texels had left a
    dotted crack along the wall base. The projection is cached (`--reuse-projection`).
  - New stage `room_surfaces.py` (in `run_pipeline.py` after the bake, before the LOD): a
    background splat goes when its 2-sigma extent comes within 6 cm of a shell surface, when
    it lies outside the room, or when it is haze (over 10 cm, opacity under 0.3), a needle
    (over 5 cm, 8x its middle axis), a floater (under 10 splats in the 30 cm cube around it)
    or a leftover inside an object's box grown by 12 cm (it would stay behind as a ghost when
    the object moves). Each rule was added for an artifact the judge named: the corner fog,
    dark strokes over the sofa, coloured blobs on the unfilmed upper walls, specks inside the
    side table, a rainbow haze above the armchair. blender01 keeps 1,959 of 263,520 splats
    (`objects/background_room.ply`). `room/shell.json` is marked `replaces_splats`, so the
    viewer draws the shell solid in its true planes and hides the baked floor quad;
    `--revert` restores the previous background. Helpers `surface_splats` and
    `neighbour_counts` in `roomshell.py`.
  - Viewer: faint contact shadows under floor-standing objects (`web/src/objects/shadows.ts`,
    0.16 alpha, inside the footprint), moving with the object; `?shadows=0` turns them off.
    Darker or wider shadows scored worse (0.35 alpha: floor 0.052), because the object boxes
    are larger than the objects.
- Eval harness bug: in a hidden browser tab no animation frames run, so Spark's LOD cut lagged
  a pose behind and the first three views showed only the shell or the coarsest splats. The
  driver now renders every 50 ms while it waits.
- Results on blender01 (30 views; pixel share is floor / wall, judge is Muse Spark's
  artifact count / severity sum / frames with an artifact):

  | run | change | pixels floor / wall | judge |
  |---|---|---|---|
  | baseline | splat walls and floor, baked floor quad | 0.035 / 0.0007 | 24 / 54 / 15 |
  | surfaces6 | depth-tested shell, fill, solid shell, background cut at 6 cm | 0.042 / 0.0015 | 32 / 58 / 21 |
  | surfaces8 | floor alpha past the outline (dotted crack) | 0.042 / 0.0015 | 26 / 57 / 16 |
  | surfaces9 | background cut by 2-sigma reach, haze | 0.044 / 0.0037 | 30 / 50 / 16 |
  | surfaces10 | floaters | 0.044 / 0.0034 | 22 / 40 / 11 |
  | surfaces11 | needles, leftovers in object boxes (5 cm) | 0.045 / 0.0039 | 13 / 25 / 9 |
  | surfaces12 | object margin 12 cm | 0.045 / 0.0043 | 3 / 5 / 3 |
  | surfaces14 | surface ghosts (below) | 0.045 / 0.0042 | 6 / 10 / 5 |
  | surfaces15 | fill seams fade in | 0.044 / 0.0043 | 4 / 8 / 4 |
  | surfaces16 | outline from edge orientation | 0.046 / 0.0040 | 2 / 4 / 2 |

  The pixel score went up while the artifacts went down. Its floor errors sit under and
  next to the furniture: the GT has the furniture's contact shadows there, the splat
  floor kept them baked in, and the shell regenerates that floor on purpose so no shadow
  stays behind when an object moves. The object boxes being too large and small pose errors
  add the rest. The 3 remaining judge items: a faint purple haze (000120), a fringe under
  the cabinet (object splats) and a smear behind the coffee table (000420).
- Modal rendering for the synthetic rooms: `modal_jobs/blender.py` runs blender_room.py in
  parallel 30-frame chunks (L4, falling back to A10G or L40S when L4s are short), writes to
  the Volume `movable-reality-renders` and skips chunks already there. Returning the files
  through the client lost the connection after 11 of 60 chunks, hence the volume.
  blender02 both walks with GT: 60 chunks in 6.3 min.
- Evening, after the reviewer pass (verdict "needs changes"; occluders skip the surface
  splats, `background_source()` finds the unreduced background on a rebuild, shell.json
  keeps the `surfaces_*` keys, contact shadows survive the mesh styles, the eval hooks live
  in `web/src/bench/evalViews.ts`, `/__eval` rejects paths outside `data/scenes`):
  - Surface ghosts: the last 1,959 background splats on blender01 were beige fuzz 10 to
    60 cm off the walls and floor in the colour of the wall behind. `room_surfaces.py` now
    drops a splat within 1 m of a surface whose colour is within 0.10 (RGB) of the shell
    texture under it (`surface_colour_gap`); blender01 keeps 161 to 325.
  - Fill seams: `copy_fill` measures the rim mismatch between local means of real texels
    and fades a larger correction (up to 0.2) in over 30 texels from the rim, so a soft
    shadow that runs on past the regenerated floor meets the copy without a step (the
    floor beside blender01's sofa). A hole no copy fits starts from the surface's median
    colour instead of push-pull from its rim.
  - Outline: `main_angle` averages the floor coverage's edge directions with 4-fold
    symmetry; the smallest bounding box had turned blender02's L 6 deg, which split the
    wall at its convex corner. An object box adds floor only within 10 cm of the floor the
    frames saw (blender02's desk#2 box reached 33 cm through the wall). Corners against GT:
    blender01 within 5 cm (9 cm before), blender02 within 6 cm (40 cm before). Snapping
    walls to the densest splat layer was tried and dropped: blender02 has a dense layer
    24 cm inside the alcove wall and the daybed hides the south wall.
  - Not a box room: when more than 10 % of the background splats survive, the room has
    structure the shell does not model and the stage keeps the scan's splats (`--force`).
- More rooms (Modal renders, 30 test-walk views each):

  | room | run | pixels floor / wall | judge |
  |---|---|---|---|
  | blender02 (L-shaped) | baseline (splat walls) | 0.055 / 0.016 | 85 / 175 / 28 |
  | blender02 | surfaces1 (angle 6 deg off) | 0.090 / 0.034 | 92 / 187 / 28 |
  | blender02 | surfaces3 (outline, median fill) | 0.087 / 0.032 | 73 / 155 / 28 |
  | blender04 (attic) | baseline | 0.044 / 0.023 | 71 / 158 / 29 |
  | blender04 | shell applied (`--force`) | 0.046 / 0.028 | 95 / 194 / 29 |

  blender04 now keeps its baseline (the gate). The judge varies by about 10 items between
  identical runs on blender02. Most of blender02's items are now object splats seen against
  the walls (halos round the armchair and plant, colour inside the desk legs), and its
  pixel errors are the regenerated floor under the daybed plus a wall the main walk saw
  darker than the test walk (in the baseline too). blender03 is still in the pipeline.
- Tests: 24 roomshell tests (surface colour gap, the fading seam, the median fill, the
  L-shaped main angle, clipped boxes); `uv run pytest` passes apart from the blender lane's
  uncommitted `test_blender_bedroom.py`; vitest 175; `npm run build` ok.
- Still open, all in the object splats rather than the walls and floor: the white fill
  under blender01's cabinet arch, colour specks inside its side table, halos round
  blender02's armchair and plant, blender04's smeared bed and lamp needles. A mask-vote
  cleanup of object splats was tried and dropped: label 0 in the SAM masks is "unlabeled"
  (the daybed mask covers only its frame), not "wall". Room height falls back to 2.5 m when
  no frame sees the ceiling (the synthetic rooms are 2.6 m). Rooms with sloped ceilings keep
  the scan's walls.

## 2 Nov (evening), blender + core: blender06 through the whole pipeline

- `run_pipeline.py data/raw/blender06.mp4 --scene blender06 --focal35 25 --ceiling-height 2.6`:
  every stage ok in 55 min (reconstruct 18 min, objects 14 min, shell 7 min).
- `--focal35` for Blender renders is 25, not 26: `extract_frames.py` takes the phone
  convention (4:3 photo, 34.6 mm wide), and Blender's 26 mm lens on a 36 mm sensor is
  25 mm in it (1386.6 px at 1920). The first run with 26 (1442 px, 4 % long) gave a pose
  ATE of 3.9 cm RMSE, 1.26 deg median rotation error and an 11 % scale error. With 25:
  median ATE 1.6 mm, RMSE 6.4 mm, 0.06 deg (blender01: 1.3 mm, 0.03 deg); the
  ceiling-height scale is 5 % long (1.749 against 1.667, viewer per GT 1.049).
- Muse Wonder listed 23 nouns; 12 objects in scene.json (bed, desk, chair, dresser,
  nightstand, pillow, two plants, trash can, ukulele, two baskets). Laptop, desk lamp, mug,
  the toy figures, elephant and books rest on furniture and are merged into it by the
  lift; SAM 3.1 found no blanket, backpack or superhero.
- Viewer check from the GT test-walk views (`blender_viewer_eval.py views`): desk, chair,
  ukulele, window, bed, nightstand, plant, posters and door read well; brown smears on the
  duvet next to the throw. Screenshots not saved.
- Next: decide whether items on furniture should stay separate objects in this room (the
  figures on the chest of drawers are the obvious demo targets), and look at the duvet
  smears.


## 2 Nov (night), blender: blender07, the realistic living room

- New builder `scripts/blender_living.py` (blender06 helpers from `blender_bedroom.py`):
  a 5.6 x 4.8 x 2.7 m living room in one modern Scandinavian style (light oak, warm grey
  fabric, matte black metal, white ceramics, abstract art). Two casement windows (north,
  park HDRI, a radiator and curtains each), two closed flush doors (south, white and oak),
  a deep-teal TV wall with a wall-mounted 55" TV (picture on), soundbar, slatted oak media
  console and games console; a low grey sofa with linen cushions and a cloth-draped throw,
  a marble coffee table with books, tray, candle, remote, mug and bowl on a bordered rug, a
  lounge chair, knitted pouf, side table with lamp, tripod floor lamp, opal globe pendant,
  an oak and black-steel bookshelf between the doors, three plants (white planters), five
  public-domain modernist paintings (Kandinsky x2, Mondrian, Klee, af Klint), skirting,
  switches, sockets, thermostat, smoke detector and a TV cable raceway. 53 GT objects.
- First pass mixed styles (carved leather sofa, industrial pendant, baroque armchair, worn
  shelf, brass urn, van Gogh and Monet); OJ asked for one modern style, so the sofa, pouf,
  bookshelf and planters are now built here and the period assets are gone. A contact
  sheet of candidate Poly Haven assets showed no modern sofa among them.
- 6 new CC0 Poly Haven models and 5 Wikimedia Commons paintings in `blender_assets.py`
  (licence checked as public domain on download). Credits in `docs/ASSETS.md`.
- `cloth_sheet` takes the room size (`half=`), so other rooms can use it. Pitfall: some
  glTFs carry keyframes and the cloth solver's frame steps replayed them (the media console
  snapped back to the origin); `blender_living.py` clears the animation data on import.
- `check_clipping` fails the build when meshes interpenetrate (BVH triangle overlap that
  no 3 mm nudge resolves; touching is fine). It found the radiator valves in the curtains,
  the cushions in the sofa (one inside the arm), a plant in a curtain, the armchair legs in
  the rug and the bookshelf 13 mm into the skirting; all fixed. Leaning books now tip back
  onto their row instead of away from it; the photo frame model faces +x, not -y.
- Build ~1 min, eight 960 x 540 previews at 64 samples in `data/raw/blender07/preview/`.
- Next: walks for blender07 in `blender_room.py` (SCENES entry with a builder, clearance
  boxes and a test like `test_blender_bedroom.py`), then the render and the pipeline run.

## 2 Nov (evening), blender: blender07 walks and Modal render

- `SCENES["blender07"]` in `blender_room.py` (builder `blender_living`, same hook as
  blender06), the walks and `LIVING_OBSTACLES` (GT bboxes with what stands on them, +2 cm).
  Every loop goes round the coffee table, through the 46 cm gap between it and the pouf and
  bulges west in front of the TV south of the pouf; the loops are close to convex, so the
  chest loop aims at its own far side (a folded loop swung the view in blender06).
- Length vs data: main walk 62 s (28 s chest at 1.5 m, 2 s crouch, 14 s knee at 0.6 m,
  2 s rise, 16 s at 1.5 m with the phone 15 deg up for the ceiling and upper walls), 1860
  frames, ~250 after the 3-5 fps frame selection. blender06 used 54 s for a room 45 %
  smaller; longer would only add redundant views. Handheld: shake seed 59, at most
  58 deg/s turn, 0.85 m/s, mean 0.48 m/s. Test walk 20 s at 1.2 m (seed 61).
- `tests/test_blender_living.py`: inside the room, clear of furniture, no whip pans,
  handheld speed, obstacles hold the built GT, chest/knee/up parts.
- Rendered on Modal (`modal_jobs/blender.py --scene blender07`, 16 samples, 82 chunks of 30
  frames, ~105 s each, 850 s wall): `data/raw/blender07.mp4` (62 s, the pipeline input) and
  `blender07_test.mp4` (20 s), GT depth, masks and cameras in `data/raw/blender07_gt/`.
  Checked on a contact sheet every 4 s.
- Next: `run_pipeline.py data/raw/blender07.mp4 --scene blender07 --focal35 25`.


## 2 Nov (night), lane core: blender06 floor and walls against GT

- blender06 through room_shell / room_surfaces: the shell outline matches GT within ~3 cm,
  height 2.734 vs 2.728 m. room_surfaces.py `--max-kept` 0.10 -> 0.18: the cluttered bedroom
  keeps 11.5 % of its background splats (lamp, shelf, curtains, radiator), the attic
  (blender04) 25 % and stays gated. A share is a heuristic: a room with a small slope near
  15 % would get the shell (known limit).
- The floor was 72 % regenerated into one flat median, the judge's most frequent complaint
  ("textureless brown smear"):
  - `room_shell.py --floor-contact 0.12`: only what stands on the floor regenerates it.
    `roomshell.base_mask` keeps the floor filmed between a desk's or chair's legs and drops
    a box whole only when the object was painted onto it (texels foreign to the floor and
    within 40 levels of the object's own splat colours); the bed's wooden base had been
    painted 0.45 m past its box. Other foreign texels (the rug, beside the bed or under a
    chair) stay.
  - shadow_mask drop 0.3 on floors (0.18 on walls as before): dark planks are 20-25 %
    below light ones and were taken for shadow.
  - `copy_fill` on floors cuts a hole larger than any clean source into 64-texel pieces,
    filled rim first, so planks continue. With `foreign`, a piece copies only from the
    material round it (rug or floor) and takes its own material's median where they meet;
    copies had run the rug on into the dresser's footprint and planks across the rug.
    Walls and ceilings fill exactly as before (checked texel for texel on blender01).
  - Rim work cropped to each piece: blender06's shell 1371 s -> 323 s, blender01 457 -> 113 s.
- Viewer eval at 960 x 540, artifact share (pixels vs GT) floor / wall, Muse Spark judge
  items / severity / frames:

  | room | before | after |
  |---|---|---|
  | blender06 | 0.020 / 0.025, 84 / 176 / 28 | 0.023 / 0.025, 54 / 110 / 20 (surfaces7) |
  | blender01 | 0.046 / 0.0040, 2 / 4 / 2 | 0.043 / 0.0080, 4 / 6 / 2 (surfaces19) |
  | blender02 | 0.087 / 0.032, 73 / 155 / 28 | 0.090 / 0.032, 65 / 144 / 28 (surfaces5) |

  blender01's wall rise is one kept background splat (a brown blotch beside the cabinet,
  view 000340) that the floor-colour ghost test no longer removes; the wall textures are
  unchanged. The judge varies by about 10 items between identical runs.
- Pitfall: the desktop app's browser pane drops an emulated viewport when its width
  changes; a capture at 1024 x 768 scored blender01 as a big regression. Check the canvas is
  960 x 540 before `mrEvalRun`.
- Remaining on blender06: the rug's edge beside the bed takes the floor median, lacy
  curtain splats, a green floater above the dresser, the desk sits a few cm off GT (the
  scene's scale is 5 % off, scale_viewer_per_gt 1.049).

## 3 Nov (night), blender + core: blender07 whole-room walk, no frame cap

- The first blender07 run looked poor. Two causes: `extract_frames.py` capped training at
  200 frames (about 7 per m² in this 27 m² room), and the 62 s walk only circled the
  coffee table, so corners, pockets and close views were missing.
- `extract_frames.py`: `--max-frames` removed, so every sharp frame at the target fps is
  kept. A longer video now means more frames. Top-up videos keep their own cap of 60.
- `blender_room.py`: `_make_key_walk` places hand-set keys (time, eye, look-at point) and
  interpolates eye position, yaw and pitch with a time-scaled Catmull-Rom spline.
  - blender07 main walk, 122 s: 64 s at 1.5 m through every free corner (bookshelf,
    lounge chair, the south-west door and lamp pocket, TV wall, corner plant, pouf gap,
    window strip, side table, sofa); 36 s at knee height round the coffee table; 20 s
    with the phone tilted up for the upper walls and ceiling.
  - Test walk, 30 s at 1.2-1.7 m, goes the other way between the main views.
  - Turn rate at most 52 deg/s; mean speed 0.25 m/s.
  - New test `test_main_walk_visits_every_part_of_the_room`.
- Rendered on Modal (152 chunks, `--detach`). The chunks keep rendering if the local
  machine goes offline. Only the download and the mp4 encode run locally.
- The old run is kept as `data/scenes/blender07_v1` (200 frames, ATE median 2.0 mm), with
  its videos as `data/raw/blender07_v1*.mp4`.
- Next: pipeline results on the new video, compared with v1.

- Pipeline on the 122 s walk: 171 min end to end after two SAM reruns.
  - 488 frames, all registered. MapAnything held on the L40S.
  - Pose ATE median 3.3 mm, max 36 mm (v1: 2.0 and 62 mm).
  - Test-walk novel views: PSNR 26.1 dB mean, 26.7 median, SSIM 0.914, LPIPS 0.168 (90 views).
  - The weakest views (17-19 dB) all look into the north-west corner from mid-room: the west
    window's curtain and radiator, and the north end of the media console. The main walk only
    sees that corner from close by.
  - 32 objects. The bookshelf is one object now (v1 split it); the floor lamp and the TV are
    missing.
- SAM: the Meta API dropped the bookshelf stream four times on the 490-frame clip, and the
  floor stream twice. `lift_ids.py` now asks for one frame in three after two drops and
  records the noun in `ids/sam_sparse.json`, so reruns and the vote use the same frames. The
  wall stream failed even on the sparse clip; the run went on without wall masks, as the code
  allows.
- Next: cover the north-west corner (a top-up video from mid-room through
  `coverage_topup.py`, or new keys in the main walk), and find out why the lift dropped the
  floor lamp and the TV.

## 3 Nov (day), ui: landing page, three-scene demo, viewer chrome

- No Apple HIG or Material 3 skill exists in the skill catalog, so both guidelines were
  applied by hand: M3 dark colour roles, tonal buttons, menus, switches and state layers;
  HIG translucent materials, system type, 44 px targets, a pause control for the moving
  background and reduced-motion support.
- `/` is now a landing page (`web/src/landing/`, loaded by `web/src/entry.ts`; three.js
  and Spark are only loaded for the viewer). A showcase video plays behind the hero,
  "Open the demo" offers the three rooms, and the page explains what the project does,
  the six pipeline stages with the Meta model each calls, and the model list. Every
  viewer URL (`?scene=`, `?ply=`, `?bench=`) still opens the viewer.
- Demo rooms in `web/src/landing/scenes.ts`: living room (`blender07`), bedroom
  (`blender06`), real room (`room01`, shown as "Coming soon" until it has a scene.json).
  Cards read `/__scenes` on the dev server ("Updating" while a stage is running) and
  fall back to HEAD on `scene.json` on a static host.
- Viewer chrome (`web/src/ui/chrome.ts`): a top bar with back to the start page, a room
  switcher, Enter VR (Spark's button, restyled, no longer the large default box), help and
  a settings menu (coverage gaps, generated splats, object style, developer info).
  A loading card stays up until the background splats have loaded, and tells you when a
  room is still being processed.
- Developer info is hidden by default: the HUD (also in XR), the save-log button, the
  detail slider and the object style label carry the class `mr-debug`. The settings
  menu's last switch or the ` key shows them, and the choice is remembered. `?debug=1|0`,
  and `?bench=1` or `?hud=1` start with it on, so benchmark runs are unchanged.
- Command bar: friendlier offline text ("Agent offline"; the server command goes to the
  console), SVG microphone, a status dot only on phones. The onboarding sample commands
  now name objects of the open room.
- Placeholder media: `showcase.mp4` is 23 s cut from the blender07 and blender06 walk
  renders (gitignored like every mp4); swap in a screen recording of the viewer for the
  final page. Card images are capture frames.
- Checked in the browser pane at desktop size and 375 x 812: landing, dialog, viewer load,
  menus and the developer-info switch; no horizontal scroll on the phone. `npm test`
  (189) and `npm run build` pass. Not checked on a headset or a real phone.
- Next: a real showcase recording, `room01` once filmed, and its card image.

## 3 Nov (evening), ui + docs: new hook, no VR, input videos per room

- Landing hero rewritten in plainer words ("Try the new layout before you lift the sofa."),
  stats no longer quote 30 s or 72 FPS (the living-room walk is 122 s).
- VR dropped from the demo and the claims (D21 in `docs/HANDOFF.md`): untested on a headset.
  The landing page no longer mentions it, and the viewer's Enter VR button is now developer
  info only.
- New landing section "The videos each room was built from": the exact walk each scene
  trained on (`capture_<scene>.mp4`, 960 px copies of `data/scenes/<scene>/input.mp4`,
  1.9 MB and 4.4 MB, gitignored). The real room shows "Capture coming soon" until filmed.
- README rewritten for the current state (try the demo, viewer controls, developer info,
  media, rooms); PLAN, ARCHITECTURE, MARKETING and AGENTS.md follow D21; HANDOFF has a
  3 Nov section.
- Review fixes: a failed object PLY no longer fails the whole room on the loading card,
  sample commands never pair an object with itself, H no longer shows the XR HUD panel
  with developer info off, settings popover without menu roles, landing numbers match
  the data (no object or frame counts).
- Next: a real screen recording for the hero, then room01.

## 3 Nov (night), core + ui: the agent in the demo works end to end

Tested in the viewer on blender06 with the scene server, Muse Wonder and the Muse Spark Open
verifier. Each fix below came from a command that went wrong:
- **Verifier saw the wrong thing.** It got screenshots of the user's view, which often did
  not show the edited object (the bedroom opened facing a wall), and rejected correct edits.
  The before shot now waits for the first edit and frames that object and where it goes
  (`edit_focus`, `verify_camera` in `agent/verify.py`), from a point between the object and
  the user: the room box from the splats includes the view through the windows, and a
  box-clamped camera ended up outside. After the edit the same viewpoint is used.
- **Verifier called scan artifacts damage** ("toppled", "smeared"). Its prompt now says
  blur, ghosting and dark patches are normal in a splat scan.
- **Agent gamed the verifier:** after a rejection it moved the chair to mid-room "so it is
  visible". The retry message now says to change nothing when the tool results show the edit
  was done, and never to move an object just for the check.
- **"Turn the chair" did nothing:** the chair overlaps the desk in the scan, every `fits`
  said "collides" and the agent ran out of steps. Overlaps that already existed where the
  object was scanned no longer count, and the prompt caps `fits` at three calls.
- **First command on the living room froze the server for minutes:** room bounds read the
  283 MB background PLY in full, inside the event loop, so no WebSocket message went out.
  Now memory-mapped and strided (0.18 s) and run in a thread.
- **Viewer screenshots in a background tab** had no furniture: a hidden tab runs no
  animation frames, so the capture now renders 12 times over 1.5 s itself.
- **Command bar:** one progress line in words ("Measuring the bed…", "Checking the
  result…") instead of raw tool calls (those stay as developer info), no duplicate reply,
  "✓ Checked" when the verifier agrees. Replies name objects by label, not id.
- **Start view:** each demo room opens on its furniture (`view` in `landing/scenes.ts`), and
  the loading card waits for the objects as well as the background.
- Result on blender06: "how wide is the bed" 5 s, "remove the ukulele" 12 s, "undo that"
  15 s, "turn the chair 45 degrees" verified. Not tested: voice (no microphone here) and
  insert. The living room was rebuilding in another session, so it was only checked for the
  bounds fix. The demo manifests were restored after each test.
- Next: test voice and the living room once its pipeline run is done.
