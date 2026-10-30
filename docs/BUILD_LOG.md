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

## 2026-10-30 (lane: lod)

- L1: `SceneObject.completed_lod` in `scene.py` and `scene.ts`. `build_lod.py` builds
  `lod/obj_<id>_completed.rad` for every object with a `completed_ply` on disk and records
  it (missing completed PLYs are skipped, reruns are idempotent).
- The viewer swaps an object to its completed PLY once its transform differs from identity
  (`shownPly` in `web/src/scene.ts`, the former M4 TODO) and back on undo. Both variants
  stay in the scene graph and the unused one is hidden, so a swap back is instant and no
  worker tree leaks. `lodChoice` loads `completed_lod` for the completed PLY.
- L2 (D7): `installLod` sets `spark.lodCleanupTimeoutMs = Infinity`.
- Checked in the browser on `demo_l1` (a local copy of `demo` with a red-tinted chair as
  the fake completion, not committed): move turns the chair red, undo turns it blue,
  rotate red again; Spark kept 5 trees and 4 traversed meshes throughout.
- Next: L3 (priority tuning on the headset); real completions arrive with M4.
