# Handoff: continue locally in Claude Code

Written Thu 29 Oct 2026 for OJ's local Claude Code sessions, after the move from Cowork.
Updated the same evening when coverage merged, and on Fri 30 Oct afternoon (Windows PC
setup, Modal image, Blender room, scope D18; see 3.0). Read this file first, then `AGENTS.md` (rules, lane ownership),
`docs/PLAN.md` (schedule, benchmarks, video), `docs/ARCHITECTURE.md` (stages) and
`docs/DEVICES.md` (headsets). Deadline: Wed 4 Nov 21:00 ET = **Thu 5 Nov 03:00 Prague**.

## 1. Where things stand

`main` contains the scaffold plus four merged lanes, the Windows setup fixes and the
Blender ground-truth room (30 Oct). Nothing has run on a headset or been reconstructed
yet. The Modal image for `reconstruct.py` is built and cached, but the GPU job is blocked
until a payment method is added to the Modal account. Viewer results so far come from
synthetic data in desktop browsers (no performance meaning).

| Lane | State | Where |
|---|---|---|
| lod | Merged (PR #1). Per-object Spark LOD trees (`.RAD` via Spark's `build-lod`, compiled from Spark's repo), load-time `lod: true` fallback, one shared splat budget, priority (`lodScale` 2.0, 300 ms ramp) for the object under the pointer/controller/head ray and the last-edited object. | `src/movable_reality/lod/`, `scripts/build_lod.py`, `web/src/lod/`, `docs/LOD.md` |
| bake | Merged (PR #2). RANSAC plane fit, CPU orthographic splat bake, hand-written GLB quads, reduced background PLY, viewer loader. Splats depth-test against the quads (verified). GPU job (gsplat 1.5.3 ortho) written, never run. | `src/movable_reality/bake/`, `scripts/bake_planes.py`, `modal_jobs/bake.py`, `web/src/planes/` |
| renderer | Merged (PR #3). HUD, frame logs (`/__bench` endpoint), budget controller (owns `lodSplatCount`, `lodRenderScale`), naive/LOD toggle, bench mode with walk paths, `/r0` run menu, synthetic bench scenes, `bench_report.py`, LAN HTTPS (`MR_HTTPS=1`). | `web/src/perf/`, `web/src/bench/`, `scripts/bench_*.py`, `docs/DEVICES.md` |
| coverage | Merged (PR #4). Synthetic capture with ground truth, per-Gaussian view count and angle spread, voxel gap map inside the room box, gaps and next-best views in `scene.json`, viewer gap overlay (`?gaps=1`, G, XR squeeze), Muse Image repair groundwork (gap view, references, cached edit call; no live call yet). U-3DGS uncertainty job on Modal written, never run. | `src/movable_reality/coverage/`, `scripts/score_coverage.py`, `scripts/coverage_repair.py`, `web/src/coverage/` |
| blender | Merged (PR #6). Synthetic CC0 room `blender01`: 38 s main walk and 20 s test walk rendered (Cycles + OIDN, D19), GT cameras, depth, instance masks and meshes in `data/raw/blender01_gt/` (on the Windows PC only), `blender_gt_eval.py` (pose ATE, mask IoU; completion chamfer is a stub). | `scripts/blender_*.py`, `docs/ASSETS.md` |
| ui | **Not started** (D18). | `web/src/ui/` |
| core (M1-M5) | **M1 in progress.** `reconstruct.py` image builds (sm_89 only, g++; 41 min once, cached). First GPU run blocked on Modal billing. Private test scene `testroom` (200 frames, motion-blurred, never publish). Agent, server, voice, insert, completion not started. | `scripts/`, `modal_jobs/reconstruct.py`, `src/movable_reality/{sam,clients}.py` |

## 2. Decisions taken on 29 Oct

| # | Decision | Source |
|---|---|---|
| D1 | Every model called at runtime is a Meta model. Coding tools are free (Claude Code, Muse Code). Difix3D+ and other non-Meta models are out; gap repair uses Muse Image only. | Cerebral Valley in Discord, 28 Oct |
| D2 | MapAnything and SAM 3D Objects count as Meta open-source models. | Same ruling |
| D3 | Keep AI co-author trailers on commits; no history rewrites, no force pushes. | OJ |
| D4 | Muse Code may run in Contributor mode (the repo is public and the credits already allow training). API calls keep Standard-mode model IDs. | OJ |
| D5 | Renderer's budget controller owns `lodSplatCount`; `?lodSplatCount=` is only the starting value. Controller on by default outside bench mode. | Renderer Q4, Q5 |
| D6 | R0 bench runs pin the budget at 500k for fair comparison, plus one run per headset with the controller on, on the largest scene. | Renderer Q2 |
| D7 | Keep hidden objects' LOD trees in memory (`lodCleanupTimeoutMs = Infinity`). | LOD Q2 |
| D8 | Add `SceneObject.completed_lod` so SAM 3D Objects completions get a precomputed tree (task L1). | LOD Q4, OJ agreed |
| D9 | `lift_ids` writes `ids/labels.json` (id to SAM label), and floor and ceiling splats stay in the background PLY. | Bake Q1, Q2 |
| D10 | Bake keeps 5 mm texels; floor-only fallback if wall seams show. | Bake Q3, Q4 |
| D11 | Bake gets an alpha mask so non-rectangular floors do not extend past the walls (task B1). | OJ agreed |
| D12 | Items resting on furniture belong to that furniture (support rule in `lift_ids`, task C-M2). | OJ agreed |
| D13 | Main demo room: living room or bedroom. Second: a different layout (L-shaped or with a doorway). Optional third: kitchen, for the "does it fit" insert. No bathroom (mirrors, glass, fixed fixtures). | OJ agreed |
| D14 | Steam Frame: OJ already has a Chromium XR build with WebXR on it; no install needed. | OJ |
| D15 | Evaluation metrics with no Meta equivalent (LPIPS in gsplat's trainer and in the benchmarks) may use their usual networks. Only product runtime models must be Meta models. | OJ, 29 Oct |
| D16 | Build order: capture, M1, M2 and M3 first; headset work on headset days (PLAN, "Build order"). | OJ asked for the best order, 29 Oct |
| D17 | C2 first does a full reconstruction rerun on all frames (best quality); the fast fine-tune path comes second and replaces it in the demo only if it comes close. | OJ, 29 Oct |
| D18 | Scope: the viewer is a full product in a desktop browser too (mouse and keyboard, typed and spoken commands), not only in VR. The hosted demo link must work on a laptop with no headset. VR stays the frontier claim and the performance target. New lane **ui** (U1-U4, section 5). | OJ, 30 Oct |
| D19 | Blender renders may use Cycles' OpenImageDenoise (offline data creation, not a runtime model) | OJ, 30 Oct |

Still open, OJ decides: **feature freeze date** (suggested Mon 2 Nov 20:00, the latest that
leaves Tue for the video), **borrowing a Quest 2** for one Meta-headset clip, **hosting** of
the demo (needs chunked `.RAD` or HTTP Range support, see `docs/LOD.md`).

## 3. Next steps

### 3.0 Later on Fri 30 Oct (written at the end of the PC session)
1. **Modal billing**: add a payment method (modal.com, Settings, Usage & Billing) and set a
   workspace spend limit. Credits are still used first.
2. **M1 on the test video**, from the Windows PC (frames are there, gitignored):
   `uv run modal run modal_jobs/reconstruct.py --scene testroom --max-steps 7000`. The image
   is cached, so it starts on the L40S at once. Expect fixes in the job itself (MapAnything
   export, gsplat trainer flags); it has never run past the image build. Then open
   `?scene=testroom` in the viewer.
3. **M1 on `blender01`** (Windows PC; `extract_frames.py` already ran): same command with
   `--scene blender01`, then `blender_gt_eval.py` for the pose error against ground truth.
   This is the first real number for the README.
4. **Film room01** and its test walk: slow, 60 fps if possible, exposure and focus locked
   (the test video was blurred). Into `data/raw/room01.mp4` and `room01_test.mp4`.
5. **R0 on the Steam Frame**: `npm run dev`, then `ssh -N -R 5173:localhost:5173 frame`
   from the same machine, and open `http://localhost:5173/r0` in Chrome XR on the Frame
   (section 3.3; the tunnel replaces HTTPS).
6. **Port the gsplat build fixes** (sm_89 or one arch, g++ env, `-v`, import check) to
   `modal_jobs/coverage.py` and `bake.py` before their first run.
7. Start lanes that need no real room: **U1** (desktop picking and dragging, ui lane) and
   **M3** agent tools on the `demo` scene.
8. On the Mac: `git pull`; it has no `.env` yet (copy it by hand, never through git) and
   needs `uv run modal token new` once.


### 3.1 Setup on the Mac (once)
1. Tools: git, Python 3.12 with uv, Node 22 or later, ffmpeg, Rust/cargo (for
   `build-lod`), adb (Pico over USB), Modal CLI (`uv run modal setup`). State on 29 Oct:
   git, uv, Node 24, ffmpeg, gh, adb and Rust (Homebrew rustup, keg-only) are installed,
   and `build_lod.py --setup` has built `build-lod` into `~/.cache/movable-reality/`.
   Rust is not on the shell PATH; only a new `--setup` needs it
   (`PATH=/opt/homebrew/opt/rustup/bin:$PATH`). Missing: the Modal token, `.env`.
2. Keys in environment variables or `.env` (never in prompts): `MODEL_API_KEY`,
   `BASETEN_API_KEY`, `HF_TOKEN` (gated SAM 3D Objects weights). Modal token via
   `modal setup`.
3. `uv sync`, `cd web && npm ci`, `uv run python scripts/build_lod.py --setup`.
4. Smoke tests: `smoke_meta.py`, `smoke_meta.py --image ... --noun chair`, `smoke_baseten.py`,
   and one tiny Modal call. Note results in `docs/BUILD_LOG.md`.

### 3.1b Setup on the Windows PC (done 30 Oct)
RTX 3060 Laptop (6 GB), Ryzen 5 5600H, 16 GB, WSL2. git, gh, uv, Node 24, cargo, adb and
ffmpeg (winget `Gyan.FFmpeg`, new shells only) installed; `uv sync`, `npm ci`,
`build_lod.py --setup` (builds `build-lod.exe`) done. Keys in `.env` (it was saved as `env`,
which git does not ignore; renamed). All keys checked live, see the build log of 30 Oct.
Modal token set (30 Oct, profile `ojytcanal`); Blender installed by the blender lane.
Windows fixes made on the way: PLYs read with `mmap=False` (a mapped file cannot be
replaced on Windows), manifest paths written with `as_posix()`, `build-lod.exe` lookup.

### 3.2 Film (nothing is filmed yet; this is the critical path)
- Rooms per D13. Even light, lamps on. 4-8 clearly separate movable pieces in the main room
  and a few items on surfaces. Avoid mirrors, glass tables, screens that are on, people,
  pets, moving curtains, moving sun patches.
- Capture: 30-45 s, slow, phone horizontal, 4K30 or 1080p60, exposure and focus locked if
  possible. Walk the perimeter facing inwards, then one pass at knee height.
- For every room also film a **separate 20 s test walk** along a different path. It is never
  used for training; it measures off-path quality (benchmark 7 in `docs/PLAN.md`).
- Raw files go to `data/raw/` (gitignored), e.g. `room01.mp4`, `room01_test.mp4`.

### 3.3 R0 device baseline, Fri 30 with the headset (`docs/DEVICES.md` section 8)
`uv run python scripts/bench_make_splats.py`, then `cd web && npm run dev`.
Pico 4: `adb reverse tcp:5173 tcp:5173`, open `http://localhost:5173/r0` in Pico Browser.
Steam Frame: `MR_HTTPS=1 npm run dev`, open `https://<pc-ip>:5173/r0` in Chromium XR
(HTTPS mode binds all interfaces: home network only). 10 runs per headset plus D6's
controller run, then `uv run python scripts/bench_report.py`; commit `benchmarks/raw/` and
`benchmarks/RESULTS.md`. Also judge by eye: HUD readability, priority strength (L3),
popping when objects move.

### 3.4 M1: first real room
`extract_frames.py data/raw/room01.mp4 --scene room01`, then
`modal run modal_jobs/reconstruct.py --scene room01`; open `?scene=room01`. Expect the
Modal image build to need fixes on the first run (never run so far). Then try
`build_lod.py room01` and the viewer on the headset.

## 4. Schedule

See `docs/PLAN.md`, "Build order". In short: Fri 30 and Sun 1 to Wed 4 have the headset.
Sat 31 OJ is away, so that day's sessions run M2, M3, C1, L1, L2 and B1 on room01 without
the headset.

## 5. Tasks by lane

Each task: branch `lane/<lane>`, PR into main, done when its check passes and the build log
is updated. Before real scenes exist, use the synthetic scenes (`demo`, `bakedemo`,
`bench_*`, the coverage synthetic capture).

**core** (new lane; owns `scripts/`, `modal_jobs/reconstruct.py`, `src/movable_reality/{sam,clients}.py`, new `agent/`, `server.py`)
- M1 Reconstruction on Modal works for room01 (section 3.4). Done when `?scene=room01` shows
  a metric scene.
- M2 `scripts/inventory.py` (Muse Wonder lists objects + "floor", "wall", "ceiling"), SAM 3.1
  per noun, `scripts/lift_ids.py` (vote per ARCHITECTURE stage 5) writing
  `ids/gaussian_ids.npy`, **`ids/labels.json`** (D9), per-object PLYs (structure stays in
  the background), floor alignment into `world_from_recon`, and the **support rule** (D12):
  unlabelled splats resting on an object's top surface inside its footprint join that object.
  Done when moving the table in room01 moves what stands on it and leaves no floating items.
- M3 Agent tools on `scene.json` (ARCHITECTURE stage 7), FastAPI server with WebSocket
  replacing the 1 s polling, verifier on Muse Spark Open (Baseten). Done when "remove the
  chair, move the lamp left" works with a logged verification.
- M4 Insert path (Muse Image, SAM 3.1, SAM 3D Objects, floor snap, `fits`) and object
  completion (`obj_<id>_completed.ply`, splats flagged `generated`); viewer swaps to the
  completed mesh once an object is moved, using `completed_lod` (L1).
- M5 Voice (Muse Voice Transcribe) and all three scenes end to end.

**ui** (new lane, D18; owns `web/src/ui/`, one `installUi()` call in `main.ts`)
- U1 Desktop editing without the agent: click an object (ray against the object boxes from
  `scene.json`), highlight it, drag it on the floor plane, Q/E rotate, Delete hides, Ctrl+Z
  undoes through `history`. Same edit path as the agent tools, so the server and the
  viewer stay in step. Done when every agent tool has a mouse/keyboard equivalent.
- U2 Command bar: a text box plus hold-to-talk (space on desktop, a controller button in
  VR). The browser records with an AudioWorklet and encodes 16 kHz mono 16-bit WAV itself
  (the ASR endpoint takes only WAV, 16 or 24 kHz; we send 16), sends it to the server; the transcript and the
  agent's reply appear in the bar. Needs M3's server.
- U3 Hosted demo: viewer + server in one public deployment (proposed: a Modal ASGI app,
  credits cover it), keys only on the server, per-visitor cost cap and rate limit, an
  "edits are local to your session" scene copy, and Range or chunked `.RAD` for big files.
  Also a no-key fallback: scripted commands with cached agent replies.
- U4 Onboarding overlay for judges: controls, sample commands, link to the video,
  "generated" toggle and gap overlay as buttons.

**lod**
- L1 `SceneObject.completed_lod: str | None` in `scene.py` and `scene.ts` with tests;
  `build_lod.py` builds it when `completed_ply` exists; viewer uses it on the swap.
- L2 `lodCleanupTimeoutMs = Infinity` on the SparkRenderer (D7).
- L3 Tune priority strength and ramp from the headset session.
- After any bake, rebuild the background tree (`build_lod.py --force`); the viewer already
  warns when the background tree predates the bake.

**bake**
- B1 Alpha mask: mark texels no splat covered (beyond small-hole filling) as transparent and
  render the quad with alpha test, so L-shaped floors stop at the walls. Test on a synthetic
  L-shaped room.
- B2 Port the gsplat build fixes from `reconstruct.py` (as C0), then run `modal_jobs/bake.py` once and compare with the CPU bake.
- B3 Real room: check seams at walls and under furniture; floor-only fallback (D10).

**renderer**
- From R0 data: per-device starting budgets (`lodSplatCount`) and controller thresholds.
- R4: the full ablation table on both headsets (PLAN benchmark 6).
- Later, for the demo walk: smooth locomotion with snap turn in VR outside bench mode.

**coverage**
- C0 Before the first run of `modal_jobs/coverage.py`: port the gsplat build fixes from
  `reconstruct.py` (one CUDA arch, gcc/g++ env, `pip -v`, import check), or it takes hours
  and fails at the link (30 Oct).
- C1 on room01 once M1 has produced it; first live `coverage_repair.py --call` (needs
  `MODEL_API_KEY`).
- C2 Recapture loop: show the next-best viewpoint in the viewer, register the top-up video
  with MapAnything together with the original frames, gsplat fine-tune, re-score.
- C3 Repair: render gap views, Muse Image edit with nearest real frames as references,
  distil back, flag `generated`, "show generated" toggle in the viewer.

## 6. Running lanes locally

- One Claude Code session per lane, each in its own worktree:
  `git worktree add ../mr-<lane> -b lane/<lane> origin/main` (or `origin/lane/<lane>` to
  continue an existing branch). Never share a working folder between sessions.
- Commit small and push at least every 30 minutes (protects work if a usage limit is hit).
- Merge order and conflicts: lanes append to `docs/BUILD_LOG.md` and the Commands block in
  `AGENTS.md`, so those conflict; keep both sides. Lane installs in `web/src/main.ts` run in
  this order: planes, lod, coverage, perf (perf wraps lod's `meshOptions`).
- Before merging a PR: the checks in AGENTS.md "Definition of done", plus
  `uv run ruff format --check .`, and open the viewer once.

### Starter prompts for local sessions
- Core: "Read docs/HANDOFF.md and AGENTS.md. You own the core lane. Do M2 from section 5
  (lift_ids with labels.json, support rule, floor alignment) against the demo scene first,
  then on room01 once M1 has produced it. Branch lane/core, PR when done."
- Coverage: "Read docs/HANDOFF.md and AGENTS.md. Run C1 on room01, then do C3 from
  section 5 on branch lane/coverage."
- LOD: "Read docs/HANDOFF.md and AGENTS.md. Do L1 and L2 from section 5 on branch lane/lod."
- Bake: "Read docs/HANDOFF.md and AGENTS.md. Do B1 from section 5 on branch lane/bake."

## 7. Facts learned on 29-30 Oct

- (30 Oct) Muse Voice Transcribe is **not** at the OpenAI `/audio/transcriptions` path
  (404). It is `POST https://api.meta.ai/v1/asr/transcribe`, multipart with a `request`
  part (JSON `{"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV", "mode":
  "PUSH_TO_TALK"}`, type application/json) and an `audio` part (WAV, mono, 16-bit PCM,
  16 or 24 kHz, at most 10 min / 32 MB). Response `{"transcript": ..., "audioDurationMs":
  ...}`. Realtime: `wss://api.meta.ai/v1/asr/realtime`, token in the first JSON frame.
  $0.18 per audio hour. Source: https://dev.meta.ai/docs/speech-to-text
- (30 Oct) Muse Image text-to-image works through `client.images.generate(model=
  "muse-image-1.0", prompt=..., size="1536x1024")`, result in `data[0].b64_json`.
- (30 Oct) Steam Frame: SteamOS 3 on aarch64, 8 cores, 15 GiB RAM, `ssh frame` (user
  `steamos`). Chrome XR launcher at `~/.local/bin/chrome-xr`. The reverse tunnel
  `ssh -N -R 5173:localhost:5173 frame` works: the Frame fetches the PC's dev server at
  `http://localhost:5173`, a secure context with no certificate warning.

- Spark 2.4.0: all LOD meshes share one budget in a joint traversal; `lodScale` scales a
  mesh's share; Spark does not re-traverse on a `lodScale` change or a one-frame move, so
  the lod lane sets `spark.lodDirty = true`. Default `lodSplatCount` depends on the user
  agent: 500k only for the Quest browser, 1M other Android, 1M or 2.5M elsewhere.
  `build-lod` panics on an empty PLY and exits 0 without output on invalid splats
  (handled in `build_lod.py`).
- Splats are hidden behind opaque meshes because SparkRenderer's material uses depthTest
  true and depthWrite false and draws in three.js's transparent pass. Never pass
  `depthTest: false` to SparkRenderer.
- Muse Image edit (Meta Model API, OpenAI-compatible): `client.images.edit(model=
  "muse-image-1.0", prompt=..., image=<file or list of files>, n=1, size=...)`, base URL
  `https://api.meta.ai/v1`, key passed explicitly; result in `data[0].b64_json`; `size` sets
  only the aspect ratio, output up to about 1600 px; reopen reference files for every call;
  $0.01 per image. Source: https://dev.meta.ai/resources/blog/build-with-muse-Image
- gsplat 1.5.3: `rasterization(..., camera_model="pinhole"|"ortho"|"fisheye"|"ftheta",
  render_mode="RGB"|"D"|"ED"|"RGB+D"|"RGB+ED")`. Source:
  https://rocm.docs.amd.com/projects/gsplat/en/latest/reference/gsplat-api-reference.html
- Steam Frame: the built-in browser has no WebXR; OJ's Chromium XR build does. Pico 4
  browser supports WebXR; over USB use `adb reverse`.
