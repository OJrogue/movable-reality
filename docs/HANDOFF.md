# Handoff: continue locally in Claude Code

Written Thu 29 Oct 2026, 18:30 Prague, by the cloud orchestration session, for OJ's local
Claude Code sessions. Read this file first, then `AGENTS.md` (rules, lane ownership),
`docs/PLAN.md` (schedule, benchmarks, video), `docs/ARCHITECTURE.md` (stages) and
`docs/DEVICES.md` (headsets). Deadline: Wed 4 Nov 21:00 ET = **Thu 5 Nov 03:00 Prague**.

## 1. Where things stand

`main` contains the scaffold plus three merged lanes. Nothing has run on a headset, on a
real room or on Modal yet; every result so far is from synthetic data and headless
Chromium (about 1 FPS, so no performance meaning).

| Lane | State | Where |
|---|---|---|
| lod | Merged (PR #1). Per-object Spark LOD trees (`.RAD` via Spark's `build-lod`, compiled from Spark's repo), load-time `lod: true` fallback, one shared splat budget, priority (`lodScale` 2.0, 300 ms ramp) for the object under the pointer/controller/head ray and the last-edited object. | `src/movable_reality/lod/`, `scripts/build_lod.py`, `web/src/lod/`, `docs/LOD.md` |
| bake | Merged (PR #2). RANSAC plane fit, CPU orthographic splat bake, hand-written GLB quads, reduced background PLY, viewer loader. Splats depth-test against the quads (verified). GPU job (gsplat 1.5.3 ortho) written, never run. | `src/movable_reality/bake/`, `scripts/bake_planes.py`, `modal_jobs/bake.py`, `web/src/planes/` |
| renderer | Merged (PR #3). HUD, frame logs (`/__bench` endpoint), budget controller (owns `lodSplatCount`, `lodRenderScale`), naive/LOD toggle, bench mode with walk paths, `/r0` run menu, synthetic bench scenes, `bench_report.py`, LAN HTTPS (`MR_HTTPS=1`). | `web/src/perf/`, `web/src/bench/`, `scripts/bench_*.py`, `docs/DEVICES.md` |
| coverage | **Not merged.** Was still running in the cloud session at handoff. Branch `lane/coverage` (merged with main at 12c3f6c) has: synthetic capture with ground truth, per-Gaussian view count and angle spread, voxel gap map, gaps written to `scene.json`, next-best views, U-3DGS uncertainty job on Modal (not run), viewer gap overlay (in progress). Missing: Muse Image repair client, finished overlay, PR. | `src/movable_reality/coverage/`, `web/src/coverage/` |
| core (M1-M5) | **Not started.** Reconstruction, objects, agent, server, voice, insert, completion. | `scripts/`, `modal_jobs/reconstruct.py`, `src/movable_reality/{sam,clients}.py` |

Check first: is there a PR from `lane/coverage`? If yes, review and merge it (section 6).
If not, continue the coverage lane from its branch (section 5, coverage).

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

Still open, OJ decides: **feature freeze date** (suggested Mon 2 Nov 20:00, the latest that
leaves Tue for the video), **borrowing a Quest 2** for one Meta-headset clip, **hosting** of
the demo (needs chunked `.RAD` or HTTP Range support, see `docs/LOD.md`).

## 3. Tonight

### 3.1 Setup on the PC (once)
1. Tools: git, Python 3.12 with uv, Node 22, ffmpeg, Rust/cargo (for `build-lod`), adb
   (Pico over USB), Modal CLI (`uv run modal setup`).
2. Keys in environment variables or `.env` (never in prompts): `MODEL_API_KEY`,
   `BASETEN_API_KEY`, `HF_TOKEN` (gated SAM 3D Objects weights). Modal token via
   `modal setup`.
3. `uv sync`, `cd web && npm ci`, `uv run python scripts/build_lod.py --setup`.
4. Smoke tests: `smoke_meta.py`, `smoke_meta.py --image ... --noun chair`, `smoke_baseten.py`,
   and one tiny Modal call. Note results in `docs/BUILD_LOG.md`.

### 3.2 Film (nothing is filmed yet; this is the critical path)
- Rooms per D13. Even light, lamps on. 4-8 clearly separate movable pieces in the main room
  and a few items on surfaces. Avoid mirrors, glass tables, screens that are on, people,
  pets, moving curtains, moving sun patches.
- Capture: 30-45 s, slow, phone horizontal, 4K30 or 1080p60, exposure and focus locked if
  possible. Walk the perimeter facing inwards, then one pass at knee height.
- For every room also film a **separate 20 s test walk** along a different path. It is never
  used for training; it measures off-path quality (benchmark 7 in `docs/PLAN.md`).
- Raw files go to `data/raw/` (gitignored), e.g. `room01.mp4`, `room01_test.mp4`.

### 3.3 R0 device baseline (`docs/DEVICES.md` section 8)
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

## 4. Schedule (unchanged from `docs/PLAN.md`, shifted by the late start of capture)

| Day | Core | Coverage | Renderer, lod, bake | Ship |
|---|---|---|---|---|
| Thu 29 | Setup, filming, M1 | Finish PR | R0 on both headsets | |
| Fri 30 | M2 (lift_ids, labels, support rule, floor alignment) | Run on room01; C2 groundwork | L1, B1; LOD and bake on room01 | |
| Sat 31 | M3 (agent tools, server, verifier) | C2 recapture loop | Tune from R0 data | |
| Sun 1 | M4 (insert, SAM 3D Objects, completed swap) | C3 Muse Image repair + tag | R3 on device | |
| Mon 2 | M5 (voice, all scenes) | Coverage metrics | R4 benchmarks | Freeze (OJ decides) |
| Tue 3 | | | | Video; draft submission by 20:00 |
| Wed 4 | | | | Final cut, README; submit by 20:00 |

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
- B2 Run `modal_jobs/bake.py` once and compare with the CPU bake.
- B3 Real room: check seams at walls and under furniture; floor-only fallback (D10).

**renderer**
- From R0 data: per-device starting budgets (`lodSplatCount`) and controller thresholds.
- R4: the full ablation table on both headsets (PLAN benchmark 6).
- Later, for the demo walk: smooth locomotion with snap turn in VR outside bench mode.

**coverage**
- Finish: Muse Image repair client (`coverage/repair.py`, API facts in section 7), finished
  overlay, merge main, PR.
- C2 Recapture loop: show the next-best viewpoint in the viewer, register the top-up video
  with MapAnything together with the original frames, gsplat fine-tune, re-score.
- C3 Repair: render gap views, Muse Image edit with nearest real frames as references,
  distil back, flag `generated`, "show generated" toggle in the viewer.

## 6. Running lanes locally

- One Claude Code session per lane, each in its own worktree:
  `git worktree add ../mr-<lane> -b lane/<lane> origin/main` (or `origin/lane/<lane>` to
  continue an existing branch). Never share a working folder between sessions.
- Agents in the repo: `explorer` (Haiku, low effort) for searches, `reviewer` (high effort)
  once per PR. `.claude/settings.json` sets compaction at 400k tokens. Lane sessions run at
  the default effort.
- Commit small and push at least every 30 minutes (protects work if a usage limit is hit).
- Merge order and conflicts: lanes append to `docs/BUILD_LOG.md` and the Commands block in
  `AGENTS.md`, so those conflict; keep both sides. Lane installs in `web/src/main.ts` run in
  this order: planes, lod, coverage, perf (perf wraps lod's `meshOptions`).
- Before merging a PR: run `uv run pytest`, `uv run ruff check .`, `uv run ruff format
  --check .`, `npm test`, `npm run build`, and open the viewer once.

### Starter prompts for local sessions
- Core: "Read docs/HANDOFF.md and AGENTS.md. You own the core lane. Do M2 from section 5
  (lift_ids with labels.json, support rule, floor alignment) against the demo scene first,
  then on room01 once M1 has produced it. Branch lane/core, PR when done."
- Coverage: "Read docs/HANDOFF.md and AGENTS.md. Continue the coverage lane from
  origin/lane/coverage: finish the repair client and overlay, merge main, open the PR."
- LOD: "Read docs/HANDOFF.md and AGENTS.md. Do L1 and L2 from section 5 on branch lane/lod."
- Bake: "Read docs/HANDOFF.md and AGENTS.md. Do B1 from section 5 on branch lane/bake."

## 7. Facts learned on 29 Oct

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
- In cloud sessions `gh pr create` fails (GraphQL blocked) and `gh api .../pulls` works;
  locally `gh pr create` is fine.
