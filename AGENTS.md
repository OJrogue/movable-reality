# AGENTS.md: Movable Reality

Instructions for coding agents working in this repo (Muse Code, Claude Code, others).
Read this whole file before your first change. Longer context lives in `docs/`; open the
file a task points you to instead of guessing.

## Project in one paragraph

Movable Reality turns a ~30 s phone video of a room into a metric 3D Gaussian Splatting
(3DGS) scene in which every object is a separate, movable splat. It finds the parts of the
room the video did not cover and fills them, either with a short top-up video or with Meta
generative models, and tags what was generated. It renders the whole room on a standalone
VR headset at a locked 72 FPS, and the user rearranges it by voice. It is an entry in the
Meta Global AI Developer Hackathon (Indie Developer track, theme "Push the Frontier").
Team "Gaussian Movers", solo developer OJ.

## Hard constraints (breaking these can disqualify the entry)

1. **New work only.** Everything in this repo must be written during the event
   (26 Oct to 4 Nov 2026). Do not copy code from other projects, including OJ's own
   earlier renderers (for example the PGR WebGPU renderer). Third-party open-source code is
   used as a dependency, never pasted in.
2. **Every model called at runtime must be a Meta model.** Cerebral Valley ruled on
   28 Oct that any model the project calls at runtime (pipeline, server, viewer, agent)
   must be a Meta model; non-Meta models are not allowed alongside them. Meta open-source
   models count however they are served: SAM 3.1, MapAnything, SAM 3D Objects, Muse Spark
   Open, Muse Glimmer, Llama. Libraries and algorithms without learned weights (gsplat,
   Spark, three.js, numpy, least-squares uncertainty) are fine. Examples of what is NOT
   allowed at runtime: Difix3D+, Stable Diffusion, CLIP, Gemini, GPT, Claude. If you are
   unsure whether something is a model or who made it, stop and ask OJ. Coding tools are
   unrestricted. All model IDs live in `src/movable_reality/config.py` so usage is easy
   to show.
3. **Public repo, licences respected.** Only add dependencies and weights whose licence
   allows this use. Use `facebook/map-anything-apache` (Apache-2.0), never the CC-BY-NC
   `facebook/map-anything`. Never commit model weights, datasets, captured video or `.env`.
4. **Deadline.** Submission closes Wed 4 Nov 21:00 ET = **Thu 5 Nov 03:00 Prague time**.
   Suggested feature freeze: **Mon 2 Nov 20:00 Prague** (OJ's call, see `docs/PLAN.md`).
   After the freeze: bug fixes, benchmarks, docs and demo polish only.
5. **Log the work.** At the end of every work session append an entry to
   `docs/BUILD_LOG.md` (date, lane, what changed, what is next). Judges must be able to see
   what was built during the event, and the log feeds the README and the demo video.
6. **Honest attribution.** Keep the AI co-author trailers your tool adds to commits. Never
   rewrite history or backdate commits.

## What we are building

```
phone video ─► frames (3-5 fps, sharpest) ─┬─► MapAnything on Modal ─► COLMAP (metric) ─► gsplat on Modal ─► scene.ply
                                           └─► sam_input.mp4 (same frames, same order)
Muse Wonder watches the video ─► object inventory (nouns) ─► SAM 3.1 video tracking per noun ─► instance masks
masks + cameras + rendered depth ─► per-Gaussian object ID vote ─► background.ply + obj_<id>.ply + scene.json
SAM 3D Objects completes each object from its best view ─► obj_<id>_completed.ply

coverage loop: score coverage (view counts, angle spread, residual uncertainty) ─► gap map in scene.json
  large gap, user in the room ─► next-best viewpoint ─► 10 s top-up video ─► MapAnything re-registers ─► gsplat fine-tune
  small gap or object back ─► render nearby views ─► Muse Image edit (real frames as references) ─► distil back ─► tag as generated
  ─► re-score until no gaps are left

for the headset: floor + ceiling ─► plane fit ─► baked opaque textured quads (glTF)
                 each object ─► Spark build-lod ─► one .RAD LOD tree per object
viewer, every frame: opaque planes ─► joint LOD cut across all objects (targeted object first)
                     ─► Spark sort + blend ─► budget controller keeps p99 frame time <= 13.9 ms

voice ─► Muse Voice Transcribe ─► Muse Wonder agent (tool calls) ─► edits scene.json ─► viewer updates
viewer screenshot ─► Muse Spark Open on Baseten (verifier) ─► accept, or agent retries
"add a sofa" ─► Muse Image ─► SAM 3.1 mask ─► SAM 3D Objects ─► new object placed on the floor
```

Rationale, stage specs and algorithms: `docs/ARCHITECTURE.md`.
Current milestones and task order: `docs/PLAN.md`. Rules and submission checklist:
`docs/HACKATHON.md`.

## Meta models and where they are used

| Model | ID (see config.py) | Access | Job in this project |
|---|---|---|---|
| Muse Wonder | `muse-wonder-1.0` | Meta Model API | Agent brain (tool calls), object inventory from the video, naming gaps |
| Muse Spark | `muse-spark-1.3` | Meta Model API | Cheap helper calls (labels, short checks) |
| SAM 3.1 | `sam-3.1` | Meta Model API (Responses) | Video segmentation and tracking of every object |
| Muse Image | `muse-image-1.0` | Meta Model API | Furniture to insert; repairing rendered views of coverage gaps |
| Muse Voice Transcribe | `muse-voice-transcribe-1.0` | Meta Model API | Voice commands |
| Muse Spark Open | `meta-models/muse-spark-open-1026` | Baseten (open weights) | Visual verifier after each edit |
| MapAnything | `facebook/map-anything-apache` | Open weights on Modal | Metric cameras and geometry; registering top-up videos |
| SAM 3D Objects | gated weights on Hugging Face (check the repo README for the exact ID) | Open weights on Modal | Completing objects and their unseen backs |

Meta Model API: base URL `https://api.meta.ai/v1`, OpenAI-compatible, key in
`MODEL_API_KEY`. Use Standard-mode IDs only (no `-contributor` suffix).
Baseten: base URL `https://inference.baseten.co/v1`, key in `BASETEN_API_KEY`.
Docs: https://dev.meta.ai/docs (SAM: `/docs/sam/overview`, video input:
`/docs/video-understanding`, tool calling: `/docs/tool-calling`, images:
`/v1/images/edits` for Muse Image editing).

Open-source components without learned models: gsplat (Apache-2.0), Spark renderer
(MIT), three.js (MIT), meta-sam-parser.

## Repo layout and lane ownership

Work runs in parallel lanes, usually one agent session per lane on its own branch
(`lane/<name>`), merged through pull requests. Edit only the paths your lane owns. Shared
files take small, additive changes only (see below).

```
AGENTS.md               this file (CLAUDE.md imports it)
docs/                   ARCHITECTURE, PLAN, HACKATHON, BUILD_LOG
src/movable_reality/    Python package
  config.py, clients.py, sam.py, scene.py      core (shared)
  lod/                  lane: lod       per-object PLY split, .RAD build helpers
  coverage/             lane: coverage  scoring, gap map, next-best view, repair, tagging
  bake/                 lane: bake      plane fit, orthographic splat bake, glTF export
scripts/                CLI entry points; name them after the lane (build_lod.py, score_coverage.py, bake_planes.py)
modal_jobs/             GPU jobs on Modal; one file per job, named after the lane
tests/                  CPU-only unit tests, no paid API calls; test_<lane>_*.py
web/                    Vite + three.js + Spark viewer (desktop and WebXR)
  src/main.ts, src/perf/, src/bench/   lane: renderer   HUD, frame-time log, budget controller, synthetic scenes, walk paths
  src/lod/              lane: lod       loading per-object .RAD trees, priority
  src/coverage/         lane: coverage  gap overlay, "show generated" toggle, recapture prompt
  src/planes/           lane: bake      loading baked planes, opaque pass
  src/scene.ts          core (shared)
data/                   gitignored: raw videos, scenes, API cache
benchmarks/             committed: raw frame-time logs (JSON) and result tables
```

Lanes: **renderer** (R0-R4), **lod**, **coverage** (C1-C3), **bake**, and **core** (M1-M5:
reconstruction, objects, agent, server, voice). The core lane owns `scripts/`,
`modal_jobs/reconstruct.py`, `src/movable_reality/{sam,clients}.py` and the future
`agent/` and `server.py`.

Shared files and how to change them:
- `scene.py` / `scene.ts`: each lane owns its own section (`lod`, `coverage`, `planes`).
  Change your section in both files and their tests in the same commit; do not touch
  other sections.
- `web/src/main.ts`: owned by the renderer lane. Other lanes export an `install...()`
  function from their folder and add one call to `main.ts` in their PR.
- `pyproject.toml` / `uv.lock`, `web/package.json` / lockfile: add a dependency only when
  needed, say why in the PR, and regenerate the lockfile after rebasing.
- `AGENTS.md` and `docs/`: append to your lane's lines; do not reorganise.

## Commands

```bash
uv sync                                        # Python 3.12 env + deps
uv run python scripts/smoke_meta.py            # Meta Model API check
uv run python scripts/smoke_meta.py --image data/raw/room.jpg --noun chair   # + SAM 3.1
uv run python scripts/smoke_baseten.py         # Muse Spark Open on Baseten
uv run python scripts/make_demo_scene.py       # synthetic scene "demo" for agent/viewer work
uv run python scripts/edit_scene.py demo move "chair#1" 0.5 0 0   # manual edit (agent tools do the same)
uv run python scripts/extract_frames.py data/raw/room01.mp4 --scene room01
uv run modal run modal_jobs/reconstruct.py --scene room01
uv run pytest                                  # unit tests (no paid calls)
uv run ruff check . && uv run ruff format .
cd web && npm install && npm run dev           # viewer, http://localhost:5173/?scene=room01
cd web && npm test                             # viewer unit tests (vitest), pure logic only
uv run python scripts/bake_planes.py --scene bakedemo --synthetic   # bake: synthetic room, then bake floor + ceiling
uv run python scripts/bake_planes.py --scene room01   # bake: planes/<label>.glb + reduced background (--force, --labels floor, --revert)
uv run modal run modal_jobs/bake.py --scene room01    # bake: same stage, gsplat ortho rasterizer on an L4 (not yet run)
# bake depth check (dev server): http://localhost:5173/src/planes/depthcheck.html?case=behind|front|both|crossing
```

Lanes add their own commands here in their PR.

## The scene contract (`data/scenes/<id>/scene.json`)

The pipeline writes it, the agent edits it, the viewer renders it. The schema is
`src/movable_reality/scene.py`; the viewer mirrors it in `web/src/scene.ts`. A schema
change updates both files and their tests in the same commit.

- Units are metres. World frame: +Y up, floor at y = 0.
- MapAnything and COLMAP use the OpenCV camera convention (+Y down, +Z forward). Convert
  once through `world_from_recon` in the manifest, nowhere else.
- Object edits are deltas in `transform`, applied about the object's bbox centre.
- Objects are never deleted from the manifest. Removal sets `visible: false`, so every
  edit is undoable, and every edit is appended to `history`.
- Optional sections, each owned by one lane: `SceneObject.lod` and `background_lod`
  (lod), `coverage` (coverage), `planes` (bake). A viewer or stage that finds a section
  missing must fall back to the plain PLY path.
- Generated splats are flagged per splat (PLY property `generated`, 0 or 1) and counted
  in the manifest, so the viewer can show or hide them and measurements can ignore them.

## Renderer targets

- Main device: Steam Frame (Snapdragon 8 Gen 3), WebXR through the community Chromium XR
  build. Second device: Pico 4 (Snapdragon XR2 Gen 1) in Pico Browser. Optional: Quest 2
  in the Meta Quest Browser.
- Budget: 72 Hz, so p99 frame time at or under 13.9 ms over a 10 min scripted walk, with
  under 1% of frames over budget and at most a 5% FPS drop between the first and last
  minute.
- Standalone only: never claim numbers measured with PC streaming.
- WebXR exposes no eye gaze; priority uses the controller ray and head direction.
- Log raw frame times as JSON under `benchmarks/`; never report a number without its log.

## Conventions

- Python 3.12, uv, type hints, `pathlib`, pydantic for data that crosses process
  boundaries, ruff (line length 100). Keep geometry as pure functions so it is testable
  on CPU.
- Pipeline stages are idempotent scripts: skip work whose output exists unless `--force`.
  Each stage reads and writes only inside `data/scenes/<id>/`.
- TypeScript strict, ES modules, Vite, three.js + `@sparkjsdev/spark`. No UI framework
  unless it clearly pays off.
- Small commits, one concern each, imperative messages with the lane as scope
  (`feat(coverage): view-count gap map`). Push often: commit history is evidence of
  in-event work.

## Cost and secrets

- Keys live in environment variables or `.env` (template: `.env.example`). Never print,
  log or commit them, and never paste them into prompts or files that agents read.
- Cache every paid API response under `data/cache/` keyed by a hash of the request;
  reruns must hit the cache. `sam.py` shows the pattern.
- Ask OJ before a single run expected to cost more than $20, GPUs bigger than one
  H100, or training runs longer than 1 h.

## Known pitfalls

- gsplat `simple_trainer.py` normalises world space by default, so its PLY is not
  metric. Pass `--no-normalize-world-space` (tyro bool flag; `--normalize_world_space
  False` is rejected), or invert `parser.transform`.
- MapAnything's COLMAP export saves images at model input resolution (~518 px). For final
  quality, train on full-resolution frames with rescaled intrinsics.
- SAM 3.1 masks are box-local rasters. Paste them into the frame with `sam.paste_mask`.
  One noun per request, max 16 objects per frame, max 15,000 frames, MP4/MOV with
  H.264/H.265 only. Zero matches is a valid (empty) result.
- SAM frame index `i` must equal `frames/{i:06d}.jpg`. Always build the SAM clip from the
  frames chosen for reconstruction (`extract_frames.py` does this).
- Muse Code loads this file only after the workspace is trusted. Do not run `muse init`
  in this repo: it writes its own `AGENTS.md` and may replace this one.
- The Steam Frame's built-in browser has no WebXR; use the Chromium XR build. WebXR
  needs a secure context (HTTPS, or `localhost` through `adb reverse`).

## Definition of done for a task

Works end to end on at least one real scene (or, before real scenes exist, on the
synthetic `demo` scene with a note in the build log); `uv run pytest` passes; ruff is
clean; if `web/` changed, `npm test` and `npm run build` pass; `docs/BUILD_LOG.md` updated;
`git status` shows no secrets, videos or `.ply` files staged.

## When unsure

Choose the simplest thing that makes the demo work. Do not invent API parameters: check
the official docs listed in `docs/ARCHITECTURE.md`. If a decision changes the demo, the
schema, the model list or the budget, stop and ask OJ.
