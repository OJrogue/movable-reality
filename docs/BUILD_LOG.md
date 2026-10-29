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
