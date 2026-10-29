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
