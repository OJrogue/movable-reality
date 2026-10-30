# Per-object LOD with Spark 2.4.0 (lane: lod)

How Spark's level-of-detail system behaves in practice, what this lane builds on top of it,
and how to precompute `.RAD` trees. Read 29 Oct 2026 from the Spark 2.4.0 sources: the npm
package in `web/node_modules/@sparkjsdev/spark/dist/` and the GitHub repository at tag
`v2.4.0` (commit `ffe210d`, https://github.com/sparkjsdev/spark). Line numbers refer to
`dist/spark.module.js` of the pinned npm package unless a Rust path is given.

## Short answers

| Question | Answer |
|---|---|
| Can we precompute `.RAD` files? | **Yes.** `build-lod` compiles from Spark's repository (tag v2.4.0) with `cargo` on CPU in about 1.5 min; `scripts/build_lod.py --setup` does it and then builds one tree per object. |
| Load-time vs precomputed | `lod: true` runs tiny-lod in a worker on every load; a `.RAD` already has the tree, so the worker only decodes it. Both end up as the same `lodSplats` + `initLodTree`. |
| Several meshes, one budget | Yes. One traversal over all LOD meshes per pass, with a single priority queue and a single `maxSplats`. |
| What `lodScale` does | It multiplies that mesh's screen-size score in the shared queue, so with a fixed budget the mesh takes splats from the others. |
| Does moving a mesh rebuild anything? | No tree rebuild, no re-decode, no `initLodTree`. A move only feeds a new `viewToObject` matrix to the next traversal. Spark does not schedule that traversal on its own, so this lane requests one (see below). |
| Paging | `paged: true` streams `.RAD` chunks into a shared GPU page pool (16M splats on desktop). It needs HTTP Range or `--rad-chunked` files. Off by default here. |
| Licence | Spark, `build-lod` and `spark-lib` are MIT (World Labs Technologies). We run the tool from a cached clone and copy no Spark code into this repo. |

## How Spark's LOD works

### Two ways to get a tree

1. **Load-time (`SplatMesh({ url: "x.ply", lod: true })`).** The worker `loadPackedSplats`
   decodes the PLY into a `CsplatArray` and builds the tree: `tiny_lod` (base 1.5) for
   `lod: true`, `bhatt_lod` (base 1.25 in the browser) for `lod: "quality"`. The result
   comes back as `lodSplats` (worker source embedded at line 2942: `loadPackedSplats`; JS
   side `PackedSplats.asyncInitialize` line 7954 and `SplatLoader.loadInternal` line 11053).
   `lodAbove: n` skips the build below n splats. `nonLod: true` keeps the original splats as
   well, and `SplatMesh.enableLod` then switches between the two. The console logs
   `Tiny LoD: 4000 -> 5896 (50 ms)`, which is how we saw the build in the headless check.
2. **Precomputed `.RAD` (`SplatMesh({ url: "x.rad" })`).** The file is detected by
   extension or by its `RAD0` magic (`getSplatFileType`, line 11146). The decoder marks the
   encoding `lodOpacity` because the file has a tree, and `loadPackedSplats` returns it as
   `lodSplats` without building anything. The `lod` flag is unnecessary: if it is set,
   `has_lod()` short-circuits the build. The docs say the same
   (`docs/docs/lod-getting-started.md`: "the `lod: true` flag is unnecessary because the
   .RAD file encodes that information").

Either way the next `driveLod` pass finds a mesh with `packedSplats.lodSplats`, queues it,
and `initLodTree` copies its tree into the LOD worker once (line 12795). From that point
the two paths behave the same at runtime.

What the precomputed path buys: no tiny-lod on the headset's CPU (the docs give 1-3 s per
1M splats; a room is several million), the higher-quality bhatt-lod tree by default, and
smaller downloads (4000-splat demo object: PLY 272 KB, `.RAD` 103 KB, see the table below).

### One budget across all meshes

Every frame `SparkRenderer.driveLod` (line 12654):

- collects the visible `SplatMesh`es that have `lodSplats` (or are paged) and do not have
  `enableLod === false`;
- sets `maxSplats = (lodSplatCount ?? platform default) * lodSplatScale`. The platform
  default (`defaultSplatTarget`, line 12651) is 500K when the user agent has "Oculus"
  and WebXR exists, 750K on Vision Pro, 1M on Android, 1.5M on iOS and 2.5M otherwise.
  Only the Meta Quest Browser matches the first rule; the Chromium XR build on the Steam
  Frame and Pico Browser fall to 1M or 2.5M depending on their user-agent strings (not
  checked on the devices). **The renderer lane should set `lodSplatCount` per device**;
- sets `pixelScaleLimit` from the projection matrix, render height and `lodRenderScale`;
- calls the worker's `traverseLodTrees` once with one instance per mesh (line 12815).

The traversal (`rust/spark-rs/src/lod_tree.rs`, `traverse_lod_trees`) puts the root of
every instance into **one** max-priority queue keyed by projected size, then keeps
splitting the largest node of any instance until the next split would exceed `maxSplats`
or every node is below `pixelScaleLimit`. The result is a joint cut that "maximizes the
minimum screen-space splat size" across all objects (`docs/docs/new-features-2.0.md`). The
headless check shows it: with `lodSplatCount=20000` the four demo meshes always sum to
exactly 20,000 drawn splats, and the split between them moves when priority changes.

### What `lodScale` does

`compute_pixel_scale` in `lod_tree.rs` scores a node as
`size / distance * lodScale * foveate`. Raising one mesh's `lodScale` makes all of its
nodes look larger in the shared queue, so they are split first. With a fixed budget the
extra detail comes **from the other meshes**. Spark's docs: "2.0 results in 2x finer
detail for this object, while 0.5 will result in 2x coarser splats". Measured on the demo
at a 20,000 budget (`rad`, `drawn` splats per mesh):

| State | background | chair | table | lamp |
|---|---|---|---|---|
| loaded, all `lodScale` 1 | 14,364 | 2,552 | 2,321 | 763 |
| mouse on table, table ramps to 2 | 12,900 | 2,417 | **3,984** | 699 |

Per-mesh foveation overrides exist too (`behindFoveate`, `coneFov0`, `coneFov`,
`coneFoveate`), with the renderer's values as defaults (90 deg / 120 deg / 0.4 / 0.2).

### Moving a mesh: no rebuild

The tree is stored once per `PackedSplats` in the worker (`lodIds`, keyed by the splats
object, not the mesh). Each traversal gets `viewToObject = mesh.matrixWorld^-1 *
cameraToWorld` per instance (line 12820), so a moved, rotated or scaled object is simply
traversed from a different relative viewpoint. Nothing in the move path calls
`initLodTree`, `createLodSplats` or a decoder.

There is one catch. `driveLod` re-traverses only when `lodDirty` is set, which happens
when the camera moves (more than a small threshold), when `maxSplats` or the pixel limit
changes, when the set of LOD meshes changes, or when a mesh's `version` rises **by more
than one** since the last pass (line 12685: it stores `version + 1`). A transform change
bumps `version` once (`SplatMesh.update`, line 10857). A `lodScale` change does not bump
it at all; `lodScale` is only read when a traversal runs. So with a still camera:

- a one-step move (an agent edit) keeps the old cut until the camera moves. The splats
  still render at the new place because the transform is applied in the shader; only the
  choice of detail is stale.
- a `lodScale` change has no effect until something else dirties the LOD.

`installLod` therefore sets `spark.lodDirty = true` after each manifest change and while a
`lodScale` ramp is running. This costs one traversal, not a rebuild. Headless check, demo
with `.RAD`, worker calls counted by wrapping `Worker.prototype.postMessage`:

| Phase | `initLodTree` | `loadPackedSplats` | `traverseLodTrees` | `disposeLodTree` |
|---|---|---|---|---|
| after load | 4 | 4 | 9 | 0 |
| after `edit_scene.py move chair#1 0.5 0 0` | 4 | 4 | 29 | 0 |
| after a second move and `rotate table#1 90` | 4 | 4 | 39 | 0 |
| with `?lodRetraverse=0` (nudge off), same edits | 4 | 4 | 8 (unchanged) | 0 |

The load-time path behaves the same: four `Tiny LoD` builds at load (console log), none
after edits, `initLodTree` stays at 4.

**Hiding** is different. `lodCleanupTimeoutMs` (default 3000 ms, line 12296) releases the
worker tree of a mesh that has not been rendered for 3 s (`cleanupLodTrees`, line 12901),
and showing it again runs `initLodTree` again. The check confirms it: `remove "floor
lamp#1"`, wait 5 s, `restore` → `disposeLodTree` 1, `initLodTree` 5. That copies the tree
from main-thread memory into the worker; it does not re-decode or rebuild (for a paged
mesh it refetches pages). Since 30 Oct (decision D7) `installLod` sets
`spark.lodCleanupTimeoutMs = Infinity`, so hidden objects keep their worker trees and
remove/restore needs no `initLodTree`.

### Completed objects (`completed_lod`)

SAM 3D Objects writes `obj_<id>_completed.ply` (stage 6) with the side the video never saw.
`build_lod.py` builds `lod/obj_<id>_completed.rad` for it and records it as
`SceneObject.completed_lod`. The viewer (`shownPly` and `SceneView.swap` in
`web/src/scene.ts`) shows the scan until the object's transform differs from identity, then
the completed version; an undo back to identity shows the scan again. Each variant is
created once and hidden when unused, so swapping is instant after the first time and an
object holds at most two trees. Disposing the unused variant would leave its tree in the
worker for good, because D7 turns Spark's cleanup off. `lodChoice` picks `completed_lod`
for the completed PLY and falls back to a load-time tree when it is missing. A completed
PLY that fails to load is remembered and the scan stays (unit tests in
`web/src/scene.swap.test.ts`).

The new variant is shown and the old one hidden once `SplatMesh.initialized` resolves.
Spark copies the tree into its worker (`initLodTree`) on the next LOD pass after that, so
the object can be missing for a frame or two at the first swap (not observed on `demo`;
a paged mesh would show the gap longest).

Checked 30 Oct on a copy of `demo` with a red-tinted copy of the chair as its completed
PLY: after `move chair#1 0.3 0 0` the chair turns red and loads
`lod/obj_chair_1_completed.rad`, the undo shows the blue scan, a later `rotate` shows red
again. Spark held 5 trees (4 meshes + 1 completed) and traversed 4 meshes throughout, so
the hidden variant takes no budget and no tree was dropped or rebuilt.

### Paging and streaming

`SplatMesh({ url: "x.rad", paged: true })` creates a `PagedSplats` (line 8659). It reads
the header with HTTP Range requests (64 KB first), then fetches chunks of 65,536 splats as
the traversal touches them, into one shared `SplatPager` pool (`maxPagedSplats`, 16M
splats on desktop and 8M on other mobile). The nearest meshes' root chunks come first
(line 12866). Two file layouts work:

- a single `.rad`: header and chunks are fetched with `Range: bytes=...`. The Vite dev
  server in `web/vite.config.ts` answers Range with 206 since 29 Oct (before that it sent
  the whole file, which breaks files of more than one chunk). A host for the demo must
  honour Range too (most static hosts do).
- `build-lod --rad-chunked`: a small `x-lod.rad` header plus `x-lod-0.radc`, `-1.radc`,
  ... each fetched as a whole file. This works on any static host. Checked headless on a
  copy of the demo: all four meshes streamed and drew a 20,000-splat cut.

For one room (a few million splats in total) everything fits in GPU memory, so paging
buys faster first frames, not capacity. `installLod` leaves it off and turns it on with
`?lodPaged=1`; `scripts/build_lod.py` writes single-file `.rad`.

### Other relevant knobs (owned by the renderer lane)

`lodRenderScale` (minimum splat size in pixels, 1 to 5), `lodTraverseMode` (`"standard"`
default, `"dynamic"` experimental and faster, per the 2.2.0 CHANGELOG), `lodInflate`,
`coneFov0`/`coneFov`/`coneFoveate`/`behindFoveate`, `lodRaycast` (Spark keeps a coarse
cut, 10K-25K splats, for raycasting against LOD meshes every 500 ms).

## `build-lod`: can it run from this repo?

Yes, as an external tool. The npm package only ships `dist/`, but its `package.json`
still lists `"build-lod": "cargo run --manifest-path rust/build-lod/Cargo.toml --release
--"`, and the crate is in the repository (`rust/build-lod`, 655 + 416 lines; depends on
`rust/spark-lib`; MIT). What we checked on 29 Oct (2 vCPU, 7 GB RAM, no GPU):

- `cargo build --release --locked -p build-lod --no-default-features` in `rust/`: 1 min
  28 s, 2.5 MB binary. `--no-default-features` drops the `gpu` feature (wgpu), which is
  only used for the optional `--cluster-sh` SH codebook; everything else is CPU.
- It reads `.ply` (with SH up to degree 3 and extra properties such as our `generated`
  uchar, which it ignores), `.spz`, `.splat`, `.ksplat`, `.sog`, `.rad`.
- Default method `--quality` = bhatt-lod with base 1.75 (the docs text says the default is
  `--quick`, but the 2.4.0 source sets `BuildLodMethod::Quality` as default).
- Output `<input stem>-lod.rad` next to the input. `--rad-chunked` writes header plus
  `.radc` chunks. `--max-sh=N` clamps SH.
- Failure modes: an empty PLY panics (exit 101); NaN splats print "Stopping processing due
  to invalid splats" and **exit 0 without writing a file**. `scripts/build_lod.py` checks
  that the output exists and skips empty PLYs.

Measured times:

| Input | Method | Wall time | Output |
|---|---|---|---|
| demo object, 4,000 splats, SH 0 | quality | 0.03 s | 103 KB, 7,486 nodes |
| demo background, 24,000 splats | quality | 0.15 s | 165 KB, 46,964 nodes |
| synthetic 1M splats, SH 3 (236 MB PLY) | quality | 27.6 s | 68 MB, 1.46M nodes |
| synthetic 1M splats, SH 3 | quick | 8.8 s | 74 MB, 1.59M nodes |

Wall time includes reading the PLY and writing the file; the tree build itself
(`lod_duration` in the `.RAD` comment) was 22.0 s and 3.6 s.

Extrapolating linearly (not measured), a room of 3-5M splats split into ~20 objects should
take 1.5-3 min with `--quality` on 2 cores, less on a laptop. It belongs in the pipeline
after the split (stage 5) and after object completion (stage 6), not on the headset.

### Using it

```bash
uv run python scripts/build_lod.py --setup         # clone Spark v2.4.0 into ~/.cache/movable-reality/tools, cargo build
uv run python scripts/build_lod.py demo            # lod/*.rad + scene.json lod fields
uv run python scripts/build_lod.py demo --force    # rebuild all
uv run python scripts/build_lod.py demo --method quick --max-sh 1
```

The binary is found in this order: `$SPARK_BUILD_LOD`, the cache
`~/.cache/movable-reality/tools/spark-v<version>/rust/target/release/build-lod`, then
`build-lod` on `PATH`. The version comes from `web/package.json`, so the tool and the
viewer's decoder always match. A tree is skipped when it exists and is newer than its PLY;
`--force` rebuilds. The manifest is re-read before writing, so agent edits made during a
build survive. On Modal the same script runs on a CPU container once the image has Rust.

## What the viewer does (`web/src/lod/`)

- `options.ts`: per-mesh options through `ctx.meshOptions`. `.RAD` when `SceneObject.lod`
  / `Scene.background_lod` is set, else the PLY with `lod: true` (`?lodBuild=quality` for
  bhatt-lod at load time). Older manifests without these fields fall back automatically.
- `?mode=naive`: `installLod` returns early; meshes are plain PLYs with no LOD.
- `pick.ts`: the ray is tested against each object's manifest bbox in its pivot frame, so
  yaw and scale edits are exact. Desktop uses the mouse; XR uses tracked controller target
  rays (rig frame), falling back to head direction when no controller has a pose. Splats
  are never raycast.
- `priority.ts`: target `lodScale` 2.0 (`?lodPriority=`) for the hovered object and the
  object of the latest history entry (if visible), 1.0 for the rest; exponential ramp with
  95% after 300 ms (`?lodRamp=`). Background stays at 1.0.
- `index.ts`: wires it together, sets `spark.lodDirty` when needed, logs every change as
  `[lod] target|reached|source|dirty ...` and exposes `window.mrLod` (`stats()` returns
  trees, budget, drawn splats per mesh). `?lodSplatCount=` is a test knob for the shared
  budget; the budget controller (renderer lane) owns it at runtime.

## Open points

- Device numbers: none yet. The headless check uses SwiftShader at 1-2 frames per second
  after load, so it says nothing about frame time.
- The platform budget default only recognises the Quest browser as an XR device.
- Priority strength: 2.0 is a guess; tune on the headset with the budget controller.
- Memory with `lodCleanupTimeoutMs = Infinity`: every object tree stays in the worker for
  the whole session, two for completed objects. Fine for one room; revisit for big scenes.
- With cleanup off, a disposed mesh keeps its worker tree. The only case today is the
  background swap after a live bake (`SceneView.apply`), which leaves the old background
  tree in the worker until reload. `cleanupLodTrees` is Spark's only path to
  `disposeLodTree`; a fix would lower the timeout for one pass around that swap.
- `mrLod.sources` records the last variant created per object, not the one shown.
