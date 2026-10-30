# Architecture

## Goal and frontier claim

Text-driven editing of Gaussian splat scenes exists in research (for example TransSplat,
SIMSplat), so editing alone is not the claim. The claim is the whole chain on consumer
input and consumer hardware: **a 30 s phone video becomes a metric, object-separated,
agent-editable 3DGS twin; gaps in the capture are found and filled (recaptured or
generated, generated parts tagged); and the whole room renders on a standalone VR headset
at 72 FPS** while the agent edits it and checks its own edits visually.

Context for the rendering claim: Meta's Spatial SDK renders one splat at a time and advises
staying under 150k splats on Quest; Hyperscape streams from the cloud. SAM 3.1 labels
decide how the headset spends its splat budget: flat structure becomes opaque geometry,
every object gets its own LOD tree, and the object being targeted gets detail first.
The benchmarks in `docs/PLAN.md` back this up.

Rule that shapes every choice: every model called at runtime is a Meta model
(see `docs/HACKATHON.md`).

## Scene folder layout

Every stage reads and writes inside one folder:

```
data/scenes/<scene_id>/
  input.mp4                  original capture (copied by extract_frames.py)
  frames/000000.jpg ...      selected sharp frames, full resolution (long side <= 1920)
  frames.json                source timestamp of every selected frame
  sam_input.mp4              H.264 clip built from frames/, frame i == frames/{i:06d}.jpg
  recon/                     MapAnything COLMAP export: images/, sparse/{cameras,images,points3D}.bin
  gsplat/                    gsplat result_dir (ckpts/, ply/, stats/)
  splat/scene.ply            trained full scene, metric, recon frame
  inventory.json             Muse Wonder object list from the video
  sam/<noun>.json            parsed SAM 3.1 instances per noun (boxes, masks, frame idx)
  ids/gaussian_ids.npy       per-Gaussian object id (-1 = background)
  objects/background.ply     splats not assigned to any object
  objects/obj_<id>.ply       one file per object
  objects/obj_<id>_completed.ply   SAM 3D Objects completion, registered
  lod/background.rad         Spark LOD tree for the background (lane: lod)
  lod/obj_<id>.rad           one Spark LOD tree per object (lane: lod)
  planes/<label>.glb         baked opaque plane with its texture, e.g. floor, ceiling (lane: bake)
  coverage/                  scores, gap map, next-best views, repaired views (lane: coverage)
  topups/<n>/                top-up videos and their frames (lane: coverage)
  scene.json                 manifest read by agent and viewer
```

## Stages

### 1. Frame selection (`scripts/extract_frames.py`, CPU)
Sample candidates at 4x the target rate, keep the sharpest frame (variance of Laplacian)
in each window. Default target 4 fps, max 200 frames. Writes `frames/`, `frames.json`
and `sam_input.mp4` from exactly those frames so SAM and reconstruction share indices.

### 2. Reconstruction (`modal_jobs/reconstruct.py`, GPU)
- MapAnything (`scripts/demo_colmap.py --apache`) predicts metric cameras and points
  feed-forward and writes a COLMAP model. Replaces COLMAP SfM, gives metric scale.
- gsplat `examples/simple_trainer.py default` trains on that model.
  Flags: `--no-normalize-world-space` (keep metric frame), `--save_ply`,
  `--disable_viewer`, `--max_steps 7000..15000`, `--data_factor 1`.
- Improvement (day 2): swap the ~518 px images for full-resolution frames and rescale
  intrinsics in `cameras.bin` (pycolmap), then train.
- Baseline for benchmarks: COLMAP or GLOMAP poses on the same frames.

### 3. Inventory (`scripts/inventory.py`, Muse Wonder)
Upload `sam_input.mp4`, ask Muse Wonder (video input, structured JSON output) for movable
objects as short concrete nouns that SAM can match ("armchair", "floor lamp"), plus
"floor" and "wall" for alignment. Cache the result.

### 4. Segmentation (`src/movable_reality/sam.py`, SAM 3.1)
One request per noun on the uploaded clip (`file_id`, `mask_encoding: one_bit`).
Instance ids are stable across frames within one request; the global object id is
`<noun>#<sam_object_id>`. Masks are box-local: `paste_mask()` puts them into the frame.

### 5. Lifting masks to Gaussians (`scripts/lift_ids.py`, GPU or CPU)
For each training view: project every Gaussian centre with the camera; skip it if
outside the image or occluded (compare its camera depth to the rendered expected depth
from gsplat, tolerance ~3% of depth); otherwise add a vote, weighted by opacity, for the
object id under its pixel (or background). Assign the argmax when it wins at least
~60% of the votes and was seen in at least 3 views; else background. Then remove
floaters per object with a radius outlier filter. Write per-object PLYs and the
manifest. The floor plane comes from RANSAC on Gaussians voted "floor" and defines
`world_from_recon` (up = floor normal, floor at y = 0).

### 6. Object completion (`modal_jobs/complete_objects.py`, GPU)
Pick each object's best view (largest unoccluded mask), run SAM 3D Objects on
image + mask, get Gaussians, register to the scanned partial object (scale from bbox,
then ICP on centres), save `obj_<id>_completed.ply`. The viewer swaps to the completed
version when the object is moved, so unseen sides look whole.

### 7. Agent (`src/movable_reality/agent/`, Muse Wonder + tool calling)
Tools act on `scene.json`, never on PLYs:

| Tool | Effect |
|---|---|
| `list_objects()` | ids, labels, bbox sizes in metres, positions |
| `move_object(id, dx, dy, dz)` / `place_object(id, x, z)` | translate, snapped to floor |
| `rotate_object(id, yaw_deg)` | rotate about bbox centre |
| `remove_object(id)` / `restore_object(id)` | toggle visibility |
| `measure(id_a, id_b?)` | size, or gap between two objects |
| `fits(id, x, z)` | collision check against other bboxes and walls |
| `insert_object(prompt, x, z)` | Muse Image → SAM 3.1 → SAM 3D Objects → new object |
| `render_view(camera?)` | screenshot from the viewer for verification |

Loop: transcript → Muse Wonder plans and calls tools → manifest updated → viewer
re-renders → screenshot → Muse Spark Open answers "does the image show <instruction>?"
→ if not, the verdict goes back to Muse Wonder for one retry. Every step is logged to
`history` with `verified: true|false`.

### 8. Server (`src/movable_reality/server.py`, FastAPI)
Serves scene folders, holds the agent, and keeps one WebSocket per viewer: the server
pushes manifest updates, the viewer pushes screenshots and voice audio. Until it exists,
the viewer polls `scene.json` once per second.

### 9. Viewer (`web/`)
three.js + Spark. One `SplatMesh` for the background and one per object, each a
`THREE.Object3D`, so move and hide are plain transform and `visible` changes. A root
group applies `world_from_recon`. `SparkXr` gives the WebXR button for the headset.
Spark finds splats with `scene.traverseVisible`, so hiding a parent group hides the object.
WebXR needs a secure context: for a headset over USB use `adb reverse tcp:5173 tcp:5173`
and open `http://localhost:5173` on the headset, otherwise serve over HTTPS.
Verified 26 Oct: `?scene=demo` renders, and remove/move edits appear without reload.

### 10. Coverage loop (lane: coverage, `src/movable_reality/coverage/`)
A phone video is enough wherever a surface was seen from several angles; it fails where
VR lets you walk and the phone never went (behind the sofa, under the desk, tops of
shelves).
1. **Score.** Per-region view count and angle spread from the camera poses, plus
   per-Gaussian uncertainty from training residuals computed post hoc, without changing
   the scene (U-3DGS-style regularised least squares). Output: `coverage/` arrays and a
   gap list in `scene.json`.
2. **Show.** Gaps glow red in the viewer. Muse Wonder names each gap by its nearest object.
3. **Recapture large gaps** while the user is still in the room: the same uncertainty
   ranks candidate viewpoints (next-best view); the app shows where to stand and point the
   phone; MapAnything registers the 10 s top-up together with the original frames; gsplat
   fine-tunes.
4. **Generate small gaps and object backs.** Render views of the gap from nearby walkable
   poses, repair them with Muse Image edit using the nearest real frames as references,
   and distil the repaired views back into the splats. Object backs come from SAM 3D
   Objects (stage 6). Baseline: no repair. Non-Meta repair models (Difix3D+, Stable
   Diffusion) are not allowed at runtime.
5. **Tag.** Generated splats get PLY property `generated = 1`; the viewer can show or hide
   them and measurements ignore them. Re-score until no gaps are left.

### 11. Per-object LOD (lane: lod)
Split the trained scene by object id (stage 5) into per-object PLYs, build one Spark LOD
tree (`.RAD`) per object and one for the background with Spark's `build-lod`, and record
the paths in `scene.json`. The viewer loads all trees and lets Spark traverse them
jointly, so the splat budget is shared across objects. Moving an object changes only its
transform, never a tree, so edits trigger no rebuild. Priority: the object under the
controller ray or being edited gets a higher detail weight.
Spark 2.4.0's types (checked 29 Oct) already expose what this needs: `SplatMesh` options
`lod` (`true` or `"quality"`, builds the tree at load time), `lodScale` (per-mesh detail
weight, used for priority) and `paged`; `SparkRenderer` options `lodSplatCount` (target
splats; default 500K-750K in WebXR), `lodSplatScale`, `lodRenderScale` (minimum splat
size in pixels, 1 to 5) and `lodTraverseMode`. The npm package ships no `build-lod`
binary, so the precomputed `.RAD` path needs Spark's repo tooling; until then, build at
load time with `lod: true`. Fork the traversal (MIT) only if these controls fall short.
Status 29 Oct (lane lod): `scripts/build_lod.py` builds Spark's `build-lod` with cargo and
writes one `.RAD` per object; the viewer falls back to `lod: true` when a tree is missing.
Moves trigger no rebuild, but Spark does not re-traverse on a one-frame move or a
`lodScale` change, so `web/src/lod/` requests a traversal. Details: `docs/LOD.md`.

### 12. Plane baking (lane: bake)
Fit planes to the Gaussians labelled floor and ceiling (RANSAC on centres, inside the
world frame), bound each plane by its inliers, render only those Gaussians with an
orthographic camera looking at the plane into a texture, and export one textured quad per
plane as glTF. Remove those Gaussians from the splat set. The quads are drawn first, opaque,
with depth writes on, so large, overdrawn splats disappear and the floor under a moved
object is no longer a hole. Fallback if seams show: floor only.
Status 29 Oct (lane bake): `scripts/bake_planes.py` fits, bakes (CPU reference rasterizer
with gsplat's classic-mode rules, 5 mm texels, push-pull hole fill) and writes
`planes/<label>.glb` plus `objects/background_baked.ply`; `background_ply` then points at
the reduced PLY and `plane_bake` keeps the original. `modal_jobs/bake.py` is the gsplat
`camera_model="ortho"` path (not yet run). Inputs: `ids/gaussian_ids.npy` indexes
`splat/scene.ply`, and `ids/labels.json` maps those ids to object ids (`"floor#1"`). Spark
2.4.0 splats depth-test against the opaque quads (verified in headless Chromium).
The alpha mask (30 Oct, B1, `bake/mask.py`) works on a 5 cm grid. Cells with baked
coverage count as surface, and on floors so do cells under the footprint of an object
whose bbox bottom lies within 15 cm of the plane (not floor, wall or ceiling). A 15 cm
closing bridges cracks, and enclosed holes stay opaque. A region that reaches the quad's
border becomes transparent only when it reaches deeper than 15 cm and background splats
0.2 to 1.5 m off the plane (walls) run along at least half of its edge, so an unobserved
corner of a rectangular room is filled, not cut. A masked plane's texture is WebP with
alpha (`EXT_texture_webp`, required) and its material `alphaMode: MASK` (cutoff 0.5),
which three.js renders as alpha test, so an L-shaped floor stops at the walls.

### 13. Renderer loop and budget controller (lane: renderer, `web/src/perf/`)
Each frame: opaque planes, then the joint LOD cut, then Spark's sort and blend. The HUD
shows frame time, visible splats and total splats, and logs XR frame times to JSON. The
controller compares measured frame time with 13.9 ms and nudges Spark's
`lodSplatCount` (or `lodSplatScale`) and `lodRenderScale`, with hysteresis so detail does
not flicker. `SplatMesh` also has foveation options (`coneFov`, `coneFoveate`,
`behindFoveate`): try them as fixed foveation before writing any of our own. A switch toggles the naive renderer (full
PLY, no LOD) for the proof shot. Synthetic splat sets (50k to 600k) and a scripted walk
path make device runs repeatable. Devices: Steam Frame through the Chromium XR build
(main), Pico 4 (low end), Quest 2 if available. Never measure through PC streaming.
Status 29 Oct (lane renderer): HUD, frame-time log, budget controller, naive/LOD toggle
and bench mode are in `web/src/perf/` and `web/src/bench/`; device steps, runtime-changeable
Spark settings and the R0 checklist are in `docs/DEVICES.md`.

## Decisions and fallbacks

| Decision | Why | Fallback |
|---|---|---|
| MapAnything instead of COLMAP | Metric scale, seconds instead of minutes | COLMAP or GLOMAP poses |
| Per-object PLY files | Editing becomes object transforms; no per-splat GPU edits | Spark `SplatEdit` SDF masks |
| Vote lifting instead of a feature field | No extra training, minutes not hours | Fewer, larger objects |
| Verifier on Muse Spark Open | Independent check, open-weight bonus | Muse Wonder self-check |
| Precomputed demo scenes | Live demo cannot fail on GPU queues | Video of the live run |
| WebXR + Spark instead of a native app | One URL on every headset; Spark already has LOD trees | Native app only if WebXR on the Frame fails |
| Per-object LOD trees | Edits are transforms; budget shared across objects | Global detail level only |
| Baked floor and ceiling | Removes the most overdrawn splats; fills holes under moved objects | Floor only |
| Muse Image for gap repair | Meta model, so allowed at runtime | Recapture, or leave the gap visible and tagged |
| Steam Frame as main device | Strongest standalone chip available | Pico 4 as main device |

## References

- Meta Model API: https://dev.meta.ai/docs/overview, models: https://dev.meta.ai/docs/models
- SAM 3.1: https://dev.meta.ai/docs/sam/overview, https://dev.meta.ai/docs/sam/segmenting
- SAM parser (Python): `pip install meta-sam-parser`
- Baseten Model APIs: https://docs.baseten.co/quickstart
- Modal: https://modal.com/docs/guide/gpu
- MapAnything: https://github.com/facebookresearch/map-anything
- gsplat: https://github.com/nerfstudio-project/gsplat, https://docs.gsplat.studio
- SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects
- Spark: https://sparkjs.dev, LOD in 2.0: https://sparkjs.dev/docs/new-features-2.0/
- U-3DGS (post-hoc uncertainty): https://chumsy0725.github.io/3DGS-Uncertainty/
- Muse Image editing on Meta Model API: https://developer.meta.com/ai/resources/blog/build-with-muse-Image/
- Chromium with WebXR for the Steam Frame: https://github.com/saphid/chromium-webxr-steam-frame
- Meta Spatial SDK splats (150k guidance): https://developers.meta.com/horizon/documentation/spatial-sdk/spatial-sdk-splats/
