# Movable Reality

Walk around a room once with your phone. What comes back is a 3D copy of it, true to
scale, in which the sofa, the lamp and every chair are separate objects you can move.
Drag them around in the browser, or say what should go where. The parts the phone did not
see are filled in, by a short top-up video or by Meta's generative models, and everything
generated stays tagged.

Under the hood it is a metric 3D Gaussian Splatting scene with one splat per object,
built by a pipeline of eight Meta models, and a three.js + Spark viewer that runs on a
laptop or a phone.

> Entry for the **Meta Global AI Developer Hackathon** (Indie Developer track, Push the
> Frontier) by team **Gaussian Movers**. Everything in this repository was built during
> the event (26 Oct to 4 Nov 2026). See [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) for what
> was built when. Third-party components are listed below.

**Status (3 Nov):** the pipeline runs end to end on the synthetic rooms. The demo has a
landing page and three rooms: the living room `blender07`, the bedroom `blender06`, and a
real phone capture (`room01`, not filmed yet). The viewer was checked in desktop Chromium
and its phone emulation. A WebXR path exists in the code but was not tested on a headset,
so the project makes no VR or headset frame-rate claims.

## Try the demo

```bash
cd web && npm install && npm run dev       # http://localhost:5173/
uv run python -m movable_reality.server    # optional: the agent behind typed and spoken commands
```

`/` is the landing page: a showcase video, what the project does, the pipeline, the Meta
models, the input video of each room, and **Open the demo**, which lets you pick a room.
In the viewer:

- **Desktop:** drag to look, W A S D to walk, click an object to select it, drag to move
  it, Q/E to rotate, Delete to hide, Ctrl+Z to undo, Alt+click twice to measure.
- **Phone:** one finger looks, two fingers pan and pinch, tap selects, drag moves.
- **Commands:** type in the bar at the bottom and press Enter, or hold Space (or the
  microphone button) to talk. Commands need the server; without it the bar says
  "Agent offline" and direct editing still works.
- **Top bar:** back to the landing page, switch room, help, and ⋮ for coverage gaps,
  generated splats and how objects are drawn (scanned splats, SAM 3D splats, SAM 3D meshes).
- **Developer info** (frame-time HUD, save log, detail slider, object style label) is off
  by default. Turn it on with the last switch in ⋮, the backtick key, or `?debug=1`.

The demo media live in `web/public/media/`: card images (`thumb_<scene>.jpg`, committed),
the hero video `showcase.mp4` and the input videos `capture_<scene>.mp4`. Videos are
gitignored like every `.mp4`, so copy them next to the build when hosting. Without them the
page shows stills and "Capture coming soon".

## How it works

1. **Capture:** one phone video walking around the room; the sharpest frames (4 per
   second) are kept automatically.
2. **Reconstruct:** MapAnything predicts metric cameras and geometry; gsplat trains the
   Gaussian splat scene (GPU jobs on Modal).
3. **Understand:** Muse Wonder watches the video and lists the objects; SAM 3.1 tracks
   each one through the video; masks are lifted onto individual Gaussians, so every object
   becomes its own splat.
4. **Edit:** a Muse Wonder agent turns spoken commands (Muse Voice Transcribe) into scene
   edits with real dimensions; Muse Spark Open checks every edit on a rendered view;
   Muse Image creates new furniture to insert.
5. **Fill gaps:** coverage is scored per region; large gaps prompt a 10 s top-up video,
   small gaps and object backs are repaired with Muse Image and SAM 3D Objects, and every
   generated splat is tagged.
6. **View:** a three.js + Spark viewer in the browser on a laptop or a phone. Walls, floor
   and ceiling become baked textured planes, every object gets its own LOD tree, and a
   frame-time budget controller picks the detail each device can hold.

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Meta models used

| Model | Used for |
|---|---|
| Muse Wonder (Meta Model API) | Editing agent with tool calls; object inventory from video |
| SAM 3.1 (Meta Model API) | Segmentation and tracking of every object in the video |
| Muse Image (Meta Model API) | Generating furniture to insert; repairing views of coverage gaps |
| Muse Voice Transcribe (Meta Model API) | Voice commands |
| Muse Spark (Meta Model API) | Helper calls |
| Muse Spark Open (open weights, served on Baseten) | Visual verification of every edit |
| MapAnything (Meta open weights, on Modal) | Metric cameras and geometry; registering top-up videos |
| SAM 3D Objects (Meta open weights, on Modal) | Completing objects and their unseen backs |

Every model the project calls at runtime is a Meta model.

## Setup

Requirements: Python 3.12 with [uv](https://docs.astral.sh/uv/), Node.js 22, ffmpeg,
a Modal account.

```bash
cp .env.example .env            # add MODEL_API_KEY and BASETEN_API_KEY
uv sync
uv run modal setup              # once
uv run python scripts/smoke_meta.py
uv run python scripts/smoke_baseten.py

uv run python scripts/make_demo_scene.py   # synthetic scene, no GPU needed
cd web && npm install && npm run dev       # open http://localhost:5173/?scene=demo
cd ..
uv run python scripts/edit_scene.py demo remove "table#1"   # viewer updates live
```

`http://localhost:5173/scenes/` lists every scene in `data/scenes/`.

### Your own room, in one command

Film the room first. Hold the phone level and walk the perimeter slowly, facing inwards,
for 30 to 60 s. Then do one loop at knee height and one loop tilted slightly up, so the
ceiling and the upper walls are covered. Lock exposure and focus if the camera app allows
it, keep the lights on, and leave out mirrors, people and pets.

```bash
uv run python scripts/run_pipeline.py path/to/room.mp4 --scene room01 --focal35 26
```

`--focal35` is the phone's 35 mm-equivalent focal length. It is in the video's EXIF, and
is 26 for most phone main cameras. The run takes about 25 to 40 min, mostly on Modal GPUs,
and costs about $1 to $2. Every paid call is cached, so a rerun is free. Each stage logs to
`data/scenes/room01/logs/` and its state goes to `data/scenes/room01/pipeline.json`.
`http://localhost:5173/scenes/` lists every scene, with a live stage strip while a run is
going.

The metric scale comes from the furniture itself. Muse Spark gives a typical height for
each object noun, and every floor-standing object votes. If you have a tape measure, pass
`--ceiling-height 2.5` instead. After a fix, `--from <stage>` reruns that stage and
everything after it. `--dry-run` shows what would run.

## Open-source components and licences

| Component | Licence | Role |
|---|---|---|
| [MapAnything](https://github.com/facebookresearch/map-anything) (`facebook/map-anything-apache`) | Apache-2.0 | Metric cameras and geometry |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 | Gaussian splat training |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | SAM License | Object completion |
| [Spark](https://sparkjs.dev) | MIT | Web splat renderer |
| [three.js](https://threejs.org) | MIT | Web 3D |
| [meta-sam-parser](https://pypi.org/project/meta-sam-parser/) | see package | Parsing SAM 3.1 output |
| [Blender](https://www.blender.org) 4.5 LTS | GPL-3.0 (tool only, not shipped) | Renders the synthetic ground-truth rooms `blender01` to `blender07` |
| [Poly Haven](https://polyhaven.com) models and textures | CC0 | Furniture and materials of the synthetic rooms; full list in [docs/ASSETS.md](docs/ASSETS.md) |

## How this was built

Solo build by OJ during the event, written with AI coding assistants (Claude Code and
Muse Code). The coding tools are not part of the product: every model the product calls
is a Meta model.

## Licence

MIT, see [LICENSE](LICENSE). Third-party components keep their own licences.
