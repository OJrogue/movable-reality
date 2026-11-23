# Submission video script

Length: about 2:36 (the limit in `docs/HACKATHON.md` is 3 min). About 267 words of voice-over at a calm pace.
Clips are in `data/broll/` (gitignored, 1920x1080, 30 fps, no audio). Every clip is a screen
recording of the real viewer; see "Honest labels" before cutting.

## The pitch

How little goes in, what comes out, and why you would build on it. It is a working demo and a good
place to start for research or a project. Say that once, near the end, without apologising, and do
not sell a finished product or a layout tool. The input comes before the result, because that is
the part that surprises people.

## Script

| Time | Voice-over | Picture | On screen |
|---|---|---:|---|
| 0:00 | Movable Reality. Record your room and get a customizable 3D room. | `16_room_before`, cut to `17_room_after` | **Movable Reality** · demo rooms rendered in Blender |
| 0:06 | All it needs is a video walkthrough of the room and the ceiling height. No depth sensor, no floor plan, no manual labelling. | New `A_room_card`, or `01_input_walk_8x` | Rendered in Blender, 132 s of video |
| 0:17 | Muse Wonder watches the video and lists the objects. SAM 3.1 tracks each one. MapAnything finds the cameras, and Gaussian splatting builds the room. One command takes it from video to finished room. | `02_landing_and_loading`, the loading part only | Muse Wonder · SAM 3.1 · MapAnything |
| 0:34 | Every object comes out as its own set of Gaussian splats. | `03_drag_armchair` |  |
| 0:40 | Select any, move it along an axis, or turn it. | `04_gizmo_move`, `05_gizmo_rotate` |  |
| 0:49 | Measure it. | `07_measure` |  |
| 0:55 | Add furniture from a library. | `08_library_add`, first 8 s |  |
| 1:03 | Or just say what you want. | `13_voice_fast` | Muse Voice Transcribe · sped up |
| 1:10 | Ask it to move the sofa half a metre, and it moves it thirty centimetres, because the side table is in the way. | New `D_sofa_move`, or `09_agent_move_fast` | Muse Wonder · sped up |
| 1:22 | It answers questions too. Will the armchair fit between the sofa and the bookshelf? No. The gap is one metre two, and the chair is wider. | `10_agent_fit` | Reply as recorded |
| 1:35 | Ask for a plant, and it picks one and places it. After every edit, Muse Spark Open looks at the room and checks the result. | `11_agent_add_fast`, hold on ✓ Checked | Muse Spark Open · sped up |
| 1:47 | No walk sees everything. The pipeline scores what the video covered and marks the spots it barely saw. The detail you get follows the data you give it. | `14_coverage_gaps` |  |
| 2:02 | This is a working demo, on two rooms rendered in Blender, and it can be a good starting point for a project or research. | New `B_build_section`, or `15_room_glide` | A working demo, and a good place to start |
| 2:14 | Runs on eight Meta models. It's MIT licensed, each stage is a plain script you can swap, and the test rooms and scoring tools come with it, so you can measure your changes. | Models card over `15_room_glide` | The eight models, repo link |
| 2:30 | Try the demo at the link below. | New `C_landing_hero`, or `17_room_after` | **Live demo URL** · team Larracat |

## Shots to record

The landing page changed after the clips were recorded. These three show the new site, and the
script names a fallback from the existing clips for each.

| Shot | What to record |
|---|---|
| `A_room_card` | Landing page, click Living room, the room card opens with both videos and the numbers |
| `B_build_section` | Scroll the landing page to "A working demo, and a good place to start" |
| `C_landing_hero` | The top of the landing page with the headline and "Open the demo" |
| `E_plant_add` | Next recording: keep the sofa end in frame. Type exactly "add a large plant from the library next to the sofa". Speed up the wait only, and hold on ✓ Checked |
| `D_sofa_move` | Next recording: keep the sofa in frame so you see it move. The Living room's start view faces away from the sofa, so turn the camera first. Type "move the sofa half a metre towards the window". Speed up the wait only, not the move, and hold on the reply. Live reply on 5 Nov: "I moved the sofa about 0.31 m towards the window — a full 0.5 m wouldn't fit, as the side table is only 0.37 m away in that direction. ✓ Checked" |

## Models card (2:14)

Muse Wonder, Muse Spark, Muse Image, Muse Voice Transcribe, SAM 3.1 (Meta Model API);
Muse Spark Open (open weights on Baseten); MapAnything, SAM 3D Objects (open weights on Modal).

Muse Image, SAM 3D Objects and Muse Spark are named on the card but not shown at work: generating
furniture is switched off in the hosted demo's mode, so no clip shows it.

## Honest labels

Keep these or say them out loud:

- **The room is synthetic.** Rendered in Blender (CC0 assets). Say so at 2:02, keep the label at 0:00 and 0:06, because the opening line says "Record your room". Do not say "filmed on a phone" over this footage.
- **Sizes are approximate.** The living room is about 12% too large against the Blender model and the
  bedroom about 5%. Do not say "true to scale" or "accurate". The agent's numbers in `09` and `10` are
  shown as replies, not as measured truth.
- **Agent clips are sped up.** `_fast` clips run at 10x while waiting. Real waits in the takes are in
  the clip table. Keep "sped up" on screen.
- **The pipeline is not instant.** About 3.7 h of stage time on blender07 (sum of `pipeline.json`).
  Do not imply the room appears in seconds.
- **The voice is a test voice.** `13_voice` feeds a Windows text-to-speech clip in place of the
  microphone; from there it takes the real recording and Muse Voice Transcribe path. The clip has no
  sound. If you add voice audio, record your own take in the viewer rather than dubbing this one.

## Do not show or claim

- **Moving the TV or removing the sofa.** The TV leaves an obvious ghost on the wall, and the sofa a
  large smear.
- **Removing the rug or clicking the coffee table.** The rug's box covers the table, so a click on the
  table can select the rug, and the floor under the rug is a smear.
- **Moving the coffee table or the ottoman.** Parts stay behind.
- **Generated furniture, the "show generated" switch.** blender07 has 0 generated splats, and the
  hosted mode has generation off.
- **Top-up capture closing the gaps.** Not recorded here.
- **Phone or VR.** No claims about either (D21).
- **The old landing page.** `02_landing_and_loading` was recorded before the rewrite: it says "with
  your phone" and shows a "coming soon" card. Use only its loading part.
- **Layout planning as a finished use.** "Try the layout before you lift the sofa" sells a product
  that needs real footage and better sizes first.
- **PSNR as a quality headline.** 27.49 dB mean over 45 test views is honest but the clean-up was
  neutral on it. Leave it for the README.

## Clips

Recorded 4 Nov from the local dev server, on an RTX 3060 at 1920x1080, with a private session copy of
blender07. Mouse and keys are scripted real input; the arrow cursor is drawn by the recording script.
Agent replies are as recorded.

| Clip | Length | What happens | Real agent wait |
|---|---:|---|---:|
| `01_input_walk_8x` | 15 s | The input walk at 8x | |
| `02_landing_and_loading` | 26 s | Old landing page, opening the room, loading bar, first view with the help panel | |
| `03_drag_armchair` | 12 s | Click and drag the armchair across the floor | |
| `04_gizmo_move` | 14 s | Select the armchair, drag its red axis 0.8 m | |
| `05_gizmo_rotate` | 15 s | Rotate mode, turn the armchair 84 degrees | |
| `07_measure` | 9 s | Alt+click twice: sofa 2.20 m | |
| `08_library_add` | 22 s | + panel, add a tall plant, drag it. It ends in the foreground over the table | |
| `09_agent_move` / `_fast` | 148 / 26 s | "move the sofa half a metre towards the window": about 0.3 m, side table in the way, ✓ Checked | 138 s |
| `10_agent_fit` | 25 s | "does the armchair fit between the sofa and the bookshelf?": No, gap 1.02 m, chair 1.24 m (1.11 m turned) | 15 s |
| `11_agent_add` / `_fast` | 28 / 14 s | "add a large plant from the library next to the sofa": ✓ Checked | 18 s |
| `13_voice` / `_fast` | 31 / 14 s | Hold Space: "turn the armchair to face the sofa", ✓ Checked | about 20 s |
| `14_coverage_gaps` | 12 s | Key G: red gap spheres and next-best views, camera glide | |
| `15_room_glide` | 30 s | Camera through the room, no interface | |
| `16_room_before`, `17_room_after` | 8 s each | Same camera move, before and after: sofa 20 cm along, armchair turned 30 degrees, plant added | |
