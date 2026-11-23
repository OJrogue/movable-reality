# Archive

Small records kept after the local `data/` folder was cleaned up on 23 Nov 2026. The large files
(renders, trained splats, LOD trees, frames, clips) are gone; everything here is text.

## video/

How the submission video (1:39, `movable_reality_final.mp4`) was made.

- `VIDEO_SCRIPT.md`: the shot list and voice-over timeline the cut followed.
- `VOICEOVER.txt`: the voice-over text.
- `movable_reality.en.srt`: the subtitles, timed to the final cut.
- `PUBLISH.md`: YouTube and X titles, description, chapters and upload steps.
- `recorder/`: screen recording of the real viewer. `lib.mjs` drives Chrome over CDP;
  `shots*.mjs` script each shot against the dev server (`?scene=blender07`); `encode.sh` turns the
  raw `.webm` captures into 1080p30 H.264 clips (and the 10x `_fast` agent cuts).
- `cut/`: the edit, run from the repo root with ffmpeg on the path.
  - `segment.py`, `words.py`, `segments.json`: split the voice-over recording into takes.
  - `music.py`: the music, generated in code (pad and pulse variants).
  - `anim_data.py`, `anim.py`, `motion.py`: the animated walk and pipeline sections.
  - `build.py` (first cut), `build_v1.py` (2:56 backup), `build2.py` (final cut): scene list with
    clip, in point, length, speed, crop, labels and voice offsets; `lbl*.txt`, `list*.txt` are its
    generated label and concat files.
  - `make_publish.py`: final copy, SRT, thumbnail and the captioned X version.
  - `check_*.py`, `retest.py`, `sheets.py`: checks and contact sheets.

The clips, voice recording and intermediate renders are not kept.

## scenes/

Every JSON, Markdown and text file from `data/scenes/<id>/`: `scene.json` (with its edit
history), `pipeline.json`, `inventory.json`, `frames.json`, `completion.json`, coverage and eval
scores. Scenes made from the private test video (`testroom*`) are left out.

## assets/

`assets.json` (Poly Haven assets used in the Blender rooms, with authors, URLs and licences) and
`catalog.json` (the premade furniture library). `docs/ASSETS.md` is the readable version;
`scripts/blender_assets.py` and `scripts/build_library.py` rebuild the files.
