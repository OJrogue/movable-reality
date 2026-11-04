# Movable Reality marketing notes

Working copy for the submission page and the pitch. Claims marked
"measured" have a log or benchmark behind them. Claims marked "planned" are not built yet and
must not appear in the submission as finished features.

## Positioning (4 Nov, OJ)

Do not claim it is finished or the best, and do not undersell it either. Lead with what it does:
one walk of video in, a room with separately movable furniture out, on one command and eight Meta
models. Show the input before the result and let that make the point. Say once, without
apologising, that it is a working demo on two synthetic rooms and a good place to start for research
or a project: on the page that is the "Build on it" section, in the video the closing beat. Do not
repeat "rough" or "not a product" anywhere else. No claim about what is being worked on next. Plain
words, no superlatives. Team name: **Larracat**.

Who it is for: researchers and developers working on object-level splats, room editing or
agents that act on 3D scenes, who can start from a pipeline that already runs.

## One line

One video of a room goes in. A 3D room comes out, and you can move the furniture.

Landing page hook: "Film a room. Move the furniture." No VR claims anywhere (D21).

## Short version, for the submission page

Movable Reality turns one video of a room into a 3D Gaussian Splatting scene where each piece
of furniture is a separate object. You can drag things around in the browser or tell an agent
what to do, and a second model checks the result in a screenshot. It runs on eight Meta models
and takes one command, end to end. The two demo rooms were rendered in Blender and built from
54 s and 132 s of video, with no depth, floor plan or hand labels. The code is open: separate
object splats, one JSON scene file, and Blender rooms with ground truth to measure your changes
against. It is a working demo on synthetic rooms and a good place to start for research or a
project.

## X post

One walk of video in, a room with movable furniture out. Eight Meta models, one command, open source. Try the two demo rooms and move things around. [link]

## Longer-term idea (not today's pitch)

Trying a layout before you buy or build is where this could go one day. Do not present it as
something the demo does today: it needs real footage to work first and better than 5 to 12%
on room sizes.

## What Movable Reality does

1. A video of one slow walk through a room goes in. Today that is a Blender render (54 s for
   the bedroom, 122 s plus a 10 s top-up for the living room), along with the focal length
   and the ceiling height.
2. The pipeline turns it into a 3D Gaussian Splatting scene in metres, with every piece of
   furniture as a separate object.
3. You change the room by dragging, or by typing or saying a command. "Delete the ukulele."
   "Put a ceramic vase on the dresser." "Undo that."
4. An agent built on Muse Wonder carries out the request, and Muse Spark Open checks the result
   in a screenshot.
5. The room opens from a link in a laptop browser, or in a phone browser (checked in emulation only).

What is true today.

- Built and run on both demo rooms: separation into objects, moving, rotating, removing and
  restoring, agent commands with a visual check, furniture from a premade library, measuring,
  undo of every edit, coverage scoring.
- Built, but only run on small test scenes: repairing thin spots with Muse Image and re-scoring.
  Neither demo room has generated splats, so the switch that hides them only appears when a
  room has some.
- Voice: the server path ran on a recorded clip. It was never tried with a live microphone.
- Not supported yet: real footage.
- Sizes: with the ceiling height given, the bedroom is about 5% too large and the living room
  about 12%, compared with the Blender models.

## Any device, best quality on each

Movable Reality is one web app. It opens from a link, with no installation and no app store.

| Device | How it is used | Quality target |
| --- | --- | --- |
| Laptop or desktop browser | Mouse and keyboard, typed or spoken commands | Full resolution and the largest splat budget the GPU sustains at the display refresh rate. Not reduced to the VR budget. |
| Phone or tablet browser | Touch (orbit, pinch, tap to select, drag to move), hold-to-talk | Resolution and splat budget scaled to the phone GPU, 30 to 60 fps |

Each device gets its own quality profile. The laptop shows the full scene; the phone gets
level-of-detail trees per object and a frame-time controller that keeps it smooth. The
same scene file feeds both. The WebXR path exists in the code but is untested on a headset
and not part of the pitch (D21).

Status. The desktop browser path is built. Per-device profiles and touch
controls are merged (PR #15) and tested in an emulated phone viewport, not yet on a real phone. Headset numbers are only claimed with their raw
frame-time logs under `benchmarks/`; none are claimed.

## The self-correcting pipeline

A 30-second video never sees everything. The back of the sofa, the floor under the table and
the corner behind the door are missing or blurry. Most scanning apps leave holes there or
fill them silently. Movable Reality finds them, tells the user, and fixes them in one of two
ways. Everything it invents stays marked.

### Step 1. Find the gaps

After reconstruction the coverage stage scores every part of the room.

- How many camera views saw it.
- How wide the spread of viewing angles was. One view straight on is weaker than three from
  different sides.
- How uncertain the fitted splats still are (a per-Gaussian uncertainty from a short fit on
  the GPU, averaged over the directions the cameras actually saw).

Places a user could look at but that score low become gaps. Each gap has a size, a position
and a name ("behind the armchair"). The viewer can draw them as an overlay (key G, or
`?gaps=1`).

### Step 2. Choose the fix

There are two options, and the choice depends on the gap and on whether the user is still in
the room.

**Option A. Film more (top-up capture).** Used for large gaps while the user is in the room.

1. The pipeline picks the next-best viewpoint, the phone position and direction that would
   see the most missing area.
2. The viewer shows it as a yellow phone frame with a ring on the floor where to stand.
3. The user films about 10 seconds from there.
4. MapAnything registers the new frames into the same metric frame as the first video. The
   pipeline checks that the alignment is consistent (scale and orientation agree on the
   shared cameras) and refuses it otherwise.
5. Gaussian splatting continues training with the new views, the objects are lifted again
   and the coverage is scored again.

The result is real data with nothing invented. This is the preferred fix.

**Option B. Generate (repair).** Used for small gaps, for the back of objects, and when the
user has left the room.

1. The pipeline renders the gap from a nearby viewpoint.
2. Muse Image repairs the rendered view, with a real frame of the room as the reference so
   colours and materials match.
3. An alignment check compares the repaired image with the render outside the gap. A repair
   that redrew the room from another angle is rejected.
4. Accepted repairs are distilled back into new Gaussians, while the scanned Gaussians stay
   frozen.
5. Every new splat carries the flag `generated = 1`. The user can hide all generated content
   with one switch (key T, or the "show generated" toggle).

For whole objects, SAM 3D Objects completes each piece of furniture from its best view, so a
sofa filmed only from the front still has a back when the user turns it around. Completed
parts are flagged the same way.

### Step 3. Score again

The loop repeats until no gaps above the size threshold remain, or until the user accepts
the rest. The final scene reports how many splats are real and how many are generated.

### Why the marking matters

For visualisation before building, the user needs to know what is measured and what is
guessed. A wall that was filmed is reliable for a clearance check. A wall that was generated
is a plausible guess. The "show generated" switch makes that difference visible instead of
hiding it.

Measured so far on synthetic and test scenes (31 Oct).

- Coverage on an 8-second test capture flagged 25 gaps. 77% of voxels in the room were never
  observed, as expected for a short capture facing one way.
- One Muse Image repair passed the alignment check (L1 0.007 outside the gap) and was
  distilled into 138 generated splats. Masked repair error fell from 0.049 to 0.011, while
  the error on the real training frames moved from 0.0254 to 0.0256.
- A top-up registration on a controlled test aligned with scale 1.0006 and 2.5 mm RMS.
  A top-up with a genuinely new video is not yet measured.

## Built on Meta models

Every model the product calls at runtime is a Meta model.

- Muse Wonder plans and calls the editing tools, watches the video and lists the objects.
- SAM 3.1 tracks every object through the video.
- MapAnything turns the video into metric cameras and geometry.
- SAM 3D Objects completes objects and their unseen backs.
- Muse Image generates new furniture and repairs coverage gaps.
- Muse Voice Transcribe hears voice commands.
- Muse Spark Open checks each edit in a rendered view and triggers a retry when it fails.

## Audiences and channels

- Researchers working on splats, scene understanding and 3D editing. The repo, the numbers and
  the Blender rooms matter here.
- Developers building agents that act on 3D scenes. The scene file and the agent's tools matter.
- Hackathon judges and the X audience: the short video, with the input shown next to the result.

## Claims to avoid until measured

- Accuracy figures in centimetres, or the word "accurate". The rooms are 5 to 12% too large.
- Anything that implies it works on real footage today.
- "Photoreal", "seamless", "production-ready".
- "Works on every phone". It was checked in phone emulation only.
- "Voice works". It only ran from a recorded clip.
- "Gaps are filled in the demo". Repair only ran on test scenes.
- "Design from a floor plan". Planned, not built.
- Anything about VR or headsets (D21). Untested.
