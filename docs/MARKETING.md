# Movable Reality marketing notes

Working copy for the submission page, the demo video script and the pitch. Claims marked
"measured" have a log or benchmark behind them. Claims marked "planned" are not built yet and
must not appear in the submission as finished features.

## One line

Film your room for 30 seconds. Walk through it, move anything in it and try out changes
before you spend money on them, on a laptop, a phone or a VR headset.

## Taglines

- See it before you build it.
- Your room, editable.
- Every mistake you catch on screen is one you do not pay a builder to undo.
- Move the sofa without lifting it.
- One video in. A room you can rearrange out.

## The problem

People decide on renovations, furniture and layouts from floor plans, catalogue photos and
imagination. Most of them cannot read a floor plan in 3D. The mistakes show up after the money
is spent. A sofa that blocks a door, a kitchen island that leaves 70 cm to walk past, a wardrobe
that does not fit under the slope of an attic. Returning furniture costs delivery fees.
Moving a wall socket or a radiator after the plaster is up costs hundreds. Professional 3D
visualisation exists, but it is priced for architects and takes days.

## What Movable Reality does

1. The user films the room with a phone for about 30 seconds.
2. The pipeline turns the video into a photoreal 3D Gaussian Splatting scene at real scale
   (metres), in which every piece of furniture is a separate object.
3. The user walks through the room and changes it, by voice or by typing. "Move the sofa
   half a metre towards the window." "Does the armchair fit next to the bookshelf?" "Add a
   reading lamp in the corner." "Undo that."
4. An agent built on Meta's Muse models carries out the request, checks the result in a
   rendered view and corrects itself when the edit went wrong.
5. The room runs on the device the user has. A laptop browser, a phone or a standalone VR
   headset.

## Visualisation before it is built

This is the use case with the largest market and the clearest value per user.

- Renovation and fit-out: capture the bare room or the building site, then try layouts.
  Check clearances, door swings and walking paths with real dimensions before ordering
  anything.
- Furniture buying: insert the piece you are about to buy and see it in your own room, at
  its real size, next to your own furniture. Measure the gap to the wall instead of
  guessing.
- Moving house: capture the new flat on viewing day and plan where everything goes before the
  van arrives.
- Interior designers and kitchen studios: send the client a link to their own room with the
  proposal in it. The client opens it on a phone or a laptop with no installation.
- Estate agents and landlords: show a property staged in two or three layouts from one
  capture.

Message for this section. A change on screen costs nothing. The same change after the
builders have finished costs real money. Movable Reality moves the decision to the cheap
side.

What is true today and what is not.

- Built: capture of an existing space, separation into movable objects, moving, rotating,
  removing and restoring objects, fit and clearance checks in metres, generated furniture
  inserted on the floor, undo of every edit.
- Built: the "show generated" switch, which hides everything the models invented so the
  user can see what was really filmed.
- Planned: starting from a floor plan or a CAD model instead of a video. Today the room has
  to exist, even if it is empty.

## Any device, best quality on each

Movable Reality is one web app. It opens from a link, with no installation and no app store.

| Device | How it is used | Quality target |
| --- | --- | --- |
| Laptop or desktop browser | Mouse and keyboard, typed or spoken commands | Full resolution and the largest splat budget the GPU sustains at the display refresh rate. Not reduced to the VR budget. |
| Phone or tablet browser | Touch (orbit, pinch, tap to select, drag to move), hold-to-talk | Resolution and splat budget scaled to the phone GPU, 30 to 60 fps |
| Standalone VR headset (WebXR) | Walk through the room at 1:1 scale, controller and voice | Locked 72 Hz, p99 frame time at or under 13.9 ms, never streamed from a PC |

Each device gets its own quality profile. The laptop shows the full scene. The headset gets
level-of-detail trees per object and a frame-time controller that keeps it at 72 Hz. The
phone sits in between. The same scene file feeds all three.

Status. The desktop browser and WebXR paths are built. Per-device profiles and touch
controls are in progress (lane/devices). Headset numbers are only claimed with their raw
frame-time logs under `benchmarks/`.

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

## Demo video outline (3 minutes)

1. 0:00 Hook. A person stands in a half-furnished room with a tape measure and a furniture
   catalogue. Cut to the same room on a laptop, then on a phone, then in a headset.
   "See it before you build it."
2. 0:20 Capture. The 30-second walk, sped up, with the filming guide in the corner.
3. 0:40 The room appears. Orbit on the laptop, every object outlined once.
4. 1:00 Voice edits. Move the sofa, ask whether the armchair fits, add a lamp. Show the
   verifier's check mark after each edit.
5. 1:40 Self-correction. Gap overlay on, next-best viewpoint shown, 10-second top-up,
   gaps close. Then the generated toggle, showing what was invented.
6. 2:10 Any device. The same room on a phone with touch, then in the headset at 1:1 scale
   with the frame-time HUD at 72 Hz.
7. 2:40 Before and after. The original layout next to the new one. Closing line and the
   list of Meta models.

## Audiences and channels

- Homeowners planning a renovation. Short vertical videos, before and after on a phone.
- Furniture retailers. Embed "see it in your room" with the customer's own capture.
- Interior designers and kitchen studios. A shareable link per client proposal.
- VR enthusiasts. Photoreal room-scale scenes at a locked 72 Hz on a standalone headset.

## Claims to avoid until measured

- Accuracy figures in centimetres for real rooms. The metric scale on blender01 came out
  1.8x too small before the ceiling-height correction, and real-room accuracy is not yet
  measured.
- "Works on every phone". Only devices with a measured frame-time log count.
- "Design from a floor plan". Planned, not built.
