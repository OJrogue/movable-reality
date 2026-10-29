# Plan

All times Prague. Hard deadline Thu 5 Nov 03:00. Own deadline: submit Wed 4 Nov 20:00.
Updated 29 Oct: nothing is cut; build lanes run in parallel; the freeze below is a
suggestion, not a rule. This file is the plan agents follow.

## Frontier claim

A phone-captured room, separated into editable objects, with gaps in the capture found
and filled (recaptured or generated, generated parts tagged), rendered **on a standalone
headset** at p99 frame time <= 13.9 ms (72 FPS) while the user rearranges it by voice.
Meta's own on-device splat path renders one splat at a time and advises staying under
150k splats; Hyperscape relies on cloud rendering. This renders a multi-million-splat,
object-separated room locally.

## Lanes and milestones

| Day | Core (M) | Coverage (C) | Renderer (R) | Ship |
|---|---|---|---|---|
| Thu 29 | M1: one real room reconstructed end to end on Modal; smoke tests | | R0: Spark viewer in WebXR on the Frame and the Pico 4; FPS logged at 50k, 150k, 300k, 600k splats; main device chosen | |
| Fri 30 | M2: per-object ids from SAM 3.1 (`lift_ids`), per-object PLYs, floor alignment | C1: gap map from view counts, angle spread and residual uncertainty, visible in the viewer | R1: one `.RAD` tree per object, joint LOD cut, priority for the targeted object | |
| Sat 31 | M3: agent tools, server + WebSocket, verifier | C2: next-best-view prompt; 10 s top-up registered with MapAnything; fine-tune | R2: floor and ceiling baked to opaque planes, depth-tested with splats | |
| Sun 1 | M4: insert path and SAM 3D Objects completion | C3: Muse Image repairs distilled back and tagged; compared with no repair on the test walk | R3: budget controller holds 13.9 ms on the walk path | |
| Mon 2 | M5: voice; all three scenes run end to end | | R4: 10 min logs for every ablation step on both devices, plus coverage metrics | Freeze 20:00 (suggested) |
| Tue 3 | | | | Record video; draft submission by 20:00 (last troubleshooting day) |
| Wed 4 | | | | Final cut, README; submit by 20:00; X post |

The `lod` and `bake` work feed R1 and R2. Before real scenes exist, every lane develops
against the synthetic `demo` scene and synthetic splat sets.

If a lane slips a day, move the freeze to Tue 3 Nov 12:00 at the latest and record that
afternoon. Nothing is dropped; `docs/ARCHITECTURE.md` lists a fallback for each part.

## Capture

Three rooms, 30 s each, slow orbit, good light. For each room also film a **second,
different walk** that is never used for training: it is the test set for off-path quality.

## Benchmarks (for README and video)

1. Time-to-twin: video in to editable scene, broken down by stage, vs COLMAP + 3DGS.
2. Reconstruction quality: PSNR / SSIM / LPIPS on held-out frames (gsplat `test_every 8`),
   MapAnything init vs COLMAP init, 2 Mip-NeRF 360 scenes + 1 own scene.
3. Object separation: share of Gaussians with consistent ids across views; manual check
   of 10 objects.
4. Agent: success rate on 20 scripted commands, with and without the verifier loop.
5. Cost per scene in API dollars and GPU minutes.
6. Headset performance, same scripted 10 min walk on both devices:
   | Metric | Bar |
   |---|---|
   | Frame time p50 / p99 | p99 <= 13.9 ms |
   | Frames over 13.9 ms | < 1% |
   | FPS first minute vs last minute | drop <= 5% |
   | Total and visible splats | report |
   | Quality retained: 10 fixed poses, headset-budget render vs full desktop render, PSNR / LPIPS | report |
   | Frame-time spike while an object moves | no frame over 13.9 ms |
   Ablation, one row per step: full PLY, + Spark LOD, + per-object trees, + baked planes,
   + budget controller. Baselines to name: full PLY without LOD, and a 150k-splat cap
   (Spatial SDK guidance).
7. Coverage:
   | Metric | How |
   |---|---|
   | Off-path quality | PSNR / LPIPS on the second test walk, before and after recapture and repair |
   | Gap share | share of the room flagged as gaps, before and after the loop |
   | Top-up cost | seconds of extra video and minutes until re-scored |

Raw logs go to `benchmarks/`.

## Video (3 min, judged live in the final round; confirm the length limit)

0:00 capture with on-screen timer → 0:20 scene appears → 0:30 headset on, no PC, HUD with
frame time and splat counts → 0:45 gaps glow red; top-up prompt; repaired areas and the
"generated" toggle → 1:10 voice edit under load, frame time flat → 1:25 proof toggle:
naive renderer judders, LOD renderer holds → 1:35 pick up a chair, unseen back is
complete → 1:50 "add a sofa, does it fit?" answered in metres → 2:05 same URL on the
Pico 4 (and a Quest 2 if borrowed) → 2:15 how it works (diagram, Meta models named) →
2:40 benchmark numbers → 2:55 what was built during the hackathon.
