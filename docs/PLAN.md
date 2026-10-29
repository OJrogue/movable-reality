# Plan

All times Prague. Hard deadline Thu 5 Nov 03:00. Own deadline: submit Wed 4 Nov 20:00.
Updated 29 Oct: nothing is cut. The freeze is a suggestion, not a rule. This file is the
plan agents follow.

## Frontier claim

A phone-captured room, separated into editable objects, with gaps in the capture found
and filled (recaptured or generated, generated parts tagged), rendered **on a standalone
headset** at p99 frame time <= 13.9 ms (72 FPS) while the user rearranges it by voice.
Meta's own on-device splat path renders one splat at a time and advises staying under
150k splats; Hyperscape relies on cloud rendering. This renders a multi-million-splat,
object-separated room locally.

## Build order (29 Oct, replaces the day-by-day lane table)

Milestone names stay as before: core M1-M5, coverage C1-C3, renderer R0-R4, lod L1-L3,
bake B1-B3 (tasks in `docs/HANDOFF.md` section 5). The order follows two facts. Every
lane after M1 needs a real reconstructed room, and the headset is only available on some
days. So the chain capture, M1, M2, M3 comes first, and headset work goes on the days
that have a headset. Nothing is cut. If time runs out, the items at the end of the list
go first.

1. **Thu 29, evening (low usage limits).** Setup on the Mac (Rust, adb, Modal token,
   `.env`), smoke tests, and filming the rooms plus their test walks. If there is time,
   start M1 on room01, because the first Modal image build takes long.
2. **Fri 30 (headset).** R0 on both headsets: the device baseline and the choice of main
   device. M1 on room01 in parallel, with the image fixes the first run needs. Set the
   per-device starting budgets from the R0 logs. By the evening `?scene=room01` should
   show a metric room.
3. **Sat 31 (OJ away, no headset).** Sessions run on their own, all on room01, with
   nothing that needs the headset or OJ.
   - M2: inventory, SAM 3.1, `lift_ids` with labels and the support rule, floor
     alignment.
   - M3: agent tools, server and WebSocket, verifier.
   - C1 on room01.
   - L1 and L2 in the lod lane, B1 in the bake lane.
   - Each session opens a PR. OJ merges on Sunday.
4. **Sun 1 (headset).**
   - R1 to R3 on room01 (per-object trees, baked planes, controller on the walk path).
   - C2, first version: full rerun of M1 on the original plus top-up frames (see below).
   - M4: insert and completion.
5. **Mon 2 (headset).** M5 voice, all scenes end to end, C3 Muse Image repair and tags,
   R4 benchmark logs. Suggested freeze 20:00.
6. **Tue 3 to Wed 4.** Video, README and submission (dates under Ship below). Only fixes,
   unless an item from step 4 or 5 is still open.

### C2 recapture: quality first, then speed

1. **Full rerun (build first).** Add the top-up frames to the original frames and run
   `reconstruct.py` from scratch: MapAnything on all frames, then gsplat from the start.
   A joint solve gives the best poses and the cleanest splats, and it reuses M1 with
   little new code. Cost: one full reconstruction per top-up (minutes, measured in M1).
   Objects (M2) and coverage (C1) are then recomputed.
2. **Fast path (second, for the "minutes, not hours" claim).** MapAnything registers only
   the top-up against the existing cameras, then gsplat fine-tunes the trained scene for a
   few thousand steps. Build it only after the full rerun works. Compare the two on the
   test walk (benchmark 7). The demo uses the fast path only if its PSNR and LPIPS come
   within about 0.5 dB of the full rerun.

### Ship

- Tue 3: record the video, draft submission by 20:00 (last troubleshooting day).
- Wed 4: final cut, README, submit by 20:00, X post.

Before real scenes exist, every lane develops against the synthetic `demo`, `cov_synth`,
`bakedemo` and `bench_*` scenes. `docs/ARCHITECTURE.md` lists a fallback for each part.

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
