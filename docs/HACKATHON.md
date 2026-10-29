# Hackathon reference

Source: Cerebral Valley participant guide and event waiver (read 26 Oct 2026), plus
Cerebral Valley's answer in Discord on 28 Oct (runtime models). If this file and the
official pages disagree, the official pages win.

## Key facts

- Event: Meta Global AI Developer Hackathon, Indie Developer track, theme Push the Frontier.
- Team: Gaussian Movers (solo, OJ is the Representative who submits).
- Building window: 26 Oct to 4 Nov 2026.
- Submission deadline: Wed 4 Nov 21:00 ET = **Thu 5 Nov 03:00 Prague**. No late entries.
- Judging 5 to 8 Nov. Closing ceremony Mon 9 Nov 18:00 Prague (top 5 per track, live final
  round on the videos). Community vote winner 16 Nov.
- Office hours: Discord #office-hours, weekdays 10:00-11:00 ET
  (15:00-16:00 Prague until 30 Oct, 16:00-17:00 from 2 Nov). Text only.
- Workshops 12:00 ET: Tue 27 Agents 101, Wed 28 Agents 102 (verifiable agents),
  Fri 30 Fireworks / Muse open models.
- Last day for submission troubleshooting: Tue 3 Nov. Plan: draft submission by Tue 3 Nov
  20:00, final by Wed 4 Nov 20:00.
- Feature freeze: not a rule. `docs/PLAN.md` suggests Mon 2 Nov 20:00.

## Rules

- Repository must be public.
- Max 4 per team, solo allowed. Every member must have been accepted individually.
- New work only. The demo must show only what was built during the event, and judges must
  be able to tell what was built. Failing this is disqualification.
- Must use Meta models: Muse Spark series, Muse Wonder, Muse Image, Muse Voice Transcribe,
  SAM 3.1 via Meta Model API; open weights Muse Glimmer and Muse Spark Open; Muse Code.
- **Every model the project calls at runtime must be a Meta model** (Cerebral Valley in
  Discord, 28 Oct). This replaced earlier guidance that non-Meta models could run alongside
  Meta's, because Use of Meta AI Tooling is scored and Stage 1 checks it. Meta models served
  through Baseten, Modal or Fireworks, and open-source Meta models (Llama, SAM, Muse Glimmer,
  Muse Spark Open) count however they are run. Any coding tool (Claude Code, Gemini, Muse
  Code) may be used to write the code.
- No code, data or assets without rights. Avoid the listed generic project types.

## Judging

1. Stage 1, pass/fail: fits the theme and reasonably uses Meta Model API and/or Meta
   open-weight models.
2. Stage 2, scored 0-10 each: Impact, Use of Meta AI Tooling, Technical Depth & Execution,
   Demo. The guide says equally weighted; the waiver says 30 / 25 / 20 / 25. Ask in
   #questions if it matters.
3. Stage 3: top 5 per track, final judges watch the videos live and rank them.

## Prizes

Indie: 150k / 100k / 75k / 50k / 25k USD. Bonus (50k total):

| Bonus | Amount | Our angle |
|---|---|---|
| Most Viral (X engagement until 8 Nov EOD, link submitted) | 20k | Post the video 4 Nov |
| Best Build on Meta Open Weight Models (Glimmer / Spark Open) | 10k | Spark Open verifier on Baseten; also MapAnything and SAM 3D Objects (ask whether they count for this bonus) |
| Community Vote (gallery, 1 per account) | 10k | Share gallery link |
| Best Agentic Use-Case | 5k | Self-verifying edit agent |
| Best Creative Build (Muse Image or Wonder) | 5k | Insert furniture from text |

## Credits

| Provider | Status | Notes |
|---|---|---|
| Meta Model API | $150, approved | Card required on account; set a spend limit. Claim by 4 Nov |
| Baseten | $100, approved | Muse Spark Open, slug `meta-models/muse-spark-open-1026` |
| Modal | approved (29 Oct) | GPU jobs: reconstruction, coverage, baking, SAM 3D Objects |
| Fireworks | approved (29 Oct) | Optional serving for Meta open-weight models |

## Submission checklist

- [ ] Draft submission by Tue 3 Nov 20:00
- [ ] Public GitHub repo, README states what was built during the event
- [ ] 3 min demo video, public link, shows only in-event work
- [ ] Headset footage with the frame-time HUD; coverage clip; proof-toggle clip
- [ ] Raw frame-time logs in `benchmarks/`
- [ ] Every runtime model listed in the README is a Meta model
- [ ] Meta models named in README and video, with where they are used
- [ ] Live demo link (hosted viewer with precomputed scenes) works logged out
- [ ] Team name, project name, all members listed on the submission page
- [ ] X post with the video, link submitted for Most Viral
- [ ] Open-weight usage (Spark Open on Baseten) stated for the bonus flag

## Links

- Event: https://cerebralvalley.ai/e/meta-ai-developer-hackathon
- Participant guide: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details
- Submit: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit
- Discord: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/discord
- Livestream: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/live
