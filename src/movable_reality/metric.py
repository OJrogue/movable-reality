"""Metric scale without a tape measure: typical object heights from Muse Spark.

MapAnything's scale came out 1.7x to 2.7x too small on the synthetic rooms (31 Oct), and the
ceiling rule needs a reconstructed ceiling plus a measured height. Most rooms hold several
pieces of furniture whose heights vary little (a sofa is about 0.85 m tall, a coffee table
about 0.45 m). Height above the floor is also the most robust thing the splats measure: pose
error smears an object's footprint, but its top stays at a clear height above y = 0.

For every scene object that stands on the floor, Muse Spark gives a typical height and a
plausible range for its noun. Each object votes for scale = typical / measured, weighted by
how narrow its range is. The weighted median of the log ratios gives a consensus. Objects
whose height at that scale falls outside their range are dropped as outliers (often a SAM
label that covered two things), and the weighted mean of the rest is the scale. On
blender01 (1 Nov) it came out 4.9% above the ground-truth Sim(3) scale; the raw scale was
50% too small.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .agent.llm import cached_chat
from .clients import meta_async_client
from .config import Models

HEIGHT_PROMPT = (
    "Typical size of a {noun!r} in a home. Reply with JSON only: "
    '{{"height_m": <typical height in metres>, "min_m": <smallest common height>, '
    '"max_m": <largest common height>, "on_floor": <true if it usually stands directly on '
    "the floor, false if it usually sits on furniture or hangs on a wall>}}."
)


@dataclass
class HeightPrior:
    height_m: float
    min_m: float
    max_m: float
    on_floor: bool


@dataclass
class ScaleVote:
    object_id: str
    noun: str
    measured: float  # top above the floor, recon units after floor alignment
    prior: float
    ratio: float
    weight: float
    used: bool
    note: str = ""


def parse_height(text: str) -> HeightPrior:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in height reply: {text[:200]!r}")
    d = json.loads(m.group(0))
    h, lo, hi = (float(d[k]) for k in ("height_m", "min_m", "max_m"))
    lo, hi = min(lo, h), max(hi, h)
    if not 0.02 <= lo <= hi <= 5.0:
        raise ValueError(f"implausible height range {lo}..{hi}")
    return HeightPrior(h, lo, hi, bool(d.get("on_floor", True)))


async def height_priors(nouns: list[str]) -> dict[str, HeightPrior]:
    """One cached Muse Spark call per noun, three at a time. Nouns that fail are left out."""
    sem = asyncio.Semaphore(3)

    async def one(noun: str):
        async with sem:
            try:
                data = await cached_chat(
                    meta_async_client(),
                    "meta",
                    model=Models.FAST,
                    messages=[{"role": "user", "content": HEIGHT_PROMPT.format(noun=noun)}],
                    temperature=0,
                )
                return noun, parse_height(data["choices"][0]["message"]["content"])
            except Exception as exc:  # noqa: BLE001 - one bad noun must not stop the lift
                print(f"  height prior for {noun!r} failed: {exc}")
                return noun, None

    pairs = await asyncio.gather(*(one(n) for n in sorted(set(nouns))))
    return {n: p for n, p in pairs if p is not None}


def object_height(world_y: np.ndarray) -> tuple[float, float]:
    """Base and top of an object above the floor (2nd and 98th percentile, floaters ignored)."""
    lo, hi = np.percentile(world_y, [2, 98])
    return float(lo), float(hi)


def scale_from_heights(
    measured: dict[str, tuple[str, float, float]],
    priors: dict[str, HeightPrior],
    floor_tol: float = 0.25,
) -> tuple[float | None, list[ScaleVote]]:
    """Metric scale from {object_id: (noun, base, top)} and the noun priors: a weighted
    median finds the consensus, votes outside their prior range at it are dropped, and the
    weighted mean of the remaining log ratios is the answer.

    An object counts when its prior says it stands on the floor and its base is within
    `floor_tol` of its own height from y = 0 (so a vase on a table does not vote). Weight is
    1 / log(max/min)^2, split among the instances of a noun: a sofa (0.7-1.0 m) counts far
    more than a potted plant (0.2-2 m), and six chairs together count as one.
    Returns (None, votes) when fewer than two objects can vote.
    """
    votes: list[ScaleVote] = []
    for oid, (noun, base, top) in measured.items():
        p = priors.get(noun)
        if p is None or top <= 0:
            continue
        ratio = p.height_m / top
        spread = max(np.log(p.max_m / p.min_m), 0.05)
        v = ScaleVote(oid, noun, round(top, 4), p.height_m, round(ratio, 4),
                      round(1 / spread**2, 3), True)  # fmt: skip
        if not p.on_floor:
            v.used, v.note = False, "not a floor object"
        elif base > floor_tol * top:
            v.used, v.note = False, f"base {base:.2f} above the floor"
        votes.append(v)
    used = [v for v in votes if v.used]
    if len(used) < 2:
        return None, votes
    # Six chairs share one prior and so one prior error: together they weigh as one object.
    per_noun = Counter(v.noun for v in used)
    for v in used:
        v.weight = round(v.weight / per_noun[v.noun], 3)
    s = _weighted_median(np.log([v.ratio for v in used]), np.array([v.weight for v in used]))
    # Second pass: drop votes whose implied height is outside the prior range at scale s.
    for v in used:
        p = priors[v.noun]
        h = v.measured * float(np.exp(s))
        if not p.min_m * 0.8 <= h <= p.max_m * 1.25:
            v.used, v.note = False, f"{h:.2f} m at the consensus scale, outside {p.min_m}-{p.max_m}"
    # Final estimate: weighted mean of the log ratios that passed (inverse-variance, if the
    # priors are log-normal with a width set by their range). The median picks one prior's
    # error, the mean averages them: on blender01 +9.5% against +5% from the GT Sim(3).
    kept = [v for v in used if v.used]
    if len(kept) >= 2:
        w = np.array([v.weight for v in kept])
        s = float(np.sum(w * np.log([v.ratio for v in kept])) / np.sum(w))
    return float(np.exp(s)), votes


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    x, w = x[order], w[order]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def votes_json(votes: list[ScaleVote]) -> list[dict]:
    return [asdict(v) for v in votes]
