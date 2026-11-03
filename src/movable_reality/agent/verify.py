"""Visual verifier: Muse Spark Open on Baseten looks at the viewer before and after an edit and
answers whether the edit did what the user asked (ARCHITECTURE stage 7)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..clients import baseten_async_client
from ..config import Models
from ..scene import UNDOABLE_OPS, HistoryEntry, Scene
from .llm import cached_chat
from .tools import RoomBounds

PROMPT = (
    "You check edits to a 3D room viewer. The first image is the view before the edit, the "
    'second after it. The user asked: "{instruction}". The edit tool reported: {summary}. '
    "Does the second image show that request carried out (objects moved, turned, removed or "
    "added as asked) with no obvious damage such as a missing or floating object? A move "
    "that stops short because the object reached another object or a wall counts as carried "
    "out; an object pushed into another object counts as damage. The room is a 3D Gaussian "
    "splat scan: blur, ghosting, smears, holes and dark patches, most of all at the bottom "
    "of an object and where it stood before, are normal and are not damage. Judge only "
    "where the object is, how it is turned and whether it is there. Offsets in the tool "
    "report are world axes, not directions in the image, and a small move can be hard to see: "
    "answer NO only when the edit is clearly wrong. Answer YES or NO on the first line, then "
    "one short sentence why."
)


def edit_focus(scene: Scene, name: str, args: dict) -> tuple[list[float], float] | None:
    """Point and radius (metres) the verifier's screenshots must show for one edit tool call:
    the object, plus where the edit takes it. None when the call edits nothing or its
    arguments are bad (run_tool reports those to the model)."""
    try:
        return _edit_focus(scene, name, args)
    except (KeyError, TypeError, ValueError):
        return None


def _edit_focus(scene: Scene, name: str, args: dict) -> tuple[list[float], float] | None:
    if name == "add_from_library" and args.get("on_object_id"):
        mn, mx = scene.get(args["on_object_id"]).world_aabb()
        return [round((a + b) / 2, 3) for a, b in zip(mn, mx, strict=True)], 0.8
    if name in ("insert_object", "add_from_library"):
        return [float(args["x"]), 0.4, float(args["z"])], 0.8
    entry = _undo_entry(scene) if name == "undo" else None
    oid = entry.object_id if entry else args.get("object_id")
    if not oid:
        return None
    obj = scene.get(oid)
    mn, mx = obj.world_aabb()
    c = [(a + b) / 2 for a, b in zip(mn, mx, strict=True)]
    r = 0.5 * math.hypot(mx[0] - mn[0], mx[2] - mn[2])
    dest = None
    if name == "move_object":
        dest = (c[0] + float(args.get("dx", 0.0)), c[2] + float(args.get("dz", 0.0)))
    elif name == "place_object":
        dest = (float(args["x"]), float(args["z"]))
    elif entry and entry.op == "move":
        dest = (c[0] - float(entry.args["dx"]), c[2] - float(entry.args["dz"]))
    elif entry and entry.op == "place":
        t, prev = obj.transform.translation, entry.args["prev"]
        dest = (c[0] + prev[0] - t[0], c[2] + prev[2] - t[2])
    if dest:
        half = 0.5 * math.hypot(dest[0] - c[0], dest[1] - c[2])
        c[0], c[2] = (c[0] + dest[0]) / 2, (c[2] + dest[1]) / 2
        r += half
    return [round(v, 3) for v in c], max(r, 0.3)


def _undo_entry(scene: Scene) -> HistoryEntry | None:
    """The edit that undo would revert (same rule as Scene.undo)."""
    undone = {e.args.get("index") for e in scene.history if e.op == "undo"}
    for i in range(len(scene.history) - 1, -1, -1):
        e = scene.history[i]
        if e.op in UNDOABLE_OPS and i not in undone and e.object_id:
            return e
    return None


def verify_camera(
    focus: list[float], radius: float, bounds: RoomBounds | None, user: dict | None
) -> dict[str, list[float]]:
    """A viewpoint that looks down at `focus` from the user's side. Screenshots of the user's
    own view often missed the edited object, and the verifier then rejected correct edits
    (blender06, 3 Nov). The eye sits on the line from the object towards the user, never
    further than the user stands: the user is inside the room, while the room box from the
    splats also holds the view through the windows, so a box-clamped eye could end up outside.
    """
    dist = min(max(1.4 + 2.0 * radius, 1.8), 5.0)
    reach = dist * 0.82
    eye_u = (user or {}).get("eye")
    if eye_u:
        dx, dz = float(eye_u[0]) - focus[0], float(eye_u[2]) - focus[2]
        n = math.hypot(dx, dz)
        if n < 0.3:  # standing on it: look from where the user faces it
            fwd = (user or {}).get("forward") or [0.0, 0.0, -1.0]
            dx, dz, n = -float(fwd[0]), -float(fwd[2]), math.hypot(fwd[0], fwd[2]) or 1.0
            reach = 1.2
        else:
            reach = min(reach, max(n, 1.0))
        ux, uz = dx / n, dz / n
    else:
        ux, uz = 0.0, 1.0
    ex, ez = focus[0] + ux * reach, focus[2] + uz * reach
    ey = max(focus[1], 0.3) + 0.4 + 0.6 * reach
    if bounds:
        ex = min(max(ex, bounds.x[0] + 0.3), bounds.x[1] - 0.3)
        ez = min(max(ez, bounds.z[0] + 0.3), bounds.z[1] - 0.3)
        ey = min(ey, bounds.height - 0.3)
    ey = min(ey, 2.4)  # below most ceilings, even when the room box overshoots
    return {"eye": [round(v, 3) for v in (ex, ey, ez)], "target": [round(v, 3) for v in focus]}


@dataclass
class Verdict:
    ok: bool | None  # None: no usable answer, so the edit stays unverified
    reason: str


def parse_verdict(text: str) -> Verdict:
    first = text.strip().splitlines()[0] if text.strip() else ""
    if re.match(r"^\W*yes\b", first, re.IGNORECASE):
        ok: bool | None = True
    elif re.match(r"^\W*no\b", first, re.IGNORECASE):
        ok = False
    else:
        ok = None
    rest = text.strip()[len(first) :].strip() or first
    return Verdict(ok, rest[:300])


async def verify_edit(
    instruction: str, summary: str, before_jpeg: str, after_jpeg: str, client=None
) -> Verdict:
    """before/after are data URLs (data:image/jpeg;base64,...) from the viewer's mrCapture()."""
    client = client or baseten_async_client()
    resp = await cached_chat(
        client,
        "baseten",
        model=Models.VERIFIER_OPEN,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT.format(instruction=instruction, summary=summary),
                    },
                    {"type": "image_url", "image_url": {"url": before_jpeg}},
                    {"type": "image_url", "image_url": {"url": after_jpeg}},
                ],
            }
        ],
        max_tokens=4000,  # a reasoning model: its thinking counts (2000 ran out, 1 Nov)
    )
    msg = resp["choices"][0]["message"]
    verdict = parse_verdict(msg.get("content") or "")
    if verdict.ok is None and not verdict.reason:
        verdict.reason = f"no answer ({resp['choices'][0].get('finish_reason')})"
    return verdict
