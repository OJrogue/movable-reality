"""Agent loop: user text -> Muse Wonder with tools -> scene edits -> viewer screenshot ->
Muse Spark Open verdict -> one retry when the verifier says no (ARCHITECTURE stage 7)."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..clients import meta_async_client
from ..config import Models
from ..scene import Scene
from .llm import cached_chat
from .tools import EDIT_TOOLS, TOOL_SCHEMAS, RoomBounds, describe, run_tool
from .verify import Verdict, verify_edit

Capture = Callable[[], Awaitable[str | None]]  # returns a JPEG data URL, or None
Emit = Callable[[dict[str, Any]], Awaitable[None]]

SYSTEM = """You edit a 3D scan of a real room for the user. Units are metres. World axes: +Y up,
floor at y = 0, x and z span the floor. Objects are identified by ids such as "armchair#2".
Always act through the tools. Never invent ids: pick them from the object list below.

The user sees the room from {eye}, looking along floor direction {forward}. From the user's
point of view, "right" is world direction {right}, "left" is the opposite, "forward"/"away"
is {forward} and "closer"/"towards me" is the opposite. Turning "clockwise" seen from above is
a negative yaw_deg. If the user gives no distance, use 0.5 m. Prefer fits before placing an
object in a tight spot. If the user asks a question (sizes, "does it fit"), answer in metres
from measure or fits. Keep the final reply to one or two short sentences.

Room floor: {room}.
Objects:
{objects}
"""


@dataclass
class CommandResult:
    reply: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    verified: bool | None = None
    verdicts: list[str] = field(default_factory=list)
    edited: bool = False


def view_directions(camera: dict | None) -> tuple[list[float], list[float], list[float]]:
    """Eye, unit floor forward and unit floor right (forward x up) from the viewer camera."""
    eye = list((camera or {}).get("eye") or [0.0, 1.6, 2.5])
    fwd = (camera or {}).get("forward") or [0.0, 0.0, -1.0]
    fx, fz = float(fwd[0]), float(fwd[2])
    n = math.hypot(fx, fz) or 1.0
    fx, fz = fx / n, fz / n
    right = [-fz, 0.0, fx]
    return (
        [round(v, 2) for v in eye],
        [round(fx, 2), 0.0, round(fz, 2)],
        [round(v, 2) for v in right],
    )


def system_prompt(scene: Scene, bounds: RoomBounds | None, camera: dict | None) -> str:
    eye, fwd, right = view_directions(camera)
    objects = "\n".join(json.dumps(describe(o)) for o in scene.objects) or "(none)"
    room = json.dumps(bounds.as_dict()) if bounds else "unknown"
    return SYSTEM.format(eye=eye, forward=fwd, right=right, room=room, objects=objects)


def summarise(steps: list[dict[str, Any]]) -> str:
    edits = [s for s in steps if s["tool"] in EDIT_TOOLS]
    return "; ".join(f"{s['tool']}({json.dumps(s['args'])}) -> {json.dumps(s['result'])[:160]}"
                     for s in edits) or "no edits"  # fmt: skip


async def _noop(_: dict[str, Any]) -> None:
    return None


async def run_command(
    scene: Scene,
    text: str,
    bounds: RoomBounds | None = None,
    camera: dict | None = None,
    capture: Capture | None = None,
    emit: Emit = _noop,
    inserter=None,
    client=None,
    verify: bool = True,
    max_rounds: int = 6,
    retries: int = 1,
) -> CommandResult:
    """Run one user command against `scene` (mutated in place). The caller saves it."""
    client = client or meta_async_client()
    result = CommandResult(reply="")
    before = await capture() if (capture and verify) else None
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(scene, bounds, camera)},
        {"role": "user", "content": text},
    ]
    first_entry = len(scene.history)
    for attempt in range(retries + 1):
        await emit({"stage": "thinking", "text": "planning" if attempt == 0 else "retrying"})
        for _ in range(max_rounds):
            resp = await cached_chat(
                client, "meta", model=Models.AGENT, messages=messages, tools=TOOL_SCHEMAS
            )
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            messages.append(
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
                if calls
                else {"role": "assistant", "content": msg.get("content") or ""}
            )
            if not calls:
                result.reply = (msg.get("content") or "").strip()
                break
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = run_tool(scene, name, args, bounds, instruction=text, inserter=inserter)
                step = {"tool": name, "args": args, "result": out}
                result.steps.append(step)
                if name in EDIT_TOOLS and "error" not in out:
                    result.edited = True
                await emit({"stage": "tool", "text": f"{name} {json.dumps(args)}", "step": step})
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(out)}
                )
        else:
            result.reply = result.reply or "I ran out of steps for that command."

        if not (verify and result.edited and capture and before):
            break
        await emit({"stage": "verify", "text": "checking the edit"})
        after = await capture()
        if not after:
            break
        try:
            verdict = await verify_edit(text, summarise(result.steps), before, after)
        except Exception as exc:  # noqa: BLE001 - a verifier outage must not lose the edit
            verdict = Verdict(ok=None, reason=f"verifier unavailable: {exc}")
            result.verdicts.append(verdict.reason)
            break
        result.verdicts.append(verdict.reason)
        result.verified = verdict.ok
        word = {True: "ok: ", False: "no: ", None: "unverified: "}[verdict.ok]
        await emit({"stage": "verify", "text": word + verdict.reason})
        if verdict.ok is not False or attempt == retries:
            break
        messages.append(
            {
                "role": "user",
                "content": f"A visual check of the viewer says the edit is not right: "
                f"{verdict.reason} Fix it with the tools (undo if needed), then reply.",
            }
        )
    for e in scene.history[first_entry:]:
        e.verified = result.verified
    await emit({"stage": "done", "text": result.reply})
    return result
