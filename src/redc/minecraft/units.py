"""Time units of the RedC redstone model.

Minecraft has two clocks:

* a **game tick** (gt): 1/20 s, the server's update step;
* a **redstone tick** (rt): two game ticks -- the unit repeater delays,
  torch delays and redstone clocks are quoted in.

Everything inside :mod:`redc.minecraft` -- the event queue, abstract component
delays, scheduled stimuli -- runs in GAME TICKS (integers, never fractions).
Reports quote redstone ticks where that is the natural unit; every field name
carries its unit (``*_gt`` / ``*_rt``) and conversions only happen through
the helpers below, never implicitly.
"""

from __future__ import annotations

GAME_TICKS_PER_REDSTONE_TICK = 2


def rt_to_gt(redstone_ticks: int) -> int:
    """Redstone ticks -> game ticks (exact)."""
    return redstone_ticks * GAME_TICKS_PER_REDSTONE_TICK


def gt_to_rt(game_ticks: int) -> int:
    """Game ticks -> redstone ticks; refuses a duration that is not a whole
    number of redstone ticks (use :func:`gt_to_rt_ceil` to round up)."""
    whole, rest = divmod(game_ticks, GAME_TICKS_PER_REDSTONE_TICK)
    if rest:
        raise ValueError(f"{game_ticks} game ticks is not a whole number of redstone ticks")
    return whole


def gt_to_rt_ceil(game_ticks: int) -> int:
    """Game ticks -> the smallest whole number of redstone ticks covering them."""
    return -(-game_ticks // GAME_TICKS_PER_REDSTONE_TICK)


def round_up_to_rt(game_ticks: int) -> int:
    """Round a game-tick duration up to a whole number of redstone ticks (in gt)."""
    return rt_to_gt(gt_to_rt_ceil(game_ticks))


def ticks_record(game_ticks: int | None) -> dict[str, float | int | None]:
    """``{"gt": .., "rt": ..}`` for a report; ``rt`` is fractional only if the
    duration is not a whole number of redstone ticks."""
    if game_ticks is None:
        return {"gt": None, "rt": None}
    rt: float | int = game_ticks / GAME_TICKS_PER_REDSTONE_TICK
    if game_ticks % GAME_TICKS_PER_REDSTONE_TICK == 0:
        rt = game_ticks // GAME_TICKS_PER_REDSTONE_TICK
    return {"gt": game_ticks, "rt": rt}


__all__ = [
    "GAME_TICKS_PER_REDSTONE_TICK",
    "gt_to_rt",
    "gt_to_rt_ceil",
    "round_up_to_rt",
    "rt_to_gt",
    "ticks_record",
]
