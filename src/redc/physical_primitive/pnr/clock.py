"""Physical clock-tree analysis and deterministic balancing.

The clock is an ordinary routed one-bit net (``role == "clock"``) from the
CLOCK_SOURCE pin to every REGISTER_BIT ``clk`` pin.  Nothing assumes it is
ideal: the arrival at a sink is the sum of the repeater delays on its tree
path (:func:`~redc.minecraft.timing.repeater_delay_gt`, the definition the
simulator and STA use; dust adds nothing).

Balancing makes every sink see the edge at the SAME modelled time (skew 0)
by adding PHYSICAL delay only:

* raising an existing repeater's delay setting (up to 4 redstone ticks), or
* turning a dust block into a repeater -- only on a repeater-eligible block
  (straight, level, one child, not a pin: the same rule legalization uses).

Algorithm (deterministic, tree-aware).  Target ``T`` = the latest sink
arrival; every sink ``s`` needs ``deficit(s) = T - arrival(s)`` more delay.
Walk the tree root-first; at each site ``v`` add

    x(v) = min(capacity(v), min over sinks s below v of deficit(s) - added above v)

i.e. as much as every sink below can still absorb, as HIGH in the tree as
possible: a repeater on a shared trunk delays every downstream sink, so delay
goes on a trunk only when all its sinks need it, and the latest sink's path
(deficit 0) is never touched.  Putting delay as high as possible is optimal
(any solution placing less at ``v`` must place at least that much on every
path below it), so if this greedy pass leaves a sink short, no assignment of
the available sites balances the tree: that is a ``timing/clock_balance``
failure and the attempt may retry with more routing room.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...minecraft.timing import MAX_REPEATER_DELAY_RT, repeater_delay_gt
from ...minecraft.units import gt_to_rt, ticks_record
from ..geometry import Coord, coord_list
from ..redstone import ElementKind
from .legalize import RealizedRoute, realize_route, repeater_eligible
from .routing import RouteRequest


@dataclass(frozen=True)
class ClockSinkArrival:
    instance: int
    pin: str
    coord: Coord
    arrival_rt: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "pin": self.pin,
            "coord": coord_list(self.coord),
            "arrival": ticks_record(2 * self.arrival_rt),
        }


@dataclass
class ClockBalance:
    """The result of balancing one clock route."""

    net: int
    before: list[ClockSinkArrival]
    after: list[ClockSinkArrival]
    route: RealizedRoute
    #: One record per changed block: raised repeater or new repeater.
    changes: list[dict[str, Any]] = field(default_factory=list)
    success: bool = True
    message: str = ""

    @staticmethod
    def _skew(arrivals: list[ClockSinkArrival]) -> int:
        values = [a.arrival_rt for a in arrivals]
        return (max(values) - min(values)) if values else 0

    @property
    def skew_before_rt(self) -> int:
        return self._skew(self.before)

    @property
    def skew_after_rt(self) -> int:
        return self._skew(self.after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "net": self.net,
            "success": self.success,
            "message": self.message,
            "sinks": len(self.before),
            "skew_before": ticks_record(2 * self.skew_before_rt),
            "skew_after": ticks_record(2 * self.skew_after_rt),
            "arrival_max": ticks_record(2 * max((a.arrival_rt for a in self.after), default=0)),
            "changes": self.changes,
            "repeaters_added": sum(1 for c in self.changes if c["change"] == "dust_to_repeater"),
            "repeaters_raised": sum(1 for c in self.changes if c["change"] == "raise_delay"),
            "delay_added_rt": sum(c["added_rt"] for c in self.changes),
        }


def _settings(route: RealizedRoute) -> dict[Coord, int]:
    return {e.coord: e.setting or 1 for e in route.elements if e.kind is ElementKind.REPEATER}


def sink_arrivals(route: RealizedRoute, request: RouteRequest) -> list[ClockSinkArrival]:
    """Clock arrival at every sink pin: the repeater delays on its tree path."""
    settings = _settings(route)
    tree = route.tree
    result = []
    for sink in request.sinks:
        total, cursor = 0, tree.parent.get(sink.cell)
        while cursor is not None:
            if cursor in settings:
                total += gt_to_rt(repeater_delay_gt(settings[cursor]))
            cursor = tree.parent[cursor]
        result.append(ClockSinkArrival(sink.instance, sink.pin, sink.cell, total))
    return result


def balance_clock_route(route: RealizedRoute, request: RouteRequest, *, max_skew_rt: int = 0) -> ClockBalance:
    """Equalize the clock arrival at every sink with physical repeater delay
    (see the module docstring).  ``max_skew_rt`` > 0 accepts an already
    sufficiently balanced tree unchanged."""
    before = sink_arrivals(route, request)
    result = ClockBalance(route.net, before, before, route)
    if not before or ClockBalance._skew(before) <= max_skew_rt:
        return result
    tree = route.tree
    settings = _settings(route)
    target = max(a.arrival_rt for a in before)
    deficit = {a.coord: target - a.arrival_rt for a in before}
    # Smallest deficit among the sinks below every block (post-order).
    below: dict[Coord, int] = {}
    for cell in reversed(tree.cells):
        values = [below[c] for c in tree.children.get(cell, ())]
        if cell in deficit:
            values.append(deficit[cell])
        below[cell] = min(values) if values else 1 << 30
    added_above: dict[Coord, int] = {}
    new_settings = dict(settings)
    changes: list[dict[str, Any]] = []
    for cell in tree.cells:  # root first
        up = tree.parent[cell]
        added = 0 if up is None else added_above[up]
        room = below[cell] - added
        if room > 0 and below[cell] < 1 << 30:
            if cell in settings:
                x = min(MAX_REPEATER_DELAY_RT - settings[cell], room)
                if x > 0:
                    new_settings[cell] = settings[cell] + x
                    changes.append({"coord": coord_list(cell), "change": "raise_delay", "from_rt": settings[cell],
                                    "to_rt": settings[cell] + x, "added_rt": x})  # fmt: skip
                    added += x
            elif repeater_eligible(tree, cell):
                x = min(MAX_REPEATER_DELAY_RT, room)
                new_settings[cell] = x
                changes.append({"coord": coord_list(cell), "change": "dust_to_repeater", "from_rt": 0, "to_rt": x,
                                "added_rt": x})  # fmt: skip
                added += x
        added_above[cell] = added
    balanced = realize_route(tree, request, new_settings)
    after = sink_arrivals(balanced, request)
    result = ClockBalance(route.net, before, after, balanced, changes)
    short = [a for a in after if a.arrival_rt != target]
    if short:
        result.success = False
        result.route = route
        worst = min(short, key=lambda a: (a.arrival_rt, a.instance))
        result.message = (
            f"{len(short)} clock sink(s) cannot be delayed to {target} rt with the available repeater sites "
            f"(e.g. the clk pin of instance {worst.instance} reaches only {worst.arrival_rt} rt)"
        )
    return result


__all__ = ["ClockBalance", "ClockSinkArrival", "balance_clock_route", "sink_arrivals"]
