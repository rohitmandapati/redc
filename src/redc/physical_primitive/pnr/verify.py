"""Independent verification of a legalized primitive design.

Nothing here trusts the router's working state: every rule of the redstone
model (:mod:`redc.physical_primitive.redstone`) is re-derived from the final
artifacts alone -- the placed cells' absolute voxels / keep-outs / pins and the
realized routes' typed signal blocks, supports and clearances.  A design that
passes has, under the implemented model:

* no component voxel collision and no component in another's keep-out;
* routes that start at their driver pin, reach every sink pin, leave / enter
  pins along the pin facing, and step only by legal redstone moves;
* no block used twice (signal, support, clearance), no route element inside a
  component or keep-out, every signal block supported, every staircase
  clearance still air;
* no unintended electrical adjacency: two signal blocks in each other's
  12-block neighbourhood belong to the same net AND are a parent/child pair of
  its tree (so there are no unresolved crossings and no shorts);
* repeaters only on straight level segments, facing downstream, powered at
  their input; every dust block powered (strength >= 1) and every sink at its
  required strength, recomputed from scratch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...minecraft.timing import REPEATER_DELAYS_RT, repeater_delay_gt
from ...minecraft.units import gt_to_rt
from ..geometry import Coord, coord_list
from ..physical import PrimitivePhysicalNetlist
from ..redstone import (
    MAX_SIGNAL_STRENGTH,
    MIN_SIGNAL_Y,
    SIGNAL_NEIGHBORHOOD,
    ElementKind,
    clearance_of,
    is_move,
    support_of,
)
from .legalize import RealizedRoute
from .routing import RouteRequest


@dataclass(frozen=True)
class Violation:
    kind: str
    message: str
    cells: tuple[Coord, ...] = ()
    nets: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "cells": [coord_list(c) for c in self.cells],
            "nets": list(self.nets),
        }


def verify_design(
    mapped: PrimitivePhysicalNetlist,
    requests: Mapping[int, RouteRequest],
    realized: Mapping[int, RealizedRoute],
    *,
    max_y: int,
    limit: int = 200,
) -> list[Violation]:
    """Every violation found (empty = legal), capped at ``limit``."""
    out: list[Violation] = []

    def bad(kind: str, message: str, cells: tuple[Coord, ...] = (), nets: tuple[int, ...] = ()) -> bool:
        out.append(Violation(kind, message, cells, nets))
        return len(out) >= limit

    # -- components ---------------------------------------------------------------
    body: dict[Coord, int] = {}
    keepout: dict[Coord, int] = {}
    pin_cells: dict[Coord, tuple[int, str]] = {}
    for inst in mapped.instances:
        if not inst.is_placed:
            bad("unplaced", f"primitive {inst.id} is not placed")
            continue
        placed = inst.placed
        for cell in placed.occupied:
            if not 0 <= cell[1] <= max_y:
                bad("height", f"primitive {inst.id} voxel {coord_list(cell)} is outside 0..{max_y}", (cell,))
            if cell in body:
                bad("voxel_collision", f"primitives {body[cell]} and {inst.id} overlap at {coord_list(cell)}", (cell,))
            body[cell] = inst.id
        for cell in placed.keepout:
            keepout.setdefault(cell, inst.id)
        for pin in placed.pins.values():
            pin_cells[pin.position] = (inst.id, pin.name)
    for cell, owner in body.items():
        if (
            cell in keepout
            and keepout[cell] != owner
            and bad("keepout_violation", f"primitive {owner} sits in the keep-out of {keepout[cell]}", (cell,))
        ):
            return out
    for cell, (owner, pin_name) in pin_cells.items():
        if (cell in body or cell in keepout) and bad(
            "pin_blocked", f"pin {pin_name!r} of primitive {owner} at {coord_list(cell)} is covered", (cell,)
        ):
            return out

    # -- routes: structure ------------------------------------------------------
    signal: dict[Coord, tuple[int, ElementKind]] = {}
    support: dict[Coord, int] = {}
    clearance: dict[Coord, set[int]] = {}
    parent_of: dict[Coord, Coord | None] = {}
    #: Nets whose tree or elements are malformed: their per-block electrical
    #: checks are skipped (the structural violation is already reported).
    malformed: set[int] = set()
    for net_id in sorted(set(realized) - set(requests)):
        if bad("unrequested_route", f"realized route for net {net_id}, which is not part of the design",
               nets=(net_id,)):  # fmt: skip
            return out
    for net_id in sorted(requests):
        request = requests[net_id]
        route = realized.get(net_id)
        if route is None:
            if bad("unrouted", f"net {net_id} has no realized route", nets=(net_id,)):
                return out
            continue
        tree = route.tree
        if tree.root != request.driver.cell:
            bad("root", f"net {net_id} does not start at its driver pin", (tree.root,), (net_id,))
        goals = {b.goal for b in tree.branches}
        missing = {s.cell for s in request.sinks} - goals
        if missing:
            bad("unreached_sink", f"net {net_id} misses sinks {sorted(missing)}", tuple(sorted(missing)), (net_id,))
        # Rebuild parent/children from the ordered paths, never trusting them.
        parent: dict[Coord, Coord | None] = {tree.root: None}
        children: dict[Coord, list[Coord]] = {tree.root: []}
        for branch in tree.branches:
            if not branch.path or branch.start not in parent:
                bad("detached_branch", f"net {net_id} branch starts off the tree",
                    tuple(branch.path[:1]), (net_id,))  # fmt: skip
                malformed.add(net_id)
                continue
            for a, b in zip(branch.path, branch.path[1:]):
                if not is_move(a, b):
                    bad("illegal_step", f"net {net_id} jumps {coord_list(a)} -> {coord_list(b)}", (a, b), (net_id,))
                    malformed.add(net_id)
                if b not in parent:
                    parent[b] = a
                    children.setdefault(a, []).append(b)
                    children.setdefault(b, [])
        face = request.driver.facing.vector
        for child in children.get(tree.root, ()):
            if (child[0] - tree.root[0], child[2] - tree.root[2]) != (face[0], face[2]):
                bad("pin_facing", f"net {net_id} leaves its driver pin against its facing", (tree.root,), (net_id,))
        for sink in request.sinks:
            if children.get(sink.cell):
                bad("pin_facing", f"net {net_id} continues past sink pin {sink.pin!r} of {sink.instance}",
                    (sink.cell,), (net_id,))  # fmt: skip
            up = parent.get(sink.cell)
            if up is None:
                continue
            fx, _, fz = sink.facing.vector
            if (sink.cell[0] - up[0], sink.cell[2] - up[2]) != (-fx, -fz):
                bad("pin_facing", f"net {net_id} enters pin {sink.pin!r} of {sink.instance} from the side",
                    (sink.cell,), (net_id,))  # fmt: skip
        coords = [e.coord for e in route.elements]
        if len(set(coords)) != len(coords) or set(coords) != set(parent):
            bad("elements", f"net {net_id} elements differ from its tree", nets=(net_id,))
            malformed.add(net_id)
        for element in route.elements:
            if element.kind not in (ElementKind.DUST, ElementKind.REPEATER):
                bad("element_kind", f"net {net_id} signal block {coord_list(element.coord)} is a "
                    f"{element.kind.value}, not dust or a repeater", (element.coord,), (net_id,))  # fmt: skip
                malformed.add(net_id)
        pins_of_net = {request.driver.cell, *(s.cell for s in request.sinks)}
        for element in route.elements:
            cell = element.coord
            if element.parent != parent.get(cell):
                bad("direction", f"net {net_id} element parent mismatch at {coord_list(cell)}", (cell,), (net_id,))
            if cell in signal and bad(
                "shared_signal",
                f"nets {signal[cell][0]} and {net_id} share {coord_list(cell)}",
                (cell,),
                (signal[cell][0], net_id),
            ):
                return out
            signal[cell] = (net_id, element.kind)
            parent_of[cell] = element.parent
            if cell in pin_cells and cell not in pins_of_net:
                bad("foreign_pin", f"net {net_id} uses pin {pin_cells[cell]} at {coord_list(cell)}", (cell,), (net_id,))
            if cell not in pin_cells and (cell in body or cell in keepout):
                bad("signal_in_component", f"net {net_id} signal inside a component at {coord_list(cell)}",
                    (cell,), (net_id,))  # fmt: skip
            if not MIN_SIGNAL_Y <= cell[1] <= max_y:
                bad("height", f"net {net_id} signal {coord_list(cell)} outside {MIN_SIGNAL_Y}..{max_y}", (cell,), (net_id,))
        for cell in route.supports:
            if cell in support:
                bad("shared_support", f"nets {support[cell]} and {net_id} share support {coord_list(cell)}",
                    (cell,), (support[cell], net_id))  # fmt: skip
            support[cell] = net_id
        for cell in route.clearances:
            clearance.setdefault(cell, set()).add(net_id)
        expected_supports = {support_of(c) for c in parent if c not in pins_of_net}
        if expected_supports != set(route.supports):
            bad("supports", f"net {net_id} supports do not match its signal blocks", nets=(net_id,))
        for cell, up in parent.items():
            if up is not None:
                clear = clearance_of(up, cell)
                if clear is not None and clear not in route.clearances:
                    bad("clearance_missing", f"net {net_id} staircase at {coord_list(cell)} lacks clearance",
                        (cell,), (net_id,))  # fmt: skip
        if len(out) >= limit:
            return out

    # -- routes: global block legality -------------------------------------------
    for cell, owner in support.items():
        if cell in signal:
            bad("support_on_signal", f"support of net {owner} is a signal block {coord_list(cell)}", (cell,), (owner,))
        if cell in body or cell in keepout or cell in pin_cells:
            bad("support_in_component", f"support of net {owner} inside a component at {coord_list(cell)}",
                (cell,), (owner,))  # fmt: skip
    for cell, (owner, _kind) in signal.items():
        below = support_of(cell)
        if cell in pin_cells:
            if below not in body:
                bad("unsupported_pin", f"pin block {coord_list(cell)} does not rest on its cell", (cell,), (owner,))
        elif support.get(below) != owner:
            bad("unsupported", f"net {owner} signal {coord_list(cell)} has no support", (cell,), (owner,))
    for cell, nets in clearance.items():
        if cell in signal or cell in support or cell in body or cell in pin_cells:
            bad("clearance_blocked", f"clearance {coord_list(cell)} of nets {sorted(nets)} is not air",
                (cell,), tuple(sorted(nets)))  # fmt: skip
    if len(out) >= limit:
        return out

    # -- electrical adjacency: the realized graph must be exactly the trees -------
    for cell, (owner, _kind) in signal.items():
        x, y, z = cell
        for dx, dy, dz in SIGNAL_NEIGHBORHOOD:
            nb = (x + dx, y + dy, z + dz)
            other = signal.get(nb)
            if other is None or nb < cell:
                continue
            if other[0] != owner:
                message = f"nets {owner} and {other[0]} touch at {coord_list(cell)} / {coord_list(nb)}"
                if bad("short", message, (cell, nb), (owner, other[0])):
                    return out
            elif parent_of.get(nb) != cell and parent_of.get(cell) != nb:
                message = f"net {owner} touches itself off-tree at {coord_list(cell)} / {coord_list(nb)}"
                if bad("self_loop", message, (cell, nb), (owner,)):
                    return out

    # -- signal strength, recomputed ------------------------------------------------
    for net_id in sorted(realized):
        route = realized[net_id]
        if net_id not in requests or net_id in malformed:
            continue
        request = requests[net_id]
        tree = route.tree
        kinds = {e.coord: e for e in route.elements}
        drive = request.driver.strength
        if route.powered != (drive > 0):
            bad("powered_record", f"net {net_id} records powered={route.powered} but its driver drives {drive}",
                nets=(net_id,))  # fmt: skip
        # Dust: its strength.  Repeater: its INPUT strength (the block behind).
        # A never-driven net (constant 0) must record strength 0 everywhere.
        strength: dict[Coord, int] = {}
        delay: dict[Coord, int] = {}
        for cell in tree.cells:
            up = tree.parent[cell]
            element = kinds[cell]
            if up is None:
                delay[cell] = 0
            elif kinds[up].kind is ElementKind.REPEATER and kinds[up].setting in REPEATER_DELAYS_RT:
                delay[cell] = delay[up] + gt_to_rt(repeater_delay_gt(kinds[up].setting or 1))
            else:
                delay[cell] = delay[up]
            if (element.kind is ElementKind.REPEATER) != (element.setting is not None) or (
                element.setting is not None and element.setting not in REPEATER_DELAYS_RT
            ):
                bad("repeater_setting", f"net {net_id} block {coord_list(cell)} has repeater delay "
                    f"{element.setting!r} (a repeater needs 1..4 rt, dust none)", (cell,), (net_id,))  # fmt: skip
            if element.delay != delay[cell]:
                bad("delay_record", f"net {net_id} records delay {element.delay} at {coord_list(cell)}, "
                    f"actual {delay[cell]}", (cell,), (net_id,))  # fmt: skip
            if up is None:
                strength[cell] = drive
            elif kinds[up].kind is ElementKind.REPEATER:
                strength[cell] = MAX_SIGNAL_STRENGTH if strength[up] >= 1 else 0
            elif element.kind is ElementKind.REPEATER:
                strength[cell] = strength[up]
            else:
                strength[cell] = max(0, strength[up] - 1)
            if element.kind is ElementKind.REPEATER:
                kids = tree.children.get(cell, [])
                ok = (
                    up is not None
                    and len(kids) == 1
                    and cell not in pin_cells
                    and up[1] == cell[1] == kids[0][1]
                    and (cell[0] - up[0], cell[2] - up[2]) == (kids[0][0] - cell[0], kids[0][2] - cell[2])
                    and element.facing is not None
                    and element.facing.vector == (kids[0][0] - cell[0], 0, kids[0][2] - cell[2])
                )
                if not ok:
                    bad("repeater", f"net {net_id} repeater at {coord_list(cell)} is not on a straight level run",
                        (cell,), (net_id,))  # fmt: skip
                if drive > 0 and strength[cell] < 1:
                    bad("repeater_unpowered", f"net {net_id} repeater at {coord_list(cell)} gets no input",
                        (cell,), (net_id,))  # fmt: skip
            elif drive > 0 and strength[cell] < 1:
                bad("weak_signal", f"net {net_id} dust at {coord_list(cell)} is unpowered", (cell,), (net_id,))
            if element.strength != strength[cell]:
                bad("strength_record", f"net {net_id} records strength {element.strength} at {coord_list(cell)}, "
                    f"actual {strength[cell]}", (cell,), (net_id,))  # fmt: skip
        if drive > 0:
            for sink in request.sinks:
                if strength.get(sink.cell, 0) < sink.strength:
                    bad("weak_sink", f"net {net_id} reaches pin {sink.pin!r} of {sink.instance} at strength "
                        f"{strength.get(sink.cell, 0)} < {sink.strength}", (sink.cell,), (net_id,))  # fmt: skip
        # The per-sink electrical summary (exported, and feeding metrics/timing).
        reports = {(r.sink.instance, r.sink.pin): r for r in route.sinks}
        if len(reports) != len(route.sinks) or set(reports) != {(s.instance, s.pin) for s in request.sinks}:
            bad("sink_record", f"net {net_id} sink reports do not match its sinks", nets=(net_id,))
        for sink in request.sinks:
            report = reports.get((sink.instance, sink.pin))
            if report is None or sink.cell not in tree.parent:
                continue
            hops, repeaters, walk = 0, 0, tree.parent[sink.cell]
            while walk is not None:
                hops += 1
                repeaters += kinds[walk].kind is ElementKind.REPEATER
                walk = tree.parent[walk]
            actual = (strength[sink.cell], sink.strength, repeaters, delay[sink.cell], hops)
            recorded = (report.strength, report.required, report.repeaters, report.delay_ticks, report.distance)
            if recorded != actual:
                bad("sink_record", f"net {net_id} pin {sink.pin!r} of {sink.instance} records "
                    f"(strength, required, repeaters, delay, distance) = {recorded}, actual {actual}",
                    (sink.cell,), (net_id,))  # fmt: skip
        if len(out) >= limit:
            return out
    return out


__all__ = ["Violation", "verify_design"]
