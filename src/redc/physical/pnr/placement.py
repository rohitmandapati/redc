"""Baseline deterministic placement: levelized columns of real-sized boxes.

Every instance occupies exactly ``component.dim`` cells from its origin; cells
holding pins are written as :attr:`CellKind.PIN`, the rest as
:attr:`CellKind.COMPONENT`, all owned by the instance id.

Each pin's routing *escape* cell (``Terminal.outward``: one step out of its face)
must stay legal: inside the grid height, not inside any body, and never covered
by a later placement.  Escapes are tracked in a placement-time keep-out map --
NOT written into the grid -- because routing must be able to occupy them.

Algorithm (all ties broken by instance id):

1. **Layers.**  Build a dependency DAG from the nets, cutting every edge out of a
   register (``Register.out`` acts as a source; its ``next``/``enable`` inputs
   are sinks of the logic before it) so sequential loops vanish.  External
   sources (input pads / peripherals, clock, reset) take layer 0, combinational
   cells and registers their longest-path depth, output sinks the final layer,
   and constants sit one layer before their earliest consumer.
2. **Columns.**  Layer ``L`` is a column at increasing ``x``: inputs on the west,
   outputs (e.g. the display peripheral) on the east.  Columns are separated by
   ``layer_gap`` free cells beyond the widest member.
3. **Shelf packing.**  Within a column, instances are ordered by the mean z of
   their already-placed drivers (a one-pass barycentre heuristic that reduces
   crossings) and stacked along ``z`` with ``component_spacing`` free cells,
   centred on ``z = 0``.  ``y`` is ``base_y``, raised for BOTTOM pins so their
   escapes stay at ``y >= 0``.  If a slot is illegal the scan moves one cell in
   ``z`` -- legality is decided by :func:`check_placement`, never by catching
   grid exceptions.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field

from ...parser import CompileError
from ..components import Component, Constant, Face
from ..grid import CellKind, Grid
from ..netlist import ComponentInstance, PhysicalNetlist
from .config import AttemptGeometry, PnRConfig, TraceLevel
from .geometry import Bounds, Coord, coord_list, escape_cell, footprint, pin_cells
from .trace import TraceRecorder

#: Slots tried along z for one instance before placement gives up.
MAX_SLOT_PROBES = 4096


class PlacementError(CompileError):
    """A design that cannot be legally placed under the current geometry."""


def check_placement(
    component: Component,
    origin: Coord,
    grid: Grid,
    escapes: Mapping[Coord, tuple[int, str]],
) -> str | None:
    """Why ``component`` cannot sit at ``origin`` -- or ``None`` if it can.

    Checks the whole body (grid height, span caps, existing occupants, existing
    escape cells) and each of its own pin escapes (inside the grid height and
    span, free, not another pin's escape).  Pins of this component may share an
    escape cell with each other; whether that is electrically fine is decided
    when nets are attached to escapes."""
    dy = component.dim[1]
    if origin[1] < 0 or origin[1] + dy > grid.height:
        return (
            f"body spans y={origin[1]}..{origin[1] + dy - 1}, outside the grid "
            f"(height {grid.height})"
        )
    for cell in footprint(component, origin):
        x, y, z = cell
        if not grid.fits_span(x, z):
            return f"cell {coord_list(cell)} exceeds the grid span limits"
        if not grid.is_free(x, y, z):
            return f"overlaps cell {coord_list(cell)} owned by {grid.owner_at(x, y, z)}"
        if cell in escapes:
            owner, owner_port = escapes[cell]
            return f"would cover the routing escape of instance {owner} pin {owner_port!r}"
    for port in component.ports:
        cell = escape_cell(port, origin)
        x, y, z = cell
        if not 0 <= y < grid.height:
            return (
                f"pin {port.name!r} on its {port.face.name} face would escape to "
                f"y={y}, outside the grid"
            )
        if not grid.fits_span(x, z):
            return f"pin {port.name!r} escape {coord_list(cell)} exceeds the span limits"
        if not grid.is_free(x, y, z):
            return (
                f"pin {port.name!r} escape {coord_list(cell)} is inside instance "
                f"{grid.owner_at(x, y, z)}"
            )
        if cell in escapes:
            owner, other = escapes[cell]
            return (
                f"pin {port.name!r} escape {coord_list(cell)} is already the escape of "
                f"instance {owner} pin {other!r}"
            )
    return None


def place_instance(
    grid: Grid,
    instance: ComponentInstance,
    origin: Coord,
    escapes: dict[Coord, tuple[int, str]],
) -> Bounds:
    """Write ``instance``'s full footprint at ``origin`` and register its pin
    escapes.  The caller must have checked legality first."""
    component = instance.component
    pins = pin_cells(component, origin)
    for cell in footprint(component, origin):
        grid.place(*cell, owner=instance.id, kind=CellKind.PIN if cell in pins else CellKind.COMPONENT)
    for port in component.ports:
        escapes.setdefault(escape_cell(port, origin), (instance.id, port.name))
    instance.origin = origin
    return Bounds.box(origin, component.dim)


@dataclass
class Placement:
    """The result of placing one attempt onto one grid."""

    grid: Grid
    layers: dict[int, int]
    boxes: dict[int, Bounds] = field(default_factory=dict)
    escapes: dict[Coord, tuple[int, str]] = field(default_factory=dict)

    @property
    def bounds(self) -> Bounds | None:
        """Bounding box of every placed body."""
        result: Bounds | None = None
        for box in self.boxes.values():
            result = box.union(result)
        return result

    @property
    def extent(self) -> Bounds | None:
        """Bounding box of every body AND pin escape -- what routing must reach."""
        result = self.bounds
        escapes = Bounds.of(self.escapes)
        if escapes is None:
            return result
        return escapes.union(result)


def placement_layers(netlist: PhysicalNetlist) -> dict[int, int]:
    """Layer (column) index of every instance; see the module docstring."""
    instances = netlist.instances

    def role(inst: ComponentInstance) -> str:
        component = inst.component
        if isinstance(component, Constant):
            return "constant"
        if component.is_source:
            return "source"
        if component.is_sink:
            return "sink"
        return "core"

    roles = {i: role(inst) for i, inst in instances.items()}
    preds: dict[int, set[int]] = {i: set() for i in instances}
    consumers: dict[int, set[int]] = {i: set() for i in instances}
    for net in netlist.nets.values():
        driver = net.driver.instance
        for sink in net.sinks:
            consumers[driver.id].add(sink.instance.id)
            if driver.component.is_stateful or roles[driver.id] == "constant":
                continue  # register outputs are cut points; constants go ALAP
            if roles[sink.instance.id] == "core":
                preds[sink.instance.id].add(driver.id)

    layer = {i: 0 for i, r in roles.items() if r == "source"}
    core = [i for i, r in roles.items() if r == "core"]
    succs: dict[int, list[int]] = {i: [] for i in core}
    indegree = {i: 0 for i in core}
    for i in core:
        for p in preds[i]:
            if roles[p] == "core":
                succs[p].append(i)
                indegree[i] += 1
    ready = [i for i in core if indegree[i] == 0]
    heapq.heapify(ready)
    while ready:
        current = heapq.heappop(ready)
        layer[current] = 1 + max((layer[p] for p in preds[current]), default=0)
        for succ in succs[current]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(ready, succ)
    stuck = sorted(i for i in core if i not in layer)
    if stuck:
        raise PlacementError(f"combinational loop through instances {stuck}")

    last = 1 + max((layer[i] for i in core), default=0)
    for i, r in roles.items():
        if r == "sink":
            layer[i] = last
    for i, r in roles.items():
        if r == "constant":
            uses = [layer[c] for c in consumers[i] if c in layer]
            layer[i] = max(0, min(uses) - 1) if uses else 0
    return layer


def _base_y(component: Component, config: PnRConfig, height: int) -> int:
    """Lowest legal origin y at or above ``base_y`` for this component, lowered
    only if the box would otherwise poke out of the top of the grid."""
    faces = {port.face for port in component.ports}
    floor = 1 if Face.BOTTOM in faces else 0
    ceiling = height - component.dim[1] - (1 if Face.TOP in faces else 0)
    y = min(max(config.base_y, floor), ceiling)
    if y < floor:
        raise PlacementError(
            f"{component.name} (dim {list(component.dim)}) does not fit the grid "
            f"height {height} with legal pin escapes"
        )
    return y


def _drivers(netlist: PhysicalNetlist) -> dict[int, list[ComponentInstance]]:
    """Instance id -> the instances driving its inputs (id order, no repeats)."""
    drivers: dict[int, dict[int, ComponentInstance]] = {i: {} for i in netlist.instances}
    for net in netlist.nets.values():
        for sink in net.sinks:
            if net.driver.instance is not sink.instance:
                drivers[sink.instance.id][net.driver.instance.id] = net.driver.instance
    return {i: [d[k] for k in sorted(d)] for i, d in drivers.items()}


def _driver_z(drivers: list[ComponentInstance]) -> float | None:
    """Mean z-centre of the already-placed drivers."""
    zs = [d.origin[2] + d.component.dim[2] / 2 for d in drivers if d.origin is not None]
    return sum(zs) / len(zs) if zs else None


def place_netlist(
    netlist: PhysicalNetlist,
    grid: Grid,
    geometry: AttemptGeometry,
    config: PnRConfig,
    trace: TraceRecorder | None = None,
) -> Placement:
    """Place every instance of ``netlist`` on ``grid`` (sets ``origin``)."""
    layers = placement_layers(netlist)
    drivers = _drivers(netlist)
    placement = Placement(grid=grid, layers=layers)
    columns: dict[int, list[ComponentInstance]] = {}
    for inst in sorted(netlist.instances.values(), key=lambda i: i.id):
        columns.setdefault(layers[inst.id], []).append(inst)

    if trace:
        trace.emit(
            "placement",
            "placement_begin",
            attempt=geometry.attempt,
            component_spacing=geometry.component_spacing,
            layer_gap=geometry.layer_gap,
            base_y=config.base_y,
            layers=[
                {"layer": layer, "instances": [inst.id for inst in members]}
                for layer, members in sorted(columns.items())
            ],
        )

    x = 0
    last_bounds: Bounds | None = None
    for layer, members in sorted(columns.items()):
        if layer > 0:
            keyed = [(_driver_z(drivers[inst.id]), inst.id, inst) for inst in members]
            keyed.sort(key=lambda k: (k[0] is None, k[0] or 0.0, k[1]))
            members = [k[2] for k in keyed]
        spacing = geometry.component_spacing
        total = sum(inst.component.dim[2] for inst in members) + spacing * (len(members) - 1)
        z = -(total // 2)
        for inst in members:
            component = inst.component
            y = _base_y(component, config, grid.height)
            for _probe in range(MAX_SLOT_PROBES):
                origin = (x, y, z)
                if trace:
                    trace.emit(
                        "placement",
                        "component_place_attempt",
                        level=TraceLevel.DETAILED,
                        instance=inst.id,
                        origin=coord_list(origin),
                        dim=list(component.dim),
                        bounds=Bounds.box(origin, component.dim).to_dict(),
                    )
                reason = check_placement(component, origin, grid, placement.escapes)
                if reason is None:
                    break
                if trace:
                    trace.emit(
                        "placement",
                        "component_place_rejected",
                        level=TraceLevel.DETAILED,
                        instance=inst.id,
                        origin=coord_list(origin),
                        reason=reason,
                    )
                z += 1
            else:
                raise PlacementError(
                    f"no legal slot for instance {inst.id} ({component.name}) in layer {layer}"
                )
            box = place_instance(grid, inst, origin, placement.escapes)
            placement.boxes[inst.id] = box
            if trace:
                trace.emit(
                    "placement",
                    "component_placed",
                    instance=inst.id,
                    layer=layer,
                    origin=coord_list(origin),
                    dim=list(component.dim),
                    bounds=box.to_dict(),
                )
                bounds = placement.bounds
                if bounds is not None and bounds != last_bounds:
                    trace.emit("placement", "design_bounds_changed", **bounds.to_dict())
                    last_bounds = bounds
            z += component.dim[2] + spacing
        x += max(inst.component.dim[0] for inst in members) + geometry.layer_gap

    if trace:
        bounds = placement.bounds
        trace.emit(
            "placement",
            "placement_complete",
            attempt=geometry.attempt,
            component_count=len(placement.boxes),
            component_cells=sum(b.volume for b in placement.boxes.values()),
            bounds=bounds.to_dict() if bounds else None,
        )
    return placement
