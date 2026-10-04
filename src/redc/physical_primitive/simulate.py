"""Functional simulation of a one-bit :class:`PrimitiveNetlist` -- the
correctness oracle for primitive synthesis.

Logical synthesis must be proven against the IR before any geometry exists:

    Graph.evaluate(inputs)             == PrimitiveSimulator.evaluate(inputs)
    Graph.reset_state / step / run     == PrimitiveSimulator.reset_state / step / run

The simulator compiles the netlist once into a topologically ordered program
over signal *slots* (one per driver terminal).  Evaluation is **bit-parallel**:
every slot holds a Python ``int`` whose bit ``k`` is the signal's value in test
vector ``k``, so AND / OR / XOR / NOT are single big-integer operations and
:meth:`PrimitiveSimulator.evaluate_many` checks thousands of input vectors in
one pass (exhaustive 8-bit operand sweeps take milliseconds).

Ports are addressed by their IR names (``in_a``, ``start``, ``result``,
``done``) through the netlist's :class:`~redc.physical_primitive.netlist.LogicalPort`
records, so pads and peripheral realizations simulate identically.  Outputs are
returned with the same signed interpretation :meth:`redc.ir.Graph.evaluate`
uses.  Register semantics match the coarse simulator: while ``rst`` is high a
register bit loads its ``init``; otherwise each clock edge latches ``d`` (the
IR's enable is already an ordinary mux in front of ``d``).
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence

from ..ir import IRType
from ..parser import CompileError
from .netlist import BitTerminal, PrimitiveKind, PrimitiveNetlist

_AND, _OR, _XOR, _NOT = 0, 1, 2, 3
_OPCODES = {
    PrimitiveKind.AND: _AND,
    PrimitiveKind.OR: _OR,
    PrimitiveKind.XOR: _XOR,
    PrimitiveKind.NOT: _NOT,
}


class PrimitiveSimulator:
    """A compiled, reusable simulation of one primitive netlist."""

    def __init__(self, netlist: PrimitiveNetlist) -> None:
        self.netlist = netlist
        slot: dict[BitTerminal, int] = {}
        for inst in netlist.instances:
            for pin in inst.outputs:
                slot[BitTerminal(inst.id, pin)] = len(slot)
        self._slot = slot
        self.slots = len(slot)

        def driver_slot(sink: BitTerminal) -> int:
            driver = netlist.driver_of(sink)
            if driver is None:
                raise CompileError(f"simulation: {sink.describe()} is undriven")
            return slot[driver]

        self._const1: list[int] = []
        self._const0: list[int] = []
        self._clock: list[int] = []
        self._reset: list[int] = []
        # Register bits: (instance id, q slot, d driver slot, rst driver slot, init).
        self._registers: list[tuple[int, int, int, int, int]] = []
        gates = []
        for inst in netlist.instances:
            kind = inst.kind
            if kind is PrimitiveKind.CONST1:
                self._const1.append(slot[BitTerminal(inst.id, "y")])
            elif kind is PrimitiveKind.CONST0:
                self._const0.append(slot[BitTerminal(inst.id, "y")])
            elif kind is PrimitiveKind.CLOCK_SOURCE:
                self._clock.append(slot[BitTerminal(inst.id, "clk")])
            elif kind is PrimitiveKind.RESET_SOURCE:
                self._reset.append(slot[BitTerminal(inst.id, "rst")])
            elif kind is PrimitiveKind.REGISTER_BIT:
                driver_slot(BitTerminal(inst.id, "clk"))  # an unclocked register cannot latch
                self._registers.append(
                    (
                        inst.id,
                        slot[BitTerminal(inst.id, "q")],
                        driver_slot(BitTerminal(inst.id, "d")),
                        driver_slot(BitTerminal(inst.id, "rst")),
                        int(bool(inst.init)),
                    )
                )
            elif kind in _OPCODES:
                gates.append(inst)
        self._program = self._schedule(gates, driver_slot)
        # Ports: name -> (type, slot per bit LSB first).
        self._inputs: dict[str, tuple[IRType, list[int]]] = {}
        self._outputs: dict[str, tuple[IRType, list[int]]] = {}
        for port in netlist.ports:
            if port.name in self._inputs or port.name in self._outputs:
                raise CompileError(f"simulation: two ports are named {port.name!r}")
            if port.direction == "input":
                self._inputs[port.name] = (port.type, [slot[bit] for bit in port.bits])
            else:
                self._outputs[port.name] = (port.type, [driver_slot(bit) for bit in port.bits])

    # -- compilation -----------------------------------------------------------

    def _schedule(self, gates, driver_slot) -> list[tuple[int, int, int, int]]:
        """Gates in a deterministic topological order (Kahn, lowest id first).
        Register bits, constants and boundaries are sources, so every cycle
        must pass through a register; anything else is a combinational loop."""
        gate_of_slot = {self._slot[BitTerminal(g.id, "y")]: g.id for g in gates}
        deps: dict[int, set[int]] = {g.id: set() for g in gates}
        users: dict[int, list[int]] = {g.id: [] for g in gates}
        operands: dict[int, tuple[int, int]] = {}
        for g in gates:
            ins = [driver_slot(BitTerminal(g.id, pin)) for pin in g.inputs]
            operands[g.id] = (ins[0], ins[1] if len(ins) > 1 else -1)
            for s in ins:
                producer = gate_of_slot.get(s)
                if producer is not None and producer not in deps[g.id]:
                    deps[g.id].add(producer)
                    users[producer].append(g.id)
        ready = [gid for gid, d in deps.items() if not d]
        heapq.heapify(ready)
        kinds = {g.id: g.kind for g in gates}
        program: list[tuple[int, int, int, int]] = []
        while ready:
            gid = heapq.heappop(ready)
            a, b = operands[gid]
            program.append((_OPCODES[kinds[gid]], self._slot[BitTerminal(gid, "y")], a, b))
            for user in users[gid]:
                deps[user].discard(gid)
                if not deps[user]:
                    heapq.heappush(ready, user)
        if len(program) != len(gates):
            stuck = sorted(gid for gid, d in deps.items() if d)[:20]
            raise CompileError(f"simulation: combinational loop through primitives {stuck}")
        return program

    # -- core evaluation (bit-parallel) ------------------------------------

    def _settle(
        self,
        packed_inputs: Mapping[str, list[int]],
        state: Mapping[int, int],
        rst: int,
        mask: int,
    ) -> list[int]:
        values = [0] * self.slots
        for s in self._const1:
            values[s] = mask
        for s in self._reset:
            values[s] = rst
        for name, (_typ, slots) in self._inputs.items():
            for s, bits in zip(slots, packed_inputs[name]):
                values[s] = bits
        for inst_id, q, _d, _r, _init in self._registers:
            values[q] = state[inst_id]
        for op, out, a, b in self._program:
            if op == _AND:
                values[out] = values[a] & values[b]
            elif op == _OR:
                values[out] = values[a] | values[b]
            elif op == _XOR:
                values[out] = values[a] ^ values[b]
            else:
                values[out] = values[a] ^ mask
        return values

    def _pack_inputs(self, inputs: Mapping[str, Sequence[int]], count: int) -> dict[str, list[int]]:
        packed: dict[str, list[int]] = {}
        for name, (typ, _slots) in self._inputs.items():
            if name not in inputs:
                raise CompileError(f"missing input {name}")
            column = [typ.bits(v) for v in inputs[name]]
            if len(column) != count:
                raise CompileError(f"input {name}: expected {count} vectors, got {len(column)}")
            packed[name] = [
                int("".join("1" if (v >> i) & 1 else "0" for v in reversed(column)), 2)
                for i in range(typ.width)
            ]
        return packed

    def _unpack_outputs(self, values: list[int], count: int) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for name, (typ, slots) in self._outputs.items():
            raw = [0] * count
            for i, s in enumerate(slots):
                bits = values[s]
                if not bits:
                    continue
                text = bin(bits)[2:].zfill(count)[::-1]
                weight = 1 << i
                for k, ch in enumerate(text):
                    if ch == "1":
                        raw[k] |= weight
            result[name] = [typ.number(v) for v in raw]
        return result

    # -- public API ----------------------------------------------------------

    @property
    def input_names(self) -> list[str]:
        return list(self._inputs)

    @property
    def output_names(self) -> list[str]:
        return list(self._outputs)

    @property
    def is_sequential(self) -> bool:
        return bool(self._registers)

    def evaluate_many(self, inputs: Mapping[str, Sequence[int]]) -> dict[str, list[int]]:
        """Settle a combinational netlist for many input vectors at once.
        ``inputs`` maps every input port to an equally long list of values."""
        if self.is_sequential:
            raise CompileError("sequential primitive netlist holds state; use step() or run()")
        counts = {len(v) for v in inputs.values()} or {1}
        if len(counts) != 1:
            raise CompileError("evaluate_many: every input needs the same number of vectors")
        count = counts.pop()
        if count < 1:
            raise CompileError("evaluate_many: need at least one vector")
        mask = (1 << count) - 1
        values = self._settle(self._pack_inputs(inputs, count), {}, 0, mask)
        return self._unpack_outputs(values, count)

    def evaluate(self, **inputs: int) -> dict[str, int]:
        """One combinational evaluation, like :meth:`redc.ir.Graph.evaluate`."""
        many = self.evaluate_many({name: [value] for name, value in inputs.items()} if self._inputs else {})
        return {name: values[0] for name, values in many.items()}

    def reset_state(self) -> dict[int, int]:
        """Register-bit state after the global reset: instance id -> init bit."""
        return {inst_id: init for inst_id, _q, _d, _r, init in self._registers}

    def state_to_ir(self, state: Mapping[int, int]) -> dict[int, int]:
        """Primitive register-bit state -> IR register state (register node id
        -> raw bits), reassembled LSB first from each register's logical bus."""
        result: dict[int, int] = {}
        for node_id, bus in self.netlist.buses.items():
            if bus.op != "register":
                continue
            result[node_id] = sum((state[bit.instance] & 1) << i for i, bit in enumerate(bus.bits))
        return result

    def state_from_ir(self, ir_state: Mapping[int, int]) -> dict[int, int]:
        """IR register state -> primitive register-bit state."""
        result: dict[int, int] = {}
        for node_id, bus in self.netlist.buses.items():
            if bus.op != "register":
                continue
            for i, bit in enumerate(bus.bits):
                result[bit.instance] = (ir_state[node_id] >> i) & 1
        return result

    def step(
        self, state: Mapping[int, int], *, rst: int = 0, **inputs: int
    ) -> tuple[dict[str, int], dict[int, int]]:
        """One clock cycle: ``(outputs visible this cycle, next state)``, like
        :meth:`redc.ir.Graph.step` (plus an explicit ``rst`` level)."""
        packed = self._pack_inputs({n: [v] for n, v in inputs.items()}, 1)
        values = self._settle(packed, state, 1 if rst else 0, 1)
        outputs = {name: vals[0] for name, vals in self._unpack_outputs(values, 1).items()}
        next_state = {
            inst_id: (init if values[r] & 1 else values[d] & 1)
            for inst_id, _q, d, r, init in self._registers
        }
        return outputs, next_state

    def run(self, *, max_cycles: int = 100_000, **inputs: int) -> dict[str, int]:
        """One transaction from reset, exactly like :meth:`redc.ir.Graph.run`:
        ``start`` (if present) pulses on the first cycle, data inputs are held,
        and the outputs are returned on the first later cycle ``done`` is high."""
        if not self.is_sequential:
            raise CompileError("run() is for sequential netlists; use evaluate()")
        if "done" not in self._outputs:
            raise CompileError("sequential netlist has no 'done' output to wait on")
        has_start = "start" in self._inputs
        state = self.reset_state()
        for cycle in range(max_cycles):
            stimulus = dict(inputs)
            if has_start:
                stimulus["start"] = 1 if cycle == 0 else 0
            outputs, next_state = self.step(state, **stimulus)
            if cycle > 0 and outputs["done"]:
                return outputs
            state = next_state
        raise CompileError(f"simulation did not finish within {max_cycles} cycles")


def simulator(netlist: PrimitiveNetlist) -> PrimitiveSimulator:
    return PrimitiveSimulator(netlist)


def evaluate(netlist: PrimitiveNetlist, **inputs: int) -> dict[str, int]:
    return PrimitiveSimulator(netlist).evaluate(**inputs)


__all__ = ["PrimitiveSimulator", "evaluate", "simulator"]
