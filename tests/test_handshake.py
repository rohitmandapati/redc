"""The reusable start/done handshake, driven on persistent state.

``Graph.run()`` starts from reset every call, so it cannot show that ONE state
machine serves back-to-back transactions.  These tests drive
``Graph.reset_state()`` + ``Graph.step()`` cycle by cycle instead.
"""

import itertools
from pathlib import Path

from redc import compile_source

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

COUNTER = "uint8 main(uint8 n) { uint8 c = 0; while (c < n) { c = c + 1; } return c; }"


def py_fib(n: int) -> int:
    last1 = last2 = 1
    for _ in range(2, n):
        last1, last2 = last2, last1 + last2
    return last2


class Harness:
    """Cycle-accurate driver around one persistent register state."""

    def __init__(self, graph) -> None:
        self.graph = graph
        self.state = graph.reset_state()  # rst asserted, then released
        self.trace: list[dict[str, int]] = []

    def cycle(self, start: int = 0, **inputs: int) -> dict[str, int]:
        outputs, self.state = self.graph.step(self.state, start=start, **inputs)
        self.trace.append(outputs)
        return outputs

    def transaction(self, max_cycles: int = 1000, **inputs: int) -> tuple[int, int]:
        """Pulse start once, run to done; return (result, cycles incl. start)."""
        out = self.cycle(start=1, **inputs)
        assert out["done"] == 0
        for elapsed in range(2, max_cycles):
            out = self.cycle(**inputs)
            if out["done"]:
                return out["result"], elapsed
        raise AssertionError("transaction never finished")


def test_idle_after_reset() -> None:
    h = Harness(compile_source(COUNTER))
    for _ in range(5):
        assert h.cycle(in_n=3)["done"] == 0


def test_done_is_one_cycle_and_fsm_returns_to_idle() -> None:
    h = Harness(compile_source(COUNTER))
    result, _ = h.transaction(in_n=3)
    assert result == 3
    # Idle again: done low, result held, for as long as nobody pulses start.
    for _ in range(5):
        out = h.cycle(in_n=3)
        assert out == {"result": 3, "done": 0}


def test_back_to_back_transactions_without_reset() -> None:
    h = Harness(compile_source(COUNTER))
    assert h.transaction(in_n=4)[0] == 4
    h.cycle(in_n=4)  # one idle cycle
    assert h.transaction(in_n=2)[0] == 2
    # Immediately after done, with no idle gap.
    assert h.transaction(in_n=5)[0] == 5
    assert sum(out["done"] for out in h.trace) == 3  # exactly one pulse each


def test_fib_transactions_reuse_one_state_machine() -> None:
    graph = compile_source((EXAMPLES / "uint8_fib.redc").read_text(), top="fib")
    h = Harness(graph)
    for n in (7, 3, 10, 1, 12):
        assert h.transaction(in_n=n)[0] == py_fib(n)


def test_zero_iteration_loop_still_finishes() -> None:
    h = Harness(compile_source(COUNTER))
    assert h.transaction(in_n=0) == (0, 2)  # start cycle, then done
    assert h.cycle(in_n=0)["done"] == 0
    assert h.transaction(in_n=0) == (0, 2)


def test_start_while_running_is_ignored() -> None:
    h = Harness(compile_source(COUNTER))
    h.cycle(start=1, in_n=4)
    seen = [h.cycle(in_n=4)["result"]]
    # Re-pulse start mid-transaction: must not reload c = 0.
    for start in (1, 0, 1, 0, 1):
        out = h.cycle(start=start, in_n=4)
        seen.append(out["result"])
        if out["done"]:
            break
    assert out == {"result": 4, "done": 1}
    assert seen == sorted(seen)  # never restarted from 0
    assert h.cycle(in_n=4)["done"] == 0  # back to idle


def test_start_held_high_does_not_restart_after_done() -> None:
    # A start held across the done cycle is ignored there (running is still
    # high), so the FSM goes idle; on the next idle cycle it is accepted again.
    h = Harness(compile_source(COUNTER))
    outs = [h.cycle(start=1, in_n=2) for _ in range(8)]
    dones = [i for i, out in enumerate(outs) if out["done"]]
    assert dones and all(outs[i]["result"] == 2 for i in dones)
    assert all(b - a > 1 for a, b in itertools.pairwise(dones))  # never back-to-back


def test_run_still_matches_single_transaction() -> None:
    graph = compile_source(COUNTER)
    for n in range(6):
        assert graph.run(in_n=n) == {"result": n, "done": 1}
        assert Harness(graph).transaction(in_n=n)[0] == n
