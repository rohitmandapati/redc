"""The primitive viewer's timing / simulation features, run headlessly in
QuickJS with the stubs of ``tests/test_primitive_viewer_js.py``: the timing
panel, the clock-tree / critical-path highlight, the replay of
``clock_balanced`` and the recorded signal playback."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

quickjs = pytest.importorskip("quickjs")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from test_primitive_viewer_js import HALF_ADDER, js, run, trace_for  # helpers only

COUNTER2 = "uint2 main(uint2 n) { uint2 x = 0; for (uint2 i = 0; i < n; i++) { x = x + 1; } return x; }"


@pytest.fixture(scope="module")
def counter() -> dict:
    return trace_for(COUNTER2, "basic")


def test_timing_panel_shows_period_skew_slacks_and_simulation(counter: dict) -> None:
    timing = counter["final"]["timing"]
    out = run(counter, "frame(); return elements['timing-panel'].textContent;")
    assert "error" not in out, out
    clock = timing["clock"]
    assert "clock period" in out and f"{clock['period']['rt']} rt ({clock['period']['gt']} gt)" in out
    assert "clock skew" in out and "0 rt (0 gt)" in out
    assert f"{timing['setup']['worst_slack']['gt']} gt" in out and f"{timing['hold']['worst_slack']['gt']} gt" in out
    assert "abstract-components · validated" in out
    assert "critical path to" in out and "closure" in out and "passed" in out


def test_combinational_designs_show_their_settle_time() -> None:
    trace = trace_for(HALF_ADDER, "basic")
    settle = trace["final"]["timing"]["combinational"]["settle"]
    out = run(trace, "frame(); return elements['timing-panel'].textContent;")
    assert "combinational settle" in out and f"{settle['rt']} rt ({settle['gt']} gt)" in out


def test_timing_highlight_marks_the_clock_tree_and_the_critical_path(counter: dict) -> None:
    nets = counter["design"]["nets"]
    clock_nets = sorted(n["id"] for n in nets if n["role"] == "clock")
    registers = sorted(i["id"] for i in counter["design"]["instances"] if i["kind"] == "register_bit")
    out = run(counter, js("""
      function flags(a) { var r = []; for (var i = 0; i < a.length; i++) if (a[i]) r.push(i); return r; }
      var r = {};
      elements['hl-mode'].value = 'timing'; elements['hl-mode'].onchange();
      r.items = elements['hl-item'].children.map(function (o) { return o.value; });
      elements['hl-item'].value = 'clock'; elements['hl-item'].onchange(); frame();
      r.clockNets = flags(viewer.highlight.net); r.clockInst = flags(viewer.highlight.inst);
      r.clockLabel = viewer.highlight.label;
      viewer.setHighlight({ mode: 'timing', name: 'setup' }); frame();
      r.setupNets = flags(viewer.highlight.net); r.setupInst = flags(viewer.highlight.inst);
      r.setupLabel = viewer.highlight.label;
      return r;
    """))
    assert "error" not in out, out
    assert out["items"] == ["", "clock", "setup", "hold"]
    assert set(clock_nets) <= set(out["clockNets"]) and set(registers) <= set(out["clockInst"])
    assert out["clockLabel"].startswith("clock tree")
    path = counter["final"]["timing"]["setup"]["critical_path"]
    want = {n for step in path["steps"] for n in step.get("nets", [])}
    assert want and want <= set(out["setupNets"])
    if path["endpoint_kind"] == "register":
        assert path["capture_register_labels"]["instance"] in out["setupInst"]
    if "launch_register_labels" in path:
        assert path["launch_register_labels"]["instance"] in out["setupInst"]
    assert out["setupLabel"].startswith("worst setup path to")


def test_clock_balanced_replaces_the_clock_route_during_replay(counter: dict) -> None:
    events = counter["events"]
    k = next(i for i, e in enumerate(events) if e["type"] == "clock_balanced")
    net = events[k]["net"]
    out = run(counter, js("""
      viewer.seek(__AT__); var before = JSON.stringify(viewer.state.realized.get(__NET__).elements);
      viewer.seek(__AT__ + 1); var after = viewer.state.realized.get(__NET__);
      return { before: before, after: JSON.stringify(after.elements), status: viewer.state.status };
    """, at=k, net=net))
    assert "error" not in out, out
    import json

    assert json.loads(out["after"]) == events[k]["realized"]["elements"] != json.loads(out["before"])
    assert "clock balanced" in out["status"]
    final = next(r for r in counter["final"]["realized"] if r["net"] == net)
    assert final["elements"] == events[k]["realized"]["elements"]


def test_signal_playback_reconstructs_dust_and_devices_at_any_tick(counter: dict) -> None:
    pb = counter["final"]["simulation_playback"]
    assert pb and pb["dust"] and pb["start_gt"] < pb["end_gt"]
    probe_ticks = [pb["start_gt"], (pb["start_gt"] + pb["end_gt"]) // 2, pb["end_gt"]]

    def expected(t: int) -> tuple[int, int]:
        dust = {tuple(r[:3]): r[3] for r in pb["initial_dust"]}
        devices = {tuple(r[:3]): r[3] for r in pb["initial_devices"]}
        for tt, x, y, z, s in sorted(pb["dust"], key=lambda r: r[0]):
            if tt <= t:
                dust[(x, y, z)] = s
        for tt, x, y, z, on in sorted(pb["devices"], key=lambda r: r[0]):
            if tt <= t:
                devices[(x, y, z)] = on
        return sum(1 for v in dust.values() if v > 0), sum(1 for v in devices.values() if v)

    out = run(counter, js("""
      var r = [];
      [__T0__, __T1__, __T2__].forEach(function (t) {
        viewer.setSimTick(t); frame();
        var st = viewer.signalState(t); var dust = 0, dev = 0;
        st.dust.forEach(function (v) { if (v > 0) dust++; });
        st.devices.forEach(function (v) { if (v) dev++; });
        r.push([dust, dev, drawn('signals').length, elements['sim-info'].textContent]);
      });
      r.push(elements['sim-controls'].hidden);
      return r;
    """, t0=probe_ticks[0], t1=probe_ticks[1], t2=probe_ticks[2]))
    assert "error" not in out, out
    hidden = out.pop()
    assert hidden is False
    for t, (dust, devices, drawn_count, info) in zip(probe_ticks, out):
        assert (dust, devices) == expected(t)
        assert drawn_count == dust + devices
        assert f"game tick {t}" in info
    # The signals actually move: the clock edges change the powered set.
    assert len({row[0] for row in out}) > 1 or pb["devices"]


def test_play_signals_advances_in_game_ticks(counter: dict) -> None:
    pb = counter["final"]["simulation_playback"]
    out = run(counter, """
      elements['btn-sim-play'].onclick();
      var started = viewer.simPlaying;
      viewer.clock.getDelta = function () { return 0.5; };   // half a second per frame
      frame(); frame();
      return [started, viewer.simTick];
    """)
    assert "error" not in out, out
    started, tick = out
    assert started is True
    assert tick == min(pb["end_gt"], pb["start_gt"] + 20)  # 20 game ticks per second
