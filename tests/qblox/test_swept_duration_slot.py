"""A pulse with a swept duration occupies the same time for every duration.

Padding and alignment use a nominal duration, so the compiled program has to make the
real one match it: everything after the swept pulse starts at the same time.
"""

import numpy as np
import pytest

from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Instr,
    Play,
    Register,
    SetAwgOffs,
    UpdParam,
    Wait,
)
from qibolab._core.instruments.qblox.sequence.experiment import experiment
from qibolab._core.instruments.qblox.sequence.sequence import compile
from qibolab._core.instruments.qblox.sequence.sweepers import (
    params,
    reserve_sweep_durations,
    sweep_sequence,
)
from qibolab._core.instruments.qblox.sequence.waveforms import waveforms
from qibolab._core.pulses import Delay, Gaussian, Pulse, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import Parameter, Sweeper, swept_pulses

DURATIONS = np.array([8.0, 12.0, 16.0, 20.0])
SHORT = np.array([4.0, 5.0, 6.0, 7.0])


def _compile(sequence, sweepers):
    options = ExecutionParameters(nshots=1, relaxation_time=100)
    return compile(
        PulseSequence(sequence), sweepers, options, sampling_rate=1.0, merged_vzs=True
    )


def _instructions(q1seq) -> list[Instr]:
    return [line.instruction for line in q1seq.program.lines]


def _sweeper(pulse, values):
    return Sweeper(parameter=Parameter.duration, values=values, pulses=[pulse])


def _swept(envelope, values):
    pulse = Pulse(duration=values[0], amplitude=0.5, envelope=envelope)
    return pulse, [[_sweeper(pulse, values)]]


def _swept_waits(instrs):
    """Wait immediates and registers of the whole program, in order."""
    return [i.duration for i in instrs if isinstance(i, Wait)]


def test_reserve_sets_the_nominal_duration_to_the_longest_value():
    pulse, sweepers = _swept(Rectangular(), DURATIONS)
    other = Pulse(duration=30, amplitude=0.1, envelope=Rectangular())
    seq = PulseSequence([("a", pulse), ("a", other)])
    reserved = reserve_sweep_durations(seq, sweepers)
    # rectangular pulse realized as an offset: the tail after its end is included
    assert reserved[0][1].duration == 20 + 4
    assert reserved[0][1].id == pulse.id
    assert reserved[1][1] is other
    assert reserved.duration == 24 + 30
    # the original sequence is not touched
    assert seq[0][1].duration == 8


def test_reserve_waveform_pulse_uses_longest_value():
    pulse, sweepers = _swept(Gaussian(rel_sigma=0.2), DURATIONS)
    reserved = reserve_sweep_durations(PulseSequence([("a", pulse)]), sweepers)
    assert reserved[0][1].duration == 20


def test_reserve_does_not_touch_swept_delays():
    delay = Delay(duration=8)
    sweepers = [
        [Sweeper(parameter=Parameter.duration, values=DURATIONS, pulses=[delay])]
    ]
    reserved = reserve_sweep_durations(PulseSequence([("a", delay)]), sweepers)
    assert reserved[0][1] is delay


def _asm(sequence, sweepers):
    return " ".join(_compile(sequence, sweepers)["a"].program.asm().split())


def test_rectangular_slot_has_a_complementary_register():
    pulse, sweepers = _swept(Rectangular(), DURATIONS)
    asm = _asm([("a", pulse)], sweepers)
    # the register that ends the pulse and the one that completes the slot, which is
    # swept in the opposite direction
    assert "role: duration.DURATION" in asm
    assert "role: duration.COMPLEMENT" in asm
    assert "sub R8,4,R8" in asm


def test_waveform_slot_is_a_fixed_wait():
    pulse, sweepers = _swept(Gaussian(rel_sigma=0.2), DURATIONS)
    asm = _asm([("a", pulse)], sweepers)
    # the waveform indices are the only registers of the sweep ...
    assert "role: duration.PULSE_I" in asm
    assert "role: duration.DURATION" not in asm
    assert "role: duration.COMPLEMENT" not in asm
    # ... and the slot is completed by a fixed wait that the compiler merges with the
    # following one (relaxation 100 ns + 4 ns minimum + 16 ns)
    assert "wait 120" in asm


def test_short_swept_pulses_get_the_minimum_slot():
    pulse, sweepers = _swept(Rectangular(), SHORT)
    asm = _asm([("a", pulse)], sweepers)
    assert "role: duration.DURATION" not in asm
    # 8 ns slot: 4 ns after the `play`, and 104 ns of relaxation and padding
    assert "wait 108" in asm


def _timeline(events, sweepers, k):
    """Real-time instants of the instructions of the experiment, for the k-th value.

    Registers are evaluated as the sweep loop would at that iteration. Returns the time
    at which each instruction starts and the total time.
    """
    swept = swept_pulses(sweepers, {Parameter.duration})
    reserved = [
        ev
        for _, ev in reserve_sweep_durations(
            PulseSequence([("a", ev) for ev in events]), sweepers
        )
    ]
    ps = params(sweepers, 0, {ev.id for ev in events}, set())
    _, realization = waveforms(events, 1.0, set(), {i: s for i, s in swept.items()})
    body = experiment(sweep_sequence(reserved, ps), realization, {}, True)
    value = {p.reg.number: p.start + k * p.step for p in ps}

    def length(duration):
        return value[duration.number] if isinstance(duration, Register) else duration

    now, times = 0, []
    for ins in body:
        ins = getattr(ins, "instruction", ins)
        times.append((now, ins))
        if isinstance(ins, (Wait, UpdParam, Play)):
            now += length(ins.duration)
    return times, now


@pytest.mark.parametrize(
    "envelope", [Rectangular(), Gaussian(rel_sigma=0.2)], ids=["offset", "waveform"]
)
def test_events_after_a_swept_pulse_do_not_move(envelope):
    pulse, sweepers = _swept(envelope, DURATIONS)
    follower = Pulse(duration=16, amplitude=0.2, envelope=Gaussian(rel_sigma=0.3))
    events = [pulse, follower]
    results = [_timeline(events, sweepers, k) for k in range(len(DURATIONS))]
    # the follower is played at the same time for every duration
    starts = {
        next(t for t, i in times if isinstance(i, Play) and i is not None and t > 4)
        for times, _ in results
    }
    totals = {total for _, total in results}
    assert len(totals) == 1
    assert len(starts) == 1


def test_offset_pulse_ends_at_its_swept_duration():
    pulse, sweepers = _swept(Rectangular(), DURATIONS)
    for k, d in enumerate(DURATIONS):
        times, _ = _timeline([pulse], sweepers, k)
        reset = next(
            t for t, i in times if isinstance(i, SetAwgOffs) and i.value_0 == 0
        )
        # the first `upd_param` precedes the pulse
        assert reset - 4 == d


def test_other_channels_wait_for_the_whole_slot():
    """The channels aligned after a swept pulse are delayed by its slot."""
    pulse, sweepers = _swept(Gaussian(rel_sigma=0.2), DURATIONS)
    follower = Pulse(duration=16, amplitude=0.2, envelope=Gaussian(rel_sigma=0.3))
    other = Pulse(duration=16, amplitude=0.2, envelope=Gaussian(rel_sigma=0.3))
    seq = PulseSequence([("a", pulse)])
    seq.align(["a", "b"])
    seq.extend([("a", follower), ("b", other)])
    aligned = reserve_sweep_durations(seq, sweepers).align_to_delays()
    delays = [ev.duration for ch, ev in aligned if ch == "b" and isinstance(ev, Delay)]
    assert delays == [DURATIONS.max()]
