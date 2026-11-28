from collections.abc import Callable, Container, Iterable
from enum import Enum, auto
from itertools import groupby

import numpy as np

from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.q1asm.ast_ import (
    Add,
    Instruction,
    Register,
    SetAwgGain,
    SetAwgOffs,
    SetFreq,
    Value,
)
from qibolab._core.instruments.qblox.sequence.asm import Registers
from qibolab._core.pulses import Pulse, PulseId, PulseLike, Rectangular
from qibolab._core.sequence import PulseSequence
from qibolab._core.serialize import Model
from qibolab._core.sweeper import (
    ParallelSweepers,
    Parameter,
    Range,
    Sweeper,
    swept_pulses,
)

from .asm import MAX_PARAM, _validate_sweeper_value, convert

__all__ = ["is_offset_rectangular", "reserve_sweep_durations"]


def is_offset_rectangular(
    pulse: PulseLike, duration_sweep: Sweeper | None = None
) -> bool:
    """Decides whether the pulse is compiled through AWG offsets instead of waveforms.

    To conserve waveform memory, rectangular pulses of at least 8 ns use
    ``set_awg_offs`` instead of a waveform expressed as an array of floats. Shorter
    pulses cannot use this optimization because the ``upd_param`` instruction takes 4 ns
    and the ``wait`` at least 4 ns in the Q1 core.

    A duration sweep is the only parameter that can change the offset-vs-waveform
    decision, since it affects the 8 ns threshold.
    """
    if not (isinstance(pulse, Pulse) and isinstance(pulse.envelope, Rectangular)):
        return False
    if duration_sweep is not None:
        assert (
            duration_sweep.parameter is Parameter.duration
            and duration_sweep.values is not None
        )
        return bool(np.all(duration_sweep.values >= 8))
    return pulse.duration >= 8


class ParamRole(Enum):
    """Parameter role in sweep.

    Discriminate the various register roles. Particularly, in the pulse
    duration sweeps.
    """

    FREQUENCY = auto(), Parameter.frequency
    "Channel frequency."
    OFFSET = auto(), Parameter.offset
    "Channel offset."
    AMPLITUDE = auto(), Parameter.amplitude
    "Pulse amplitude."
    RELATIVE_PHASE = auto(), Parameter.relative_phase
    "Pulse relative phase."
    PHASE = auto(), Parameter.phase
    "Virtual Z phase."
    DURATION = auto(), Parameter.duration
    "Pulse duration."
    PULSE_I = auto(), Parameter.duration
    "Pulse I component."
    PULSE_Q = auto(), Parameter.duration
    "Pulse Q component."
    COMPLEMENT = auto(), Parameter.duration
    "Wait that completes the slot of a swept rectangular pulse (see :func:`slot`)."

    @classmethod
    def from_sweeper(cls, sweep: Sweeper) -> "ParamRole":
        for var in cls:
            if sweep.parameter is var.kind:
                return var
        raise ValueError("Sweeper parameter kind does not correspond to any role.")

    @property
    def kind(self):
        return self.value[1]


class Param(Model):
    reg: Register
    """Register used for the parameter value."""
    start: int
    """Initial value."""
    step: int
    """Increment."""
    role: ParamRole
    """The parameter type."""
    pulse: PulseId | None
    """The target pulse (if the sweeper targets pulses)."""
    channel: ChannelId | None
    """The target channel (if the sweeper targets channels)."""
    loop: int | None = None
    """The loop which is associated to."""
    limit: int | None = None
    """The largest value of a duration sweep, i.e. the slot the pulse occupies."""

    @property
    def description(self) -> str:
        """Textual description, used in some accompanying comments."""
        return (
            "sweeper ("
            + (f"loop: {self.loop}, " if self.loop is not None else "")
            + (f"pulse: 0x{self.pulse.hex[:5]}, " if self.pulse is not None else "")
            + f"role: {self.role.kind.name}.{self.role.name}"
            + ")"
        )

    @classmethod
    def from_range(
        cls,
        irange: Range,
        kind: Parameter,
        role: ParamRole,
        pulse: PulseLike | None,
        channel: ChannelId | None,
        limit: int | None = None,
    ) -> "Param":
        """Create a Param from a sweep range with validation.

        Validates the entire range (start, stop, step) before converting.
        """
        start_, stop, step = irange
        start = start_ - _duration_shift(role, pulse)
        # Validate all points in the range
        for value in (start, stop, step):
            _validate_sweeper_value(value, kind)
        return cls(
            reg=Register(number=0),
            start=int(convert(start, kind)),
            step=int(convert(step, kind)),
            pulse=pulse.id if pulse is not None else None,
            channel=channel,
            role=role,
            limit=limit,
        )


class LoopParams(Model):
    """Parameters involved in a single loop level."""

    channel: list[Param]
    pulse: list[Param]

    @property
    def all(self):
        return self.channel + self.pulse


IndexedParams = dict[int, LoopParams]
"""Sweep parameters, organized by loop.

Keys are going to be loop counters.
"""


def _channels_pulses(
    pars: Iterable[Param],
) -> LoopParams:
    channels = []
    pulses = []
    for p in pars:
        (channels if p.channel is not None else pulses).append(p)
    return LoopParams(channel=channels, pulse=pulses)


def params_reshape(params: list[Param]) -> IndexedParams:
    """Split parameters related to channels and pulses.

    Moreover, it reorganize them by loop, to group the updates.
    """
    return {
        key: _channels_pulses(pars)
        for key, pars in groupby(params, key=lambda p: p.loop)
        if key is not None
    }


# The instruction that ends a swept rectangular pulse, `upd_param 4`, is followed by a
# wait that completes the slot. The wait cannot be shorter than 4 ns, so the slot is
# that much longer than the longest pulse.
RECTANGULAR_TAIL = 4
MIN_WAVEFORM_SLOT = 8
"""Shortest slot of a swept waveform pulse: `play` (4 ns) plus a wait of at least 4 ns."""


def sweep_values(sweep: Sweeper) -> np.ndarray:
    return np.arange(*sweep.irange)


def slot(pulse: Pulse, sweep: Sweeper) -> float:
    """Time reserved for a pulse whose duration is swept.

    The pulse occupies the same time for every value of the sweep, so that everything
    after it, on any channel, starts at the same time. A wait completes the slot after
    the pulse. The slot is set by the longest value, so there is no need for a
    different nominal duration, which would otherwise be what padding and alignment use.
    """
    longest = float(np.max(sweep_values(sweep)))
    return (
        longest + RECTANGULAR_TAIL
        if is_offset_rectangular(pulse, sweep)
        else max(longest, MIN_WAVEFORM_SLOT)
    )


def reserve_sweep_durations(
    sequence: PulseSequence, sweepers: list[ParallelSweepers]
) -> PulseSequence:
    """Set the nominal duration of the swept pulses to the slot they occupy.

    Padding and the alignment of channels are computed from nominal durations, so this
    has to be done before them.
    """
    swept = swept_pulses(sweepers, {Parameter.duration})
    return type(sequence)(
        (
            ch,
            ev.model_copy(update={"duration": slot(ev, swept[ev.id])})
            if isinstance(ev, Pulse) and ev.id in swept
            else ev,
        )
        for ch, ev in sequence
    )


def _registers(sweep: Sweeper) -> list[tuple[Range, ParamRole]]:
    """Reserve registers for sweeping.

    Most sweepers require only one register. The exception is a duration sweep, where
    the pulse is followed by a wait that keeps the length of its slot fixed:

    - pulses played from waveform memory compile into ``play <I>, <Q>, 4`` followed by a
      fixed wait, so the registers are the two waveform indices (the sequencer plays the
      whole waveform by itself, whatever the wait);
    - rectangular pulses realized as offsets need the duration and the complementary
      wait, to end the pulse and to complete the slot;
    - a swept delay is a single wait.
    """
    if sweep.parameter is not Parameter.duration:
        return [(sweep.irange, ParamRole.from_sweeper(sweep))]
    targets = sweep.pulses or []
    pulses = [p for p in targets if isinstance(p, Pulse)]
    played = any(not is_offset_rectangular(p, sweep) for p in pulses)
    offset = any(is_offset_rectangular(p, sweep) for p in pulses)
    start, _, step = sweep.irange
    top = float(np.max(sweep_values(sweep)))
    registers: list[tuple[Range, ParamRole]] = []
    if played:
        registers += [
            ((0, 2 * len(sweep), 2), ParamRole.PULSE_I),
            ((1, 2 * len(sweep) + 1, 2), ParamRole.PULSE_Q),
        ]
    if offset or len(pulses) < len(targets) or not targets:
        registers.append((sweep.irange, ParamRole.DURATION))
    if offset:
        registers.append(
            (
                (top + RECTANGULAR_TAIL - start, 0, -step),
                ParamRole.COMPLEMENT,
            )
        )
    return registers


def _duration_shift(role: ParamRole, pulse: PulseLike | None) -> int:
    """Shift of the register values for a swept pulse duration.

    Only rectangular pulses realized as offsets have a duration register, and they start
    with a 4 ns latch instruction followed by a wait (see ``_process_rectangular``)::

        upd_param 4
        wait <duration - 4>

    Any other duration sweeper requires no duration shift.
    """
    return 4 if role is ParamRole.DURATION and isinstance(pulse, Pulse) else 0


def _unravel_sweeps(
    sweepers: list[ParallelSweepers],
    pulses: Container[PulseId],
    channels: Container[ChannelId],
) -> Iterable[tuple[int, Param]]:
    """Turn sweepers into ranges, selecting this sequencer's targets once per sweep."""
    return (
        (
            j,
            Param.from_range(
                irange,
                sweep.parameter,
                role,
                pulse,
                channel,
                limit=int(np.max(sweep_values(sweep)))
                if sweep.parameter is Parameter.duration
                else None,
            ),
        )
        # the first sweeper should be the outermost, thus reverse them during the
        # enumeration
        for j, parsweep in enumerate(sweepers[::-1])
        for sweep in parsweep
        for targets in [
            [(p, None) for p in sweep.pulses if p.id in pulses]
            if sweep.pulses is not None
            else [(None, ch) for ch in (sweep.channels or []) if ch in channels]
        ]
        for irange, role in _registers(sweep)
        for pulse, channel in targets
    )


def params(
    sweepers: list[ParallelSweepers],
    allocated: int,
    pulses: Container[PulseId],
    channels: Container[ChannelId],
) -> list[Param]:
    """Initialize parameters' registers.

    `allocated` is the number of already allocated registers for loop counters, as
    initialized by :func:`loops`.

    Targets outside `pulses` and `channels` are excluded before constructing their
    parameters, and registers are allocated just for the retained ones. Since each
    sequencer runs its own program, this spares the registers of parameters belonging
    to other sequencers.
    """
    return [
        p.model_copy(update={"reg": Register(number=i + allocated + 1), "loop": j})
        for i, (j, p) in enumerate(_unravel_sweeps(sweepers, pulses, channels))
    ]


class _Update(Model):
    update: Callable[[Value], Instruction] | None
    reset: Callable[[Value], Instruction] | None


_SWEEP_UPDATE: dict[Parameter, _Update] = {
    Parameter.frequency: _Update(update=lambda v: SetFreq(value=v), reset=None),
    Parameter.offset: _Update(
        update=lambda v: SetAwgOffs(value_0=v, value_1=v), reset=None
    ),
    Parameter.amplitude: _Update(
        update=lambda v: SetAwgGain(value_0=v, value_1=v),
        reset=lambda _: SetAwgGain(
            value_0=MAX_PARAM[Parameter.amplitude],
            value_1=MAX_PARAM[Parameter.amplitude],
        ),
    ),
    Parameter.relative_phase: _Update(
        update=lambda v: Add(
            a=Registers.phase_delta.value, b=v, destination=Registers.phase_delta.value
        ),
        reset=None,
    ),
    # The phase is handled in _process_virtualz, where the relative phase is added to
    # the accumulated phase delta
    Parameter.phase: _Update(update=None, reset=None),
    Parameter.duration: _Update(update=None, reset=None),
}


def update_instructions(
    role: ParamRole, value: Value, reset: bool = False
) -> list[Instruction]:
    wrapper = _SWEEP_UPDATE[role.kind]
    up = wrapper.update if not reset else wrapper.reset
    instr = [up(value)] if up is not None else []
    return instr


def reset_instructions(role: ParamRole, value: Value) -> list[Instruction]:
    return update_instructions(role, value, reset=True)


ParameterizedPulse = tuple[PulseLike, set[Param]]
SweepSequence = list[ParameterizedPulse]


def sweep_sequence(sequence: Iterable[PulseLike], params: list[Param]) -> SweepSequence:
    """Wrap swept pulses with updates markers."""
    parsbyid = {
        p: {pair[1] for pair in pairs}
        for p, pairs in groupby(
            sorted(
                ((p.pulse, p) for p in params if p.pulse is not None),
                key=lambda pair: pair[0],
            ),
            key=lambda pair: pair[0],
        )
    }
    return [(e, parsbyid.get(e.id, set())) for e in sequence]
