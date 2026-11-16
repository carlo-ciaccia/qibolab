from qibolab._core.serialize import Model

from ....q1asm.ast_ import (
    Block,
    Jlt,
    Jmp,
    Line,
    Loop,
    Move,
    Reference,
    Register,
    Sub,
    Wait,
)
from ...asm import Registers
from ..components import BlockRule, LineRule

__all__ = [
    "LongWaits",
    "long_wait",
    "merge_wait",
    "register_wait",
]

MAX_WAIT = 2**16 - 1


class LongWaits(Model):
    n: int = 0
    """Number of long waits expanded."""


def _match_long_wait(line: Line, state: LongWaits) -> tuple[bool, LongWaits]:
    instr = line.instruction
    match = (
        isinstance(instr, Wait)
        and isinstance(instr.duration, int)
        and instr.duration > MAX_WAIT
    )
    return match, state


def _map_long_wait(line: Line, state: LongWaits) -> tuple[Block, LongWaits]:
    instr = line.instruction
    assert isinstance(instr, Wait)
    duration = instr.duration
    assert isinstance(duration, int)

    iterations = duration // MAX_WAIT
    remainder = duration % MAX_WAIT
    register = Registers.wait.value
    label = f"wait{state.n}"

    block = ([Wait(duration=remainder)] if remainder > 0 else []) + [
        Move(source=iterations, destination=register),
        Line(instruction=Wait(duration=MAX_WAIT), label=label),
        Line.instr(Loop(a=register, address=Reference(label=label))),
    ]

    return block, LongWaits(n=state.n + 1)


long_wait = LineRule(init_state=LongWaits, match=_match_long_wait, map=_map_long_wait)
"""Split a long immediate wait.

It accounts for the wait instruction limit, defined by :const:`MAX_WAIT`.

Each loop is labelled with its own unique tag, which is assigned sequentially by keeping
the count of expanded loops.
"""


CHUNK_WAIT = MAX_WAIT - 4
"""Duration of each wait in the loop of a register wait.

It is shorter than :const:`MAX_WAIT`, so that the remainder is never below the 4 ns
minimum of the wait instruction.
"""


def _match_register_wait(line: Line, state: LongWaits) -> tuple[bool, LongWaits]:
    instr = line.instruction
    return isinstance(instr, Wait) and isinstance(instr.duration, Register), state


def _map_register_wait(line: Line, state: LongWaits) -> tuple[Block, LongWaits]:
    instr = line.instruction
    assert isinstance(instr, Wait)
    source = instr.duration
    assert isinstance(source, Register)

    scratch = Registers.wait.value
    loop = f"rwait{state.n}"
    tail = f"rwait{state.n}_tail"

    # the source register is left untouched, since it is the swept value
    block = [
        Move(source=source, destination=scratch),
        Line(
            instruction=Jlt(a=scratch, b=MAX_WAIT + 1, address=Reference(label=tail)),
            label=loop,
        ),
        Wait(duration=CHUNK_WAIT),
        Sub(a=scratch, b=CHUNK_WAIT, destination=scratch),
        Jmp(address=Reference(label=loop)),
        Line(instruction=Wait(duration=scratch), label=tail),
    ]
    return block, LongWaits(n=state.n + 1)


register_wait = LineRule(
    init_state=LongWaits, match=_match_register_wait, map=_map_register_wait
)
"""Split a long wait, whose duration is read from a register.

The wait instruction only accepts up to :const:`MAX_WAIT` ns, and a swept duration is
only known at runtime, so it can not be split ahead of time::

    move  <source>, <scratch>
    loop: jlt   <scratch>, MAX_WAIT + 1, @tail
          wait  CHUNK_WAIT
          sub   <scratch>, CHUNK_WAIT, <scratch>
          jmp   @loop
    tail: wait  <scratch>

The loop is skipped by durations that already fit in a single wait. Only the preassigned
wait register is used, and the source one is not modified. Each expansion is labelled
with its own unique tag, assigned sequentially.
"""


def _start_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    instr = line.instruction
    intwait = isinstance(instr, Wait) and isinstance(instr.duration, int)
    return intwait, None


def _end_merge_wait(line: Line, state: None) -> tuple[bool, None]:
    instr = line.instruction
    intwait = isinstance(instr, Wait) and isinstance(instr.duration, int)
    return not intwait or line.label is not None, None


def _merge_wait(lines: list[Line], state: None) -> tuple[Block, None]:
    """Merge subsequent static (immediate) waits."""
    duration: int = 0
    comment: list[str] = []
    label: str | None = lines[0].label

    for line in lines:
        instr = line.instruction
        assert isinstance(instr, Wait) and isinstance(instr.duration, int)

        duration += instr.duration
        if line.comment is not None:
            comment.append(line.comment)
        if line.label is not None:
            label = line.label

    merged = (
        [
            Line(
                label=label,
                instruction=Wait(duration=duration),
                comment="\n".join(comment),
            )
        ]
        if duration > 0
        else []
    )
    return merged, None


merge_wait = BlockRule(
    init_state=lambda: None,
    start=_start_merge_wait,
    end=_end_merge_wait,
    map=_merge_wait,
)
