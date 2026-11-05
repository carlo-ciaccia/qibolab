"""Sequencer assignment for QRC modules.

A QRC has 12 sequencers and 6 RF outputs. The reachable outputs depend on the
sequencer (cf. ``get_allowed_channels`` in ``qblox_instruments``):

- sequencers 0-7 have an acquisition path and can drive outputs 0 and 1 (the paired
  ``io`` ports), plus one more output each: sequencers ``s`` and ``s + 4`` reach
  output ``2 + s % 4``
- sequencers 8-11 have no acquisition path and can drive outputs 2-5 only

Outputs are 0-based here, as in the hardware parameters (qibolab paths are 1-based).
"""

from .config.port import PortAddress
from .identifiers import ChannelId
from .limits import QRC_READOUT_SEQUENCERS, QRC_SEQUENCERS

__all__ = []


def allowed_sequencers(output: int, readout: bool) -> list[int]:
    """Sequencers that can serve a channel on the given (0-based) output.

    The list is sorted by preference: control-only sequencers come first, since
    those with an acquisition path are the scarcer resource.
    """
    if readout:
        return list(range(QRC_READOUT_SEQUENCERS))
    if output < 2:
        return list(range(QRC_READOUT_SEQUENCERS))
    return list(range(QRC_READOUT_SEQUENCERS, QRC_SEQUENCERS)) + [
        s for s in range(QRC_READOUT_SEQUENCERS) if s % 4 == output - 2
    ]


def assign_sequencers(
    channels: list[tuple[ChannelId, PortAddress]],
) -> dict[ChannelId, int]:
    """Assign a distinct sequencer to each channel of a QRC module.

    Raises :class:`ValueError` when no valid assignment exists. The result is
    deterministic, and depends only on the given channels.
    """
    candidates = {
        ch: allowed_sequencers(address.ports[0] - 1, address.input)
        for ch, address in channels
    }
    owner: dict[int, ChannelId] = {}

    def place(ch: ChannelId, seen: set[int]) -> bool:
        for seq in candidates[ch]:
            if seq in seen:
                continue
            seen.add(seq)
            if seq not in owner or place(owner[seq], seen):
                owner[seq] = ch
                return True
        return False

    # constrained channels first, to keep the assignment stable and compact
    for ch in sorted(candidates, key=lambda c: (len(candidates[c]), str(c))):
        if not place(ch, set()):
            raise ValueError(
                "No QRC sequencer assignment satisfies the output reachability "
                f"constraints for channels {sorted(map(str, candidates))}."
            )
    return {ch: seq for seq, ch in owner.items()}
