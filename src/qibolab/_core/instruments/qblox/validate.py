from qibolab._core.components import AcquisitionChannel, Channel, IqChannel
from qibolab._core.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab._core.identifier import ChannelId
from qibolab._core.sequence import PulseSequence
from qibolab._core.sweeper import ParallelSweepers

from .config.port import PortAddress
from .limits import (
    ACQUISITION_MEMORY,
    ACQUISITION_NUMBER,
    QCM_INSTRUCTION_MEMORY,
    QRC_OUTPUTS,
    QRC_READOUT_PORTS,
    QRC_SEQUENCERS,
    QRM_INSTRUCTION_MEMORY,
    WAVEFORM_MEMORY,
    WAVEFORM_NUMBER,
    WEIGHT_MEMORY,
    WEIGHT_NUMBER,
)
from .q1asm.ast_ import Line
from .qrc import assign_sequencers
from .sequence.acquisition import Acquisition, Weight
from .sequence.sequence import Q1Sequence
from .sequence.waveforms import Waveform

__all__ = []


def assert_channels_exclusion(ps: PulseSequence, excluded: set[ChannelId]) -> None:
    """Deny group of channels.

    Mainly used to deny probe channels, since probe events are currently required to be
    bundled with acquisitions, cf. :class:`qibolab.Readout`.

    An extended discussion about the probe-acquisition merge is found at:
    https://github.com/qiboteam/qibolab/pull/1088#issuecomment-2637800857
    """
    assert not any(ch in excluded for ch, _ in ps), (
        "Probe channels can not be controlled independently, please use Readout events."
    )


def assert_waveform_memory(waveforms: list[Waveform]) -> None:
    """Assert waveform memory limitations."""
    assert len(waveforms) <= WAVEFORM_NUMBER
    assert sum(len(w.data) for w in waveforms) <= WAVEFORM_MEMORY


def assert_weight_memory(weights: list[Weight]) -> None:
    """Assert weight memory limitations."""
    assert len(weights) <= WEIGHT_NUMBER
    assert sum(len(w.data) for w in weights) <= WEIGHT_MEMORY


def assert_acquisition_memory(acquisitions: list[Acquisition]) -> None:
    """Assert acquisition memory limitations."""
    assert len(acquisitions) <= ACQUISITION_NUMBER
    assert sum(a.num_bins for a in acquisitions) <= ACQUISITION_MEMORY


def assert_instruction_memory(lines: list[Line], is_qrm: bool) -> None:
    """Assert instruction memory limitations.

    The limit depends on the hosting module kind (QRM vs QCM), not on channel type.
    """
    mem = QRM_INSTRUCTION_MEMORY if is_qrm else QCM_INSTRUCTION_MEMORY
    assert len(lines) <= mem


def validate_raw_acquisitions(
    sequence: Q1Sequence, acquisition_type: AcquisitionType
) -> None:
    """Ensure each sequencer performs no more than one RAW acquisition per run.

    A run may perform multiple scope acquisitions if they are assigned to
    different sequencers. Because a sequencer cannot acquire more than once per
    run, RAW sequences are not batched; batching could place multiple
    acquisitions on the same sequencer and violate this constraint.
    """
    if acquisition_type is not AcquisitionType.RAW:
        return

    n_acquisitions = len(sequence.acquisitions)
    if n_acquisitions > 1:
        raise ValueError(
            "RAW acquisition supports at most one acquisition pulse per "
            f"sequencer: {n_acquisitions}."
        )


def validate_sequence(
    sequence: Q1Sequence, is_qrm: bool, acquisition_type: AcquisitionType
) -> None:
    """Validate sequence memory limits and RAW acquisition constraints."""
    assert_instruction_memory(sequence.program.lines, is_qrm)
    assert_acquisition_memory(list(sequence.acquisitions.values()))
    assert_waveform_memory(list(sequence.waveforms.values()))
    assert_weight_memory(list(sequence.weights.values()))
    validate_raw_acquisitions(sequence, acquisition_type)


def validate_raw_options(
    options: ExecutionParameters, sweepers: list[ParallelSweepers]
) -> None:
    """Reject RAW sweepers and averaging modes that exceed Qblox scope limits.

    RAW acquisition supports neither sweepers nor single-shot averaging: the scope
    memory cannot retain a separate trace for every shot or sweep point. Any averaged
    mode is accepted, since the Qblox driver executes ``SEQUENTIAL`` and ``CYCLIC``
    identically (a single hardware-averaged trace).
    """
    if options.acquisition_type is not AcquisitionType.RAW:
        return

    if any(sweepers):
        raise ValueError("RAW acquisition does not support sweepers.")

    if not options.averaging_mode.average:
        # Qblox scope memory cannot retain a separate RAW trace for every shot.
        raise NotImplementedError(
            "Qblox RAW acquisition requires an averaged AveragingMode because of "
            "scope acquisition memory limits; AveragingMode.SINGLESHOT is not "
            "supported."
        )


def validate_qrc_channels(
    slot: int,
    channels: list[tuple[ChannelId, PortAddress]],
    all_channels: dict[ChannelId, Channel],
) -> None:
    """Validate the channels assigned to a QRC module.

    Two kinds of channels are supported:

    - readout: each acquisition channel must sit on a paired input/output port (``io1``
      or ``io2``) shared with its probe channel
    - drive: IQ channels on any of the outputs ``o1`` to ``o6``, as long as the output
      is not already used as a readout probe

    Besides, a sequencer must be available for each channel, according to the
    outputs they can reach (cf. :func:`.qrc.assign_sequencers`).
    """
    if len(channels) > QRC_SEQUENCERS:
        raise ValueError(
            f"QRC slot {slot} has only {QRC_SEQUENCERS} sequencers; "
            f"{len(channels)} channels are configured."
        )

    probed = set()
    for ch, address in channels:
        channel = all_channels[ch]
        if isinstance(channel, AcquisitionChannel):
            probe = all_channels.get(channel.probe)
            probe_address = (
                PortAddress.from_path(probe.path) if probe is not None else None
            )
            if (
                not address.input
                or address.ports[0] not in QRC_READOUT_PORTS
                or address.ports[1] is not None
                or probe_address is None
                or probe_address.input
                or probe_address.slot != address.slot
                or probe_address.ports[0] != address.ports[0]
                or probe_address.ports[1] is not None
            ):
                raise NotImplementedError(
                    "QRC readout currently requires a matching probe and "
                    "acquisition on paired IO port 1 or 2."
                )
            probed.add(address.ports[0])
        elif not isinstance(channel, IqChannel):
            raise NotImplementedError(
                "QRC supports only readout and IQ drive channels, "
                f"not {type(channel).__name__} ({ch!r})."
            )

    for ch, address in channels:
        if not isinstance(all_channels[ch], IqChannel):
            continue
        if (
            address.input
            or address.ports[1] is not None
            or not 1 <= address.ports[0] <= QRC_OUTPUTS
        ):
            raise NotImplementedError(
                f"QRC drive channel {ch!r} must sit on a single output port "
                f"(o1 to o{QRC_OUTPUTS}), not on {address.local_address!r}."
            )
        if address.ports[0] in probed:
            raise ValueError(
                f"QRC output o{address.ports[0]} of drive channel {ch!r} is already "
                "used by a readout probe."
            )

    assign_sequencers(channels)
