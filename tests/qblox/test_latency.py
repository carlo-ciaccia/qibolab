"""Alignment of the signals of channels in modules with a different output latency."""

from types import SimpleNamespace

import pytest

from qibolab._core.components import AcquisitionConfig
from qibolab._core.execution_parameters import ExecutionParameters
from qibolab._core.instruments.qblox.cluster import (
    Cluster,
    _align_latencies,
    _batch_sequences,
)
from qibolab._core.instruments.qblox.limits import (
    FILTER_DELAY_COMPENSATION,
    OUTPUT_PATH_LATENCY,
    QRC_OUTPUT_PATH_LATENCY,
)
from qibolab._core.instruments.qblox.platform import map_ports
from qibolab._core.instruments.qblox.utils import time_of_flights
from qibolab._core.pulses import Acquisition, Delay, Pulse, Readout, Rectangular
from qibolab._core.qubits import Qubit
from qibolab._core.sequence import PulseSequence

QCM_LATENCY = OUTPUT_PATH_LATENCY + FILTER_DELAY_COMPENSATION
OFFSET = QCM_LATENCY - QRC_OUTPUT_PATH_LATENCY
"""Delay of the QRC channels with respect to the QCM ones: 280 ns."""

READOUT, DRIVE, FLUX = "q0/acquisition", "q1/drive", "q1/flux"


def pulse(duration: float = 20) -> Pulse:
    return Pulse(duration=duration, amplitude=0.1, envelope=Rectangular())


def cluster(**kwargs) -> Cluster:
    """A cluster with a QRC in slot 7 (readout and drive) and a QCM in slot 8 (flux)."""
    qubits = {
        "q0": Qubit.default("q0", channels=["probe", "acquisition"]),
        "q1": Qubit.default("q1", channels=["drive", "flux"]),
    }
    channels = map_ports(
        {"qrc": (7, {"io1": ["q0"], 3: ["q1"]}), "qcm": (8, {1: ["q1"]})}, qubits
    )
    result = Cluster(name="latency", address="dummy", channels=channels, **kwargs)
    # the module types are read from the connected cluster
    result.__dict__["_modules"] = {
        7: SimpleNamespace(is_qrc_type=True),
        8: SimpleNamespace(is_qrc_type=False),
    }
    return result


def delays(sequence: PulseSequence) -> dict[str, float]:
    """Delays at the beginning of each channel."""
    first = {}
    for ch, event in sequence:
        first.setdefault(ch, event)
    return {ch: ev.duration for ch, ev in first.items() if isinstance(ev, Delay)}


def test_latencies_of_the_channels():
    latencies = cluster()._channel_latencies()

    # the readout, with its probe, and the drive are on the QRC, the flux on the QCM
    assert latencies[READOUT] == QRC_OUTPUT_PATH_LATENCY
    assert latencies[DRIVE] == QRC_OUTPUT_PATH_LATENCY
    assert latencies[FLUX] == QCM_LATENCY
    assert OFFSET == 280


def test_latencies_with_the_alignment_off_are_only_the_delays_given():
    off = cluster(align_module_latencies=False, channel_delays={FLUX: 12.0})

    assert off._channel_latencies() == {READOUT: 0.0, DRIVE: 0.0, FLUX: 12.0}


def test_channel_delays_are_added_to_the_latency_of_the_module():
    trimmed = cluster(channel_delays={FLUX: -20.0})

    assert trimmed._channel_latencies()[FLUX] == QCM_LATENCY - 20.0


def test_delays_for_unknown_channels_are_rejected():
    with pytest.raises(ValueError, match="q9/flux"):
        cluster(channel_delays={"q9/flux": 4.0})._channel_latencies()


def test_the_qrc_channels_start_later_than_the_qcm_ones():
    sequence = PulseSequence([(FLUX, pulse()), (DRIVE, pulse())])

    aligned = _align_latencies(sequence, cluster()._channel_latencies())

    assert delays(aligned) == {DRIVE: OFFSET}
    # the original pulses are kept
    assert [e for _, e in aligned if not isinstance(e, Delay)] == [
        e for _, e in sequence
    ]


def test_the_readout_is_delayed_with_its_probe_and_acquisition_together():
    """The time of flight is inside the readout, so delaying its channel keeps it."""
    mz = Readout(acquisition=Acquisition(duration=100), probe=pulse(100))
    sequence = PulseSequence([(FLUX, pulse()), (READOUT, mz)])

    aligned = _align_latencies(sequence, cluster()._channel_latencies())

    assert delays(aligned) == {READOUT: OFFSET}
    assert (READOUT, mz) in list(aligned)


def test_the_time_of_flight_of_the_readout_is_the_same_with_the_alignment():
    options = ExecutionParameters(nshots=1, relaxation_time=100)
    configs = {READOUT: AcquisitionConfig(delay=300, smearing=0, threshold=0.0)}
    mz = Readout(acquisition=Acquisition(duration=100), probe=pulse(100))
    sequence = PulseSequence([(FLUX, pulse()), (READOUT, mz)])
    latencies = cluster()._channel_latencies()

    def readout(latencies) -> Readout:
        (batched,) = _batch_sequences(
            [sequence], [], options, {FLUX}, {READOUT}, configs, None, latencies
        )
        return next(e for _, e in batched if isinstance(e, Readout))

    assert readout(latencies).time_of_flight == readout(None).time_of_flight
    assert readout(latencies).time_of_flight == 300 + FILTER_DELAY_COMPENSATION


def test_a_sequence_with_channels_of_the_same_latency_is_not_changed():
    qrc_only = PulseSequence([(DRIVE, pulse()), (READOUT, pulse())])
    qcm_only = PulseSequence([(FLUX, pulse())])
    latencies = cluster()._channel_latencies()

    assert _align_latencies(qrc_only, latencies) == qrc_only
    assert _align_latencies(qcm_only, latencies) == qcm_only


def test_without_latencies_the_sequence_is_not_changed():
    sequence = PulseSequence([(FLUX, pulse()), (DRIVE, pulse())])

    assert _align_latencies(sequence, None) == sequence
    assert _align_latencies(sequence, {}) == sequence


def test_delays_are_on_the_grid():
    # 433 - 170 = 263 ns, rounded to the closest multiple of 4 ns
    latencies = cluster(channel_delays={FLUX: -17.0})._channel_latencies()
    sequence = PulseSequence([(FLUX, pulse()), (DRIVE, pulse())])

    assert delays(_align_latencies(sequence, latencies)) == {DRIVE: 264}


def test_a_trim_can_compensate_a_measured_delay():
    """A measured offset of 260 ns, instead of 280, is obtained with a trim of -20 ns."""
    latencies = cluster(channel_delays={FLUX: -20.0})._channel_latencies()
    sequence = PulseSequence([(FLUX, pulse()), (DRIVE, pulse())])

    assert delays(_align_latencies(sequence, latencies)) == {DRIVE: 260}


def test_batched_sequences_have_the_qrc_channels_delayed():
    options = ExecutionParameters(nshots=1, relaxation_time=100)
    sequence = PulseSequence([(FLUX, pulse()), (DRIVE, pulse())])
    latencies = cluster()._channel_latencies()

    (batched,) = _batch_sequences(
        [sequence], [], options, {FLUX}, {DRIVE, READOUT}, {}, None, latencies
    )

    # both pulses are 20 ns long: the QRC one starts 280 ns after the QCM one
    assert batched.channel_duration(FLUX) == pytest.approx(20)
    assert batched.channel_duration(DRIVE) == pytest.approx(OFFSET + 20)
    unaligned = _batch_sequences(
        [sequence], [], options, {FLUX}, {DRIVE, READOUT}, {}, None, None
    )[0]
    assert unaligned.channel_duration(DRIVE) == pytest.approx(20)


def test_the_qrc_channels_are_the_ones_without_delay_compensation():
    assert cluster()._qrc_channels() == {READOUT, DRIVE}


def test_the_time_of_flight_of_a_qrc_readout_is_its_delay():
    """The QRC outputs have no delay compensation: its probe is not 340 ns late."""
    other = "q9/acquisition"
    configs = {
        READOUT: AcquisitionConfig(delay=324, smearing=0, threshold=0.0),
        other: AcquisitionConfig(delay=324, smearing=0, threshold=0.0),
    }

    assert time_of_flights(configs) == {
        READOUT: 324 + FILTER_DELAY_COMPENSATION,
        other: 324 + FILTER_DELAY_COMPENSATION,
    }
    assert time_of_flights(configs, {READOUT}) == {
        READOUT: 324,
        other: 324 + FILTER_DELAY_COMPENSATION,
    }
    short = {READOUT: AcquisitionConfig(delay=0, smearing=0, threshold=0.0)}
    assert time_of_flights(short, {READOUT}) == {READOUT: 4}  # the sequencer grid


def test_the_batched_readout_of_a_qrc_has_the_time_of_flight_without_the_340_ns():
    options = ExecutionParameters(nshots=1, relaxation_time=100)
    configs = {READOUT: AcquisitionConfig(delay=300, smearing=0, threshold=0.0)}
    mz = Readout(acquisition=Acquisition(duration=100), probe=pulse(100))
    sequence = PulseSequence([(FLUX, pulse()), (READOUT, mz)])
    target = cluster()

    (batched,) = _batch_sequences(
        [sequence],
        [],
        options,
        {FLUX},
        {READOUT},
        configs,
        None,
        target._channel_latencies(),
        target._qrc_channels(),
    )

    assert next(e for _, e in batched if isinstance(e, Readout)).time_of_flight == 300
