"""Tests for QRC readout and drive support."""

import uuid
from unittest.mock import Mock

import numpy as np
import pytest
import qblox_instruments as qblox
from qblox_instruments.ieee488_2.dummy_transport import DummyScopeAcquisitionData
from qblox_instruments.types import ClusterType

from qibolab._core.components import (
    AcquisitionConfig,
    DcChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import (
    AcquisitionType,
    AveragingMode,
    ExecutionParameters,
)
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config.module import ModuleConfig
from qibolab._core.instruments.qblox.config.port import PortAddress
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig
from qibolab._core.instruments.qblox.platform import map_ports
from qibolab._core.instruments.qblox.qrc import allowed_sequencers, assign_sequencers
from qibolab._core.instruments.qblox.validate import validate_qrc_channels
from qibolab._core.pulses import Acquisition, Pulse, Readout, Rectangular
from qibolab._core.pulses.modulation import demodulate
from qibolab._core.qubits import Qubit
from qibolab._core.sequence import PulseSequence


def _readout_channels(pair=1):
    """Create one QRC readout wired to a paired output/input port."""
    qubits = {"q0": Qubit.default("q0", channels=["probe", "acquisition"])}
    channels = map_ports({"qrc": (7, {f"io{pair}": ["q0"]})}, qubits)
    port = pair - 1
    oscillator = f"qrc/out{port}_in{port}"
    channels["q0/probe"] = channels["q0/probe"].model_copy(update={"lo": oscillator})
    configs = {
        "q0/probe": IqConfig(frequency=5.1e9),
        "q0/acquisition": AcquisitionConfig(delay=0, smearing=0),
        oscillator: OscillatorConfig(frequency=5e9, power=-20),
    }
    return channels, configs


def test_map_qrc_readout_ports():
    channels, _ = _readout_channels()

    assert channels["q0/probe"].path == "7/o1"
    assert channels["q0/acquisition"].path == "7/i1"
    assert channels["q0/acquisition"].probe == "q0/probe"


@pytest.mark.parametrize("pair", [1, 2])
def test_qrc_module_and_sequencer_configuration(pair):
    channels, configs = _readout_channels(pair)
    port = pair - 1
    oscillator = f"qrc/out{port}_in{port}"
    module_config = ModuleConfig.build(
        channels,
        configs,
        {"q0/acquisition": configs[oscillator]},
        {},
        is_qcm_non_rf_type=False,
        is_qrc_type=True,
    )
    assert module_config.ports == {
        f"out{port}_in{port}_lo_freq": 5e9,
        f"out{port}_att": 20,
        f"in{port}_att": 20,
    }

    name = f"qrc_readout_{uuid.uuid4().hex[:8]}"
    cluster = qblox.Cluster(
        name,
        identifier="dummy",
        dummy_cfg={7: ClusterType.CLUSTER_QRC},
    )
    try:
        module = next(module for module in cluster.modules if module.slot_idx == 7)
        disconnect_outputs = Mock(wraps=module.disconnect_outputs)
        disconnect_inputs = Mock(wraps=module.disconnect_inputs)
        module.disconnect_outputs = disconnect_outputs
        module.disconnect_inputs = disconnect_inputs
        module_config.apply(module, disconnect=False)
        disconnect_outputs.assert_not_called()
        disconnect_inputs.assert_not_called()
        assert module.parameters[f"out{port}_in{port}_lo_freq"]() == 5e9

        sequencer_config = SequencerConfig.build(
            address=PortAddress.from_path(channels["q0/acquisition"].path),
            channel_id="q0/acquisition",
            channels=channels,
            configs=configs,
            acquisition=AcquisitionType.INTEGRATION,
            rf=True,
            qrc=True,
        )
        sequencer = module.sequencers[0]
        sequencer.connect_sequencer = lambda *_: pytest.fail(
            "QRC routing must use the dedicated route parameters, not CHANnelmap."
        )
        sequencer_config.apply(sequencer, qrc=True)

        assert getattr(sequencer, f"connect_out{port}")() == "IQ"
        assert sequencer.connect_acq() == f"in{port}"
        assert sequencer.nco_freq() == 100e6
    finally:
        cluster.close()


def _addresses(channels):
    return [
        (ch, PortAddress.from_path(channel.path))
        for ch, channel in channels.items()
        if ch != "q0/probe"
    ]


@pytest.mark.parametrize("pair", [1, 2])
def test_validate_qrc_channels_accepts_paired_readout(pair):
    channels, _ = _readout_channels(pair)
    validate_qrc_channels(7, _addresses(channels), channels)


def test_validate_qrc_channels_rejects_too_many_channels():
    channels, _ = _readout_channels()
    addresses = _addresses(channels) * 13
    with pytest.raises(ValueError, match="only 12 sequencers"):
        validate_qrc_channels(7, addresses, channels)


def test_validate_qrc_channels_rejects_unpaired_probe():
    channels, _ = _readout_channels()
    probe = channels["q0/probe"]
    channels["q0/probe"] = probe.model_copy(update={"path": "7/o2"})
    with pytest.raises(NotImplementedError, match="paired IO port"):
        validate_qrc_channels(7, _addresses(channels), channels)


def test_validate_qrc_channels_rejects_unsupported_channel_kinds():
    channels = {"q0/flux": DcChannel(path="7/o1")}
    with pytest.raises(NotImplementedError, match="readout and IQ drive"):
        validate_qrc_channels(7, [("q0/flux", PortAddress.from_path("7/o1"))], channels)


def _drive_channels(output=3, readout_pair=2):
    """Readout on a paired port, plus a drive channel on a separate output."""
    channels, configs = _readout_channels(readout_pair)
    channels["q0/drive"] = IqChannel(path=f"7/o{output}", lo=f"qrc/out{output - 1}")
    configs["q0/drive"] = IqConfig(frequency=5.05e9)
    configs[f"qrc/out{output - 1}"] = OscillatorConfig(frequency=5e9, power=-20)
    return channels, configs


def test_validate_qrc_channels_accepts_readout_and_drive():
    channels, _ = _drive_channels()
    validate_qrc_channels(7, _addresses(channels), channels)


def test_validate_qrc_channels_rejects_drive_on_readout_output():
    channels, _ = _drive_channels(output=2, readout_pair=2)
    with pytest.raises(ValueError, match="already used by a readout probe"):
        validate_qrc_channels(7, _addresses(channels), channels)


def test_validate_qrc_channels_accepts_drive_on_unused_io_output():
    channels, _ = _drive_channels(output=1, readout_pair=2)
    validate_qrc_channels(7, _addresses(channels), channels)


def test_validate_qrc_channels_rejects_drive_output_out_of_range():
    channels, _ = _drive_channels(output=7)
    with pytest.raises(NotImplementedError, match="single output port"):
        validate_qrc_channels(7, _addresses(channels), channels)


def test_qrc_sequencer_assignment_respects_reachability():
    drives = [(f"d{n}", PortAddress.from_path(f"7/o{n}")) for n in range(3, 7)]
    readout = [("r", PortAddress.from_path("7/i1"))]
    assignment = assign_sequencers(readout + drives)

    assert len(set(assignment.values())) == 5
    assert assignment["r"] < 8
    for name, address in drives:
        assert assignment[name] in allowed_sequencers(address.ports[0] - 1, False)
    # control-only sequencers are preferred for the outputs they can reach
    assert {assignment[name] for name, _ in drives} == {8, 9, 10, 11}


def test_qrc_sequencer_assignment_falls_back_to_readout_sequencers():
    # five drives on output 2: only four control-only sequencers, plus s=0 and s=4
    drives = [(f"d{n}", PortAddress.from_path("7/o3")) for n in range(6)]
    assignment = assign_sequencers(drives)
    assert sorted(assignment.values()) == [0, 4, 8, 9, 10, 11]

    drives.append(("extra", PortAddress.from_path("7/o3")))
    with pytest.raises(ValueError, match="No QRC sequencer assignment"):
        assign_sequencers(drives)


def test_qrc_drive_module_and_sequencer_configuration():
    channels, configs = _drive_channels(output=3, readout_pair=2)
    ids = {"q0/acquisition", "q0/drive"}
    los = {
        "q0/acquisition": configs["qrc/out1_in1"],
        "q0/drive": configs["qrc/out2"],
    }
    module_config = ModuleConfig.build(
        {ch: channels[ch] for ch in ids},
        configs,
        los,
        {},
        is_qcm_non_rf_type=False,
        is_qrc_type=True,
    )
    assert module_config.ports == {
        "out1_in1_lo_freq": 5e9,
        "out1_att": 20,
        "in1_att": 20,
        "out2_lo_freq": 5e9,
        "out2_att": 20,
    }

    name = f"qrc_drive_{uuid.uuid4().hex[:8]}"
    cluster = qblox.Cluster(
        name, identifier="dummy", dummy_cfg={7: ClusterType.CLUSTER_QRC}
    )
    try:
        module = next(module for module in cluster.modules if module.slot_idx == 7)
        module_config.apply(module, disconnect=False)
        assert module.parameters["out2_lo_freq"]() == 5e9
        assert module.parameters["out2_att"]() == 20

        sequencer_config = SequencerConfig.build(
            address=PortAddress.from_path(channels["q0/drive"].path),
            channel_id="q0/drive",
            channels=channels,
            configs=configs,
            acquisition=AcquisitionType.INTEGRATION,
            rf=True,
            qrc=True,
        )
        sequencer = module.sequencers[8]
        sequencer_config.apply(sequencer, qrc=True)

        assert sequencer.connect_out2() == "IQ"
        assert sequencer.nco_freq() == 50e6
    finally:
        cluster.close()


@pytest.mark.parametrize("output", [1, 2])
def test_qrc_drive_on_shared_io_output_uses_shared_lo(output):
    channels, configs = _drive_channels(output=output, readout_pair=3 - output)
    oscillator = f"qrc/out{output - 1}_in{output - 1}"
    channels["q0/drive"] = channels["q0/drive"].model_copy(update={"lo": oscillator})
    configs[oscillator] = OscillatorConfig(frequency=5e9, power=-20)
    module_config = ModuleConfig.build(
        {"q0/drive": channels["q0/drive"]},
        configs,
        {"q0/drive": configs[oscillator]},
        {},
        is_qcm_non_rf_type=False,
        is_qrc_type=True,
    )
    assert module_config.ports == {
        f"out{output - 1}_in{output - 1}_lo_freq": 5e9,
        f"out{output - 1}_att": 20,
    }


@pytest.mark.parametrize("power", [-40, 3, -20.25])
def test_qrc_invalid_attenuation_is_rejected(power):
    channels, configs = _drive_channels()
    configs["qrc/out2"] = OscillatorConfig(frequency=5e9, power=power)
    with pytest.raises(ValueError, match="attenuation"):
        ModuleConfig.build(
            {"q0/drive": channels["q0/drive"]},
            configs,
            {"q0/drive": configs["qrc/out2"]},
            {},
            is_qcm_non_rf_type=False,
            is_qrc_type=True,
        )


def test_qrc_sequencer_rejects_unknown_connection():
    config = SequencerConfig(address="out6")
    with pytest.raises(ValueError, match="Unsupported QRC connection"):
        config.apply(Mock(), qrc=True)


def test_qrc_sequencer_warns_on_ignored_iq_corrections():
    channels, configs = _readout_channels()
    configs["q0/probe"] = IqConfig(frequency=5.1e9, scale_q=1.1, phase_q=0.1)
    with pytest.warns(UserWarning, match="not supported on QRC"):
        SequencerConfig.build(
            address=PortAddress.from_path(channels["q0/acquisition"].path),
            channel_id="q0/acquisition",
            channels=channels,
            configs=configs,
            acquisition=AcquisitionType.INTEGRATION,
            rf=True,
            qrc=True,
        )


@pytest.mark.parametrize("qrc", [False, True])
def test_marker_override_only_outside_qrc(qrc):
    """RF-module markers gate the output switches, so they are forced high; the only
    QRC marker is the user digital output M1, which must not be held high."""
    channels, configs = _drive_channels()
    config = SequencerConfig.build(
        address=PortAddress.from_path(channels["q0/drive"].path),
        channel_id="q0/drive",
        channels=channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        qrc=qrc,
    )
    assert config.marker_ovr_en is (not qrc)
    assert config.marker_ovr_value == (None if qrc else 15)


def test_qrc_marker_override_is_cleared_on_the_instrument():
    channels, configs = _drive_channels()
    config = SequencerConfig.build(
        address=PortAddress.from_path(channels["q0/drive"].path),
        channel_id="q0/drive",
        channels=channels,
        configs=configs,
        acquisition=AcquisitionType.INTEGRATION,
        rf=True,
        qrc=True,
    )
    name = f"qrc_marker_{uuid.uuid4().hex[:8]}"
    cluster = qblox.Cluster(
        name, identifier="dummy", dummy_cfg={7: ClusterType.CLUSTER_QRC}
    )
    try:
        module = next(module for module in cluster.modules if module.slot_idx == 7)
        sequencer = module.sequencers[8]
        # e.g. left forced high by an earlier configuration
        sequencer.marker_ovr_en(True)
        sequencer.marker_ovr_value(15)
        config.apply(sequencer, qrc=True)
        assert sequencer.marker_ovr_en() is False
    finally:
        cluster.close()


def test_validate_modules_reports_unavailable_module():
    class UnavailableModule:
        @property
        def is_qrc_type(self):
            raise KeyError("Module at slot 7 is not available.")

    channels, _ = _readout_channels()
    cluster = Cluster(name="qrc_unavailable", address="dummy", channels=channels)
    cluster.__dict__["_modules"] = {7: UnavailableModule()}

    with pytest.raises(RuntimeError, match="may still be booting"):
        cluster._validate_modules()


def test_qrc_drive_and_readout_sequencer_routing():
    channels, configs = _drive_channels(output=3, readout_pair=2)
    name = f"qrc_route_{uuid.uuid4().hex[:8]}"
    dummy = qblox.Cluster(
        name, identifier="dummy", dummy_cfg={7: ClusterType.CLUSTER_QRC}
    )
    try:
        driver = Cluster(name=name, address="dummy", channels=channels)
        driver._cluster = dummy
        sequencers, _, _ = driver._configure_sequencers(configs)

        readout, drive = sequencers[7]["q0/acquisition"], sequencers[7]["q0/drive"]
        assert readout < 8
        assert drive >= 8
        module = driver._modules[7]
        assert module.sequencers[readout].connect_out1() == "IQ"
        assert module.sequencers[readout].connect_acq() == "in1"
        assert module.sequencers[drive].connect_out2() == "IQ"
    finally:
        dummy.close()


def test_qrc_input_attenuation_can_differ_from_output():
    channels, configs = _readout_channels()
    configs["q0/acquisition"] = AcquisitionConfig(delay=0, smearing=0, attenuation=30)
    module_config = ModuleConfig.build(
        channels,
        configs,
        {"q0/acquisition": configs["qrc/out0_in0"]},
        {},
        is_qcm_non_rf_type=False,
        is_qrc_type=True,
    )
    assert module_config.ports == {
        "out0_in0_lo_freq": 5e9,
        "out0_att": 20,
        "in0_att": 30,
    }


def test_qrc_invalid_input_attenuation_is_rejected():
    channels, configs = _readout_channels()
    configs["q0/acquisition"] = AcquisitionConfig(delay=0, smearing=0, attenuation=40)
    with pytest.raises(ValueError, match="attenuation"):
        ModuleConfig.build(
            channels,
            configs,
            {"q0/acquisition": configs["qrc/out0_in0"]},
            {},
            is_qcm_non_rf_type=False,
            is_qrc_type=True,
        )


def test_qrc_lo_frequency_is_set_before_attenuation():
    """The maximum attenuation depends on the LO frequency."""
    calls = []

    class Parameter:
        def __init__(self, name):
            self.name = name

        def set(self, value):
            calls.append(self.name)

    class Parameters(dict):
        def __missing__(self, name):
            self[name] = Parameter(name)
            return self[name]

    class FakeModule:
        is_qrc_type = True
        is_qrm_type = False
        is_rf_type = True

        def __init__(self):
            self.parameters = Parameters()

    ModuleConfig(
        ports={"in0_att": 30, "out0_att": 23.5, "out0_in0_lo_freq": 5.4e9}
    ).apply(FakeModule(), disconnect=False)

    # module-wide options (e.g. the scope ones) follow the port configurations
    ports = [name for name in calls if not name.startswith("scope_acq_")]
    assert ports[0] == "out0_in0_lo_freq"
    assert set(ports) == {"in0_att", "out0_att", "out0_in0_lo_freq"}


def test_qrc_scope_triggering_and_averaging_on_both_inputs():
    """As on QRM modules, the scope is triggered by the sequencers and averages the
    traces over the shots, on the paths of both QRC inputs (after a reset, averaging
    is disabled)."""
    name = f"qrc_scope_{uuid.uuid4().hex[:8]}"
    cluster = qblox.Cluster(
        name, identifier="dummy", dummy_cfg={7: ClusterType.CLUSTER_QRC}
    )
    try:
        cluster.reset()
        module = next(module for module in cluster.modules if module.slot_idx == 7)
        assert not module.scope_acq_avg_mode_en_path2()
        ModuleConfig(ports={}).apply(module, disconnect=False)
        for path in range(4):
            assert module.parameters[f"scope_acq_avg_mode_en_path{path}"]() is True
            assert module.parameters[f"scope_acq_trigger_mode_path{path}"]() == (
                "sequencer"
            )
    finally:
        cluster.close()


def _raw_readouts(*pairs):
    """One readout qubit on each of the given paired IO ports, probed at the LO."""
    qubits = {f"q{i}": Qubit.default(f"q{i}", channels=["probe", "acquisition"])
              for i in range(len(pairs))}  # fmt: skip
    ports: dict[str, list[str]] = {}
    for i, pair in enumerate(pairs):
        ports.setdefault(f"io{pair}", []).append(f"q{i}")
    channels = map_ports({"qrc": (7, ports)}, qubits)
    configs = {}
    for i, pair in enumerate(pairs):
        oscillator = f"qrc/out{pair - 1}_in{pair - 1}"
        channels[f"q{i}/probe"] = channels[f"q{i}/probe"].model_copy(
            update={"lo": oscillator}
        )
        # probed at the LO frequency: the traces are not demodulated in software
        configs[f"q{i}/probe"] = IqConfig(frequency=5e9)
        configs[f"q{i}/acquisition"] = AcquisitionConfig(delay=4, smearing=0)
        configs[oscillator] = OscillatorConfig(frequency=5e9, power=-20)
    return channels, configs


def _play_raw(channels, configs, readouts, samples=1000):
    """Play RAW readouts on a dummy QRC whose two inputs record distinct traces:
    (0.5, -0.5) on the first, (0.25, 0.75) on the second."""
    name = f"qrc_raw_{uuid.uuid4().hex[:8]}"
    dummy = qblox.Cluster(
        name, identifier="dummy", dummy_cfg={7: ClusterType.CLUSTER_QRC}
    )
    data = DummyScopeAcquisitionData(
        data=[(0.5, -0.5, 0.25, 0.75)] * 16384,
        out_of_range=(False,) * 4,
        avg_cnt=(8,) * 4,
    )
    reset = dummy.reset

    def reset_and_inject():
        # injected data is cleared by the reset at the beginning of each execution
        reset()
        dummy.set_dummy_scope_acquisition_data(7, None, data)

    dummy.reset = reset_and_inject
    driver = Cluster(name=name, address="dummy", channels=channels)
    driver._cluster = dummy
    sequence = PulseSequence(
        [
            (
                acquisition,
                Readout(
                    acquisition=Acquisition(duration=samples),
                    probe=Pulse(
                        duration=samples, amplitude=0.1, envelope=Rectangular()
                    ),
                ),
            )
            for acquisition in readouts
        ]
    )
    options = ExecutionParameters(
        nshots=8,
        relaxation_time=1000,
        acquisition_type=AcquisitionType.RAW,
        averaging_mode=AveragingMode.CYCLIC,
    )
    try:
        results = driver.play(configs, [sequence], options, [])
        module = driver._modules[7]
        by_input = {
            module.sequencers[s].connect_acq(): s
            for s in range(8)
            if module.sequencers[s].connect_acq() != "off"
        }
        selected = module.scope_acq_sequencer_select()
    finally:
        dummy.close()
    ids = {ch: event.id for ch, event in sequence if isinstance(event, Readout)}
    return {ch: results[ids[ch]] for ch in readouts}, by_input, selected


def _expected(trace, samples=1000):
    return demodulate(np.array(trace)[:, None] * np.ones(samples), 0, 1e9).T


def test_qrc_raw_on_both_inputs():
    """Each QRC input is recorded by its own scope, on its own pair of paths."""
    channels, configs = _raw_readouts(1, 2)
    results, by_input, selected = _play_raw(
        channels, configs, ["q0/acquisition", "q1/acquisition"]
    )
    np.testing.assert_allclose(results["q0/acquisition"], _expected((0.5, -0.5)))
    np.testing.assert_allclose(results["q1/acquisition"], _expected((0.25, 0.75)))
    # one scope sequencer for each input
    assert list(selected) == [by_input["in0"], by_input["in1"]]


def test_qrc_raw_on_the_second_input_only():
    channels, configs = _raw_readouts(2)
    results, by_input, selected = _play_raw(channels, configs, ["q0/acquisition"])
    np.testing.assert_allclose(results["q0/acquisition"], _expected((0.25, 0.75)))
    # the parameter takes a sequencer for each scope: the used one is repeated,
    # which the instrument reports back as a single index
    assert selected in (by_input["in1"], [by_input["in1"]] * 2)


def test_qrc_raw_rejects_two_sequencers_on_one_input():
    channels, configs = _raw_readouts(1, 1)
    with pytest.raises(ValueError, match="one acquisition sequencer per scope"):
        _play_raw(channels, configs, ["q0/acquisition", "q1/acquisition"])


@pytest.mark.parametrize("frequency", [5.05e9, 0.4e9, 10.2e9])
@pytest.mark.parametrize("kind", ["readout", "drive"])
def test_qrc_lo_off_grid_or_out_of_range_is_rejected(kind, frequency):
    channels, configs = _drive_channels(output=3, readout_pair=2)
    oscillator, channel = (
        ("qrc/out1_in1", "q0/acquisition")
        if kind == "readout"
        else ("qrc/out2", "q0/drive")
    )
    configs[oscillator] = OscillatorConfig(frequency=frequency, power=-20)
    with pytest.raises(ValueError, match="multiple of 100 MHz") as error:
        ModuleConfig.build(
            {channel: channels[channel]},
            configs,
            {channel: configs[oscillator]},
            {},
            is_qcm_non_rf_type=False,
            is_qrc_type=True,
        )
    # the message names the qibolab channel path, to find it in the platform (for a
    # readout, the shared LO is configured through the acquisition channel)
    assert channels[channel].path in str(error.value)


@pytest.mark.parametrize(("offset", "valid"), [(500e6, True), (-500e6, True),
                                               (501e6, False), (-700e6, False)])  # fmt: skip
def test_qrc_nco_range_is_checked(offset, valid):
    channels, configs = _drive_channels(output=3, readout_pair=2)
    configs["q0/drive"] = IqConfig(frequency=5e9 + offset)

    def build():
        return SequencerConfig.build(
            address=PortAddress.from_path(channels["q0/drive"].path),
            channel_id="q0/drive",
            channels=channels,
            configs=configs,
            acquisition=AcquisitionType.INTEGRATION,
            rf=True,
            qrc=True,
        )

    if valid:
        assert build().nco_freq == offset
    else:
        with pytest.raises(ValueError, match="beyond the NCO range"):
            build()
