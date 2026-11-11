"""Tests for QRC readout and drive support."""

import uuid
from unittest.mock import Mock

import pytest
import qblox_instruments as qblox
from qblox_instruments.types import ClusterType

from qibolab._core.components import (
    AcquisitionConfig,
    DcChannel,
    IqChannel,
    IqConfig,
    OscillatorConfig,
)
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config.module import ModuleConfig
from qibolab._core.instruments.qblox.config.port import PortAddress
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig
from qibolab._core.instruments.qblox.platform import map_ports
from qibolab._core.instruments.qblox.qrc import allowed_sequencers, assign_sequencers
from qibolab._core.instruments.qblox.validate import validate_qrc_channels
from qibolab._core.qubits import Qubit


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


def test_qrc_input_attenuation_can_exceed_the_output_one():
    """The input attenuators go beyond 31.5 dB, e.g. 40 dB is used by Qblox."""
    channels, configs = _readout_channels()
    configs["q0/acquisition"] = AcquisitionConfig(delay=0, smearing=0, attenuation=40)
    module_config = ModuleConfig.build(
        channels,
        configs,
        {"q0/acquisition": configs["qrc/out0_in0"]},
        {},
        is_qcm_non_rf_type=False,
        is_qrc_type=True,
    )
    assert module_config.ports["in0_att"] == 40


@pytest.mark.parametrize("attenuation", [-0.5, 63.5, 30.25])
def test_qrc_invalid_input_attenuation_is_rejected(attenuation):
    channels, configs = _readout_channels()
    configs["q0/acquisition"] = AcquisitionConfig(
        delay=0, smearing=0, attenuation=attenuation
    )
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

    class FakeModule:
        is_qrc_type = True
        is_qrm_type = False
        is_rf_type = True

        def __init__(self):
            self.parameters = {
                name: Parameter(name)
                for name in ("in0_att", "out0_att", "out0_in0_lo_freq")
            }

    ModuleConfig(
        ports={"in0_att": 30, "out0_att": 23.5, "out0_in0_lo_freq": 5.4e9}
    ).apply(FakeModule(), disconnect=False)

    assert calls[0] == "out0_in0_lo_freq"
    assert set(calls) == {"in0_att", "out0_att", "out0_in0_lo_freq"}
