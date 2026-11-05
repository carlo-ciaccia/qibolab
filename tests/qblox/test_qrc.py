"""Tests for QRC readout-only support."""

import uuid
from unittest.mock import Mock

import pytest
import qblox_instruments as qblox
from qblox_instruments.types import ClusterType

from qibolab._core.components import AcquisitionConfig, IqConfig, OscillatorConfig
from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.config.module import ModuleConfig
from qibolab._core.instruments.qblox.config.port import PortAddress
from qibolab._core.instruments.qblox.config.sequencer import SequencerConfig
from qibolab._core.instruments.qblox.platform import map_ports
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
        oscillator: OscillatorConfig(frequency=5e9, power=0),
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
    assert module_config.ports == {f"out{port}_in{port}_lo_freq": 5e9}

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
    addresses = _addresses(channels) * 9
    with pytest.raises(ValueError, match="only 8 readout sequencers"):
        validate_qrc_channels(7, addresses, channels)


def test_validate_qrc_channels_rejects_unpaired_probe():
    channels, _ = _readout_channels()
    probe = channels["q0/probe"]
    channels["q0/probe"] = probe.model_copy(update={"path": "7/o2"})
    with pytest.raises(NotImplementedError, match="paired IO port"):
        validate_qrc_channels(7, _addresses(channels), channels)


def test_validate_qrc_channels_rejects_control_outputs():
    channels, _ = _readout_channels()
    with pytest.raises(NotImplementedError, match="control-output"):
        validate_qrc_channels(
            7,
            [("q0/probe", PortAddress.from_path(channels["q0/probe"].path))],
            channels,
        )


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
