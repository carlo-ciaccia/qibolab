"""Unit tests for the Qblox port- and module-level configuration."""

import pytest

from qibolab._core.components import Channel, Configs, DcChannel, DcConfig
from qibolab._core.components.filters import (
    ExponentialFilter,
    FiniteImpulseResponseFilter,
)
from qibolab._core.identifier import ChannelId
from qibolab._core.instruments.qblox.config.module import ModuleConfig


def test_module_config_dc_offset_end_to_end():
    offset_val = 0.3

    channels: dict[ChannelId, Channel] = {ChannelId("flux"): DcChannel(path="4/o1")}
    configs: Configs = {ChannelId("flux"): DcConfig(offset=offset_val)}

    qcm = ModuleConfig.build(channels, configs, {}, {}, is_qcm_non_rf_type=True)
    assert qcm.ports["out0_offset"] == pytest.approx(offset_val * 2.5)

    with pytest.raises(AssertionError):
        ModuleConfig.build(channels, configs, {}, {}, is_qcm_non_rf_type=False)


class _Parameter:
    def __init__(self, values: dict, name: str):
        self.values, self.name = values, name

    def set(self, value):
        self.values[self.name] = value


class _Module:
    """A module with the given parameters, which records what is set."""

    is_qrc_type = False

    def __init__(self, *names: str):
        self.values: dict = {}
        self.parameters = {name: _Parameter(self.values, name) for name in names}

    def disconnect_outputs(self) -> None: ...

    def disconnect_inputs(self) -> None: ...

    is_qrm_type = False


def _exponential_config() -> ModuleConfig:
    channels: dict[ChannelId, Channel] = {ChannelId("flux"): DcChannel(path="4/o1")}
    configs: Configs = {
        ChannelId("flux"): DcConfig(
            offset=0.0, filters=[ExponentialFilter(amplitude=0.3, tau=100.0)]
        )
    }
    return ModuleConfig.build(channels, configs, {}, {}, is_qcm_non_rf_type=True)


def test_exponential_filter_is_written_in_seconds_where_the_module_has_tau_constant():
    config = _exponential_config()
    assert config.ports["out0_exp0_time_constant"] == pytest.approx(
        130.0
    )  # ns, tau (1 + A)
    assert config.ports["out0_exp0_amplitude"] == pytest.approx(0.3 / 1.3)

    module = _Module(*config.ports.keys(), "out0_exp0_tau_constant")
    config.apply(module, disconnect=False)  # type: ignore[arg-type]

    assert module.values["out0_exp0_tau_constant"] == pytest.approx(130e-9)
    assert module.values["out0_exp0_amplitude"] == pytest.approx(0.3 / 1.3)
    assert "out0_exp0_time_constant" not in module.values


def test_exponential_filter_is_written_in_ns_with_the_old_parameter():
    config = _exponential_config()
    module = _Module(*config.ports.keys())

    config.apply(module, disconnect=False)  # type: ignore[arg-type]

    assert module.values["out0_exp0_time_constant"] == pytest.approx(130.0)


@pytest.mark.parametrize(
    "filter_",
    [
        ExponentialFilter(amplitude=0.3, tau=4.0),
        ExponentialFilter(amplitude=0.3, tau=3e6),
        ExponentialFilter(amplitude=-0.6, tau=100.0),
        ExponentialFilter(amplitude=-1.5, tau=100.0),
        FiniteImpulseResponseFilter(coefficients=[2.0]),
        FiniteImpulseResponseFilter(coefficients=[0.5, -2.5]),
    ],
)
def test_filters_out_of_the_range_of_the_hardware_are_refused(filter_):
    channels: dict[ChannelId, Channel] = {ChannelId("flux"): DcChannel(path="4/o1")}
    configs: Configs = {ChannelId("flux"): DcConfig(offset=0.0, filters=[filter_])}

    with pytest.raises(ValueError, match="must be"):
        ModuleConfig.build(channels, configs, {}, {}, is_qcm_non_rf_type=True)


@pytest.mark.parametrize("amplitude", [0.05, 0.3, -0.2])
def test_exponential_filter_of_the_qblox_hardware_inverts_the_distortion(amplitude):
    """The correction of ``1 + A exp(-t/tau)`` is exact, not only to first order."""
    import numpy as np
    from scipy.signal import lfilter

    from qibolab._core.instruments.qblox.config.port import qblox_exponential

    tau = 100.0
    corrected, tau_q = qblox_exponential(
        ExponentialFilter(amplitude=amplitude, tau=tau)
    )
    t = np.arange(1500)
    distorted_step = 1 + amplitude * np.exp(-t / tau)
    # step response of a filter 1 - A' exp(-t/tau'): x - A' high-pass(x)
    highpass = lfilter([1, -1], [1, -np.exp(-1 / tau_q)], distorted_step)
    response = distorted_step - corrected * highpass

    assert np.abs(response[3:] - 1).max() < 5e-4
