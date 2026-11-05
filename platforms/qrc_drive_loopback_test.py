"""Drive check of the QRC through qibolab (testing only).

Usage (from the repository root)::

    python platforms/qrc_drive_loopback_test.py          # dry run
    python platforms/qrc_drive_loopback_test.py --run    # on hardware

Cabling: output O3 -> input I2 of the QRC in slot 2. The readout on port ``io2``
measures what the drive pulse on O3 sends into I2, and is played at zero amplitude
(nothing is connected to O2). All used ports have 20 dB of attenuation (cf. the
platform). The dry run loads the platform and validates the QRC channels; ``--run``
connects, plays a drive pulse and compares the integrated response with a
zero-amplitude run (noise floor).
"""

import argparse
import os
from pathlib import Path

import numpy as np

from qibolab import (
    Acquisition,
    AcquisitionType,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    create_platform,
)

PLATFORM = "qblox_qrc_drive_loopback"
DRIVE_NS = 2000
READOUT_NS = 1000
AMPLITUDE = 0.05  # normalized, kept low on purpose
NSHOTS = 256


def sequence(amplitude: float) -> PulseSequence:
    drive = Pulse(
        duration=DRIVE_NS,
        amplitude=amplitude,
        relative_phase=0.0,
        envelope=Rectangular(),
    )
    readout = Readout(
        probe=Pulse(
            duration=READOUT_NS,
            amplitude=0.0,
            relative_phase=0.0,
            envelope=Rectangular(),
        ),
        acquisition=Acquisition(duration=READOUT_NS),
    )
    return PulseSequence([("q0/drive", drive), ("q0/acquisition", readout)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(Path(__file__).parent))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    print("Platform:", platform.name, "| cluster address:", controller.address)
    for name, channel in platform.channels.items():
        print(f"  {name}: {channel.path}")
    for name, config in platform.parameters.configs.items():
        print(f"  {name}: {config}")

    if not args.run:
        # imported here only: importing it at module level changes the import order,
        # which leaves an unrelated asyncio error on Windows at exit
        from qibolab._core.instruments.qblox.validate import validate_qrc_channels

        for slot, chs in controller._channels_by_module.items():
            validate_qrc_channels(slot, chs, controller.channels)
        print("QRC channels are valid.")
        print("Dry run only. Use --run to execute on the cluster.")
        return

    signal = sequence(AMPLITUDE)
    noise = sequence(0.0)
    platform.connect()
    try:
        res_signal = platform.execute(
            [signal], nshots=NSHOTS, acquisition_type=AcquisitionType.INTEGRATION
        )
        res_noise = platform.execute(
            [noise], nshots=NSHOTS, acquisition_type=AcquisitionType.INTEGRATION
        )
        # attenuation actually applied by the hardware, in dB
        module = controller._modules[2]
        attenuation = {
            name: module.parameters[name]()
            for name in ("out1_att", "in1_att", "out2_att")
        }
        print("attenuation (dB):", attenuation)
    finally:
        platform.disconnect()

    iq_signal = np.asarray(res_signal[signal.acquisitions[0][1].id], dtype=float)
    iq_noise = np.asarray(res_noise[noise.acquisitions[0][1].id], dtype=float)
    for label, iq in (("signal", iq_signal), ("noise", iq_noise)):
        assert iq.shape == (NSHOTS, 2) and np.isfinite(iq).all(), iq.shape
        print(f"{label}: mean I/Q = {iq.mean(axis=0)}, std = {iq.std(axis=0)}")

    amp_signal = np.abs(complex(*iq_signal.mean(axis=0)))
    amp_noise = np.abs(complex(*iq_noise.mean(axis=0)))
    ratio = amp_signal / max(amp_noise, 1e-12)
    print(f"|signal| / |noise| = {ratio:.1f}")
    print("PASS" if ratio > 10 else "FAIL: no clear drive signal")


if __name__ == "__main__":
    main()
