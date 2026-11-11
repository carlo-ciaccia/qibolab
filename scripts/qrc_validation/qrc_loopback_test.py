"""Loopback check of the QRC through qibolab.

Usage (from the repository root)::

    python scripts/qrc_validation/qrc_loopback_test.py          # dry run
    python scripts/qrc_validation/qrc_loopback_test.py --run    # on hardware

The dry run only loads the platform and prints what would be played. ``--run``
connects to the cluster, plays a flat readout pulse and compares the integrated
response with a zero-amplitude run (noise floor). Before using ``--run``, check
the cabling (O1 -> I1) and that the attenuation is safe for the QRC input.
"""

import argparse
import os
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PLATFORMS

PLATFORM = "qblox_qrc_loopback"
DURATION_NS = 1000
AMPLITUDE = 0.05  # normalized, kept low on purpose
NSHOTS = 256


def sequence(qubit: str, amplitude: float) -> PulseSequence:
    return PulseSequence(
        [
            (
                f"{qubit}/acquisition",
                Readout(
                    probe=Pulse(
                        duration=DURATION_NS,
                        amplitude=amplitude,
                        relative_phase=0.0,
                        envelope=Rectangular(),
                    ),
                    acquisition=Acquisition(duration=DURATION_NS),
                ),
            )
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    print("Platform:", platform.name, "| cluster address:", controller.address)
    for name, channel in platform.channels.items():
        print(f"  {name}: {channel.path}")
    for name, config in platform.parameters.configs.items():
        print(f"  {name}: {config}")

    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    signal = sequence("q0", AMPLITUDE)
    noise = sequence("q0", 0.0)
    platform.connect()
    try:
        res_signal = platform.execute(
            [signal], nshots=NSHOTS, acquisition_type=AcquisitionType.INTEGRATION
        )
        res_noise = platform.execute(
            [noise], nshots=NSHOTS, acquisition_type=AcquisitionType.INTEGRATION
        )
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
    print("PASS" if ratio > 10 else "FAIL: no clear loopback signal")


if __name__ == "__main__":
    main()
