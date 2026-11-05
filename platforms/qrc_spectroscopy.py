"""Resonator-spectroscopy style frequency sweep on the QRC loopback platform.

Usage (from the repository root)::

    python platforms/qrc_spectroscopy.py            # dry run
    python platforms/qrc_spectroscopy.py --run      # on hardware, saves a PNG

Sweeps the probe frequency and plots the integrated amplitude and phase. In the
O1 -> I1 loopback there is no resonator, so the trace shows the response of the
cabling and the module (flat apart from a slow ripple), not a resonance dip.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qibolab import (
    Acquisition,
    AcquisitionType,
    AveragingMode,
    Parameter,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    Sweeper,
    create_platform,
)

PLATFORM = "qblox_qrc_loopback"
START, STOP, STEP = 5.01e9, 5.09e9, 1e6  # Hz, LO at 5 GHz -> IF 10-90 MHz
DURATION_NS = 1000
AMPLITUDE = 0.05
NSHOTS = 256
OUTPUT = Path(__file__).parent / "qrc_spectroscopy.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(Path(__file__).parent))
    platform = create_platform(PLATFORM)

    readout = Readout(
        probe=Pulse(
            duration=DURATION_NS,
            amplitude=AMPLITUDE,
            relative_phase=0.0,
            envelope=Rectangular(),
        ),
        acquisition=Acquisition(duration=DURATION_NS),
    )
    sequence = PulseSequence([("q0/acquisition", readout)])
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        range=(START, STOP, STEP),
        channels=["q0/probe"],
    )
    freqs = np.arange(START, STOP, STEP)
    print(f"{len(freqs)} points, {START / 1e9:.3f}-{STOP / 1e9:.3f} GHz")
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    platform.connect()
    try:
        res = platform.execute(
            [sequence],
            [[sweeper]],
            nshots=NSHOTS,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
    finally:
        platform.disconnect()

    iq = np.asarray(res[readout.acquisition.id], dtype=float).reshape(-1, 2)
    z = iq[:, 0] + 1j * iq[:, 1]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(freqs / 1e9, np.abs(z))
    ax1.set_ylabel("|I + iQ| (a.u.)")
    ax2.plot(freqs / 1e9, np.unwrap(np.angle(z)))
    ax2.set_ylabel("phase (rad)")
    ax2.set_xlabel("probe frequency (GHz)")
    fig.suptitle("QRC loopback frequency sweep")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print("Saved", OUTPUT)
    plt.show()


if __name__ == "__main__":
    main()
