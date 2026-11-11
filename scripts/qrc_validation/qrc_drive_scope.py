"""Scope trace of a Gaussian pulse from O3 to I2 on the QRC (testing only).

Usage (from the repository root)::

    python scripts/qrc_validation/qrc_drive_scope.py            # dry run
    python scripts/qrc_validation/qrc_drive_scope.py --run      # on hardware, saves a PNG

Uses the ``qblox_qrc_drive_loopback`` platform (O3 cabled to I2, 20 dB attenuation
on all ports). A Gaussian pulse is played on the drive channel while the scope
acquisition (RAW) on I2 records the trace. Besides the trace, the dominant
frequency of the returned signal is printed: if the QRC already demodulates in
hardware, the software demodulation applied to RAW data (IF = probe frequency -
LO) would leave an oscillation at the IF.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qibolab import (
    Acquisition,
    AcquisitionType,
    AveragingMode,
    Gaussian,
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    create_platform,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "qblox_qrc_drive_loopback"
PULSE_NS = 1000
AMPLITUDE = 0.1  # normalized, kept low on purpose
NSHOTS = 256
SCOPE_NS = 16384  # scope memory: the trace length, and the minimum RAW duration
OUTPUT = OUTPUT_DIR / "qrc_drive_scope.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--show", action="store_true", help="open the plot window")
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)

    drive = Pulse(
        duration=PULSE_NS,
        amplitude=AMPLITUDE,
        relative_phase=0.0,
        envelope=Gaussian(rel_sigma=0.2),
    )
    readout = Readout(
        probe=Pulse(
            duration=PULSE_NS,
            amplitude=0.0,
            relative_phase=0.0,
            envelope=Rectangular(),
        ),
        acquisition=Acquisition(duration=SCOPE_NS),
    )
    sequence = PulseSequence([("q0/drive", drive), ("q0/acquisition", readout)])
    print(f"Gaussian, {PULSE_NS} ns, amplitude {AMPLITUDE}; scope {SCOPE_NS} ns")
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    platform.connect()
    try:
        res = platform.execute(
            [sequence],
            nshots=NSHOTS,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
        )
    finally:
        platform.disconnect()

    trace = np.asarray(res[readout.acquisition.id], dtype=float).reshape(-1, 2)
    z = trace[:, 0] + 1j * trace[:, 1]
    t = np.arange(len(z))  # ns, 1 GSa/s

    spectrum = np.abs(np.fft.fft(z - z.mean()))
    freqs = np.fft.fftfreq(len(z), d=1e-9)
    print(f"samples: {len(z)}, peak |z| = {np.abs(z).max():.4g}")
    print(f"dominant frequency: {freqs[spectrum.argmax()] / 1e6:.2f} MHz")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
    ax1.plot(t, z.real, label="I")
    ax1.plot(t, z.imag, label="Q")
    ax1.plot(t, np.abs(z), "k", label="|I + iQ|")
    ax1.set_xlim(0, 4 * PULSE_NS)
    ax1.set_xlabel("time (ns)")
    ax1.set_ylabel("scope signal (a.u.)")
    ax1.legend()
    ax2.plot(t, np.abs(z))
    ax2.set_xlabel("time (ns), full scope trace")
    ax2.set_ylabel("|I + iQ|")
    fig.suptitle("QRC O3 to I2: Gaussian pulse, scope acquisition")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print("Saved", OUTPUT)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
