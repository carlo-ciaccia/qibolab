"""Resonator spectroscopy on the CQT_QPU platform (readout only, no drive).

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/resonator_spectroscopy.py            # dry run
    python scripts/cqt_qpu/calibration/resonator_spectroscopy.py --run      # on hardware

Sweeps the probe frequency around the readout frequency of the device config, using
the platform's MZ native gate (1 us square probe, amplitude 0.3). The drive is not
played. Saves a PNG with the amplitude and phase of the integrated response, and
prints the frequency of the amplitude minimum.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper, create_platform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
QUBIT = "q2"
SPAN, STEP = 40e6, 0.5e6  # Hz, around the configured readout frequency
NSHOTS = 1024
OUTPUT = OUTPUT_DIR / "cqt_resonator_spectroscopy.png"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--show", action="store_true", help="open the plot window")
    parser.add_argument(
        "--amplitude", type=float, help="override the MZ probe amplitude"
    )
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]

    mz = platform.natives.single_qubit[QUBIT].MZ
    sequence = mz.create_sequence()
    readout = next(e for _, e in sequence if hasattr(e, "probe"))
    if args.amplitude is not None:
        readout.probe = readout.probe.model_copy(update={"amplitude": args.amplitude})
    probe = platform.qubits[QUBIT].probe
    center = platform.parameters.configs[probe].frequency
    freqs = center + np.arange(-SPAN / 2, SPAN / 2 + STEP / 2, STEP)
    sweeper = Sweeper(
        parameter=Parameter.frequency,
        values=freqs,
        channels=[probe],
    )
    print(
        f"cluster {controller.address} | readout {center / 1e9:.5f} GHz | "
        f"{len(freqs)} points, {freqs[0] / 1e9:.4f}-{freqs[-1] / 1e9:.4f} GHz | "
        f"probe amplitude {readout.probe.amplitude}"
    )
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
    amp = np.abs(z)
    dip = freqs[amp.argmin()]
    print(f"amplitude: min {amp.min():.4g} at {dip / 1e9:.5f} GHz, max {amp.max():.4g}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(freqs / 1e9, amp)
    ax1.axvline(center / 1e9, color="gray", ls="--", label="configured")
    ax1.axvline(dip / 1e9, color="r", ls=":", label="minimum")
    ax1.set_ylabel("|I + iQ| (a.u.)")
    ax1.legend()
    ax2.plot(freqs / 1e9, np.unwrap(np.angle(z)))
    ax2.set_ylabel("phase (rad)")
    ax2.set_xlabel("probe frequency (GHz)")
    fig.suptitle(f"CQT_QPU {QUBIT} resonator spectroscopy")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print("Saved", OUTPUT)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
