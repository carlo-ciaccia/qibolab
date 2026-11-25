"""Time of flight of the readout of the CQT_QPU platform.

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/time_of_flight.py                       # dry run
    python scripts/cqt_qpu/calibration/time_of_flight.py --run                # all qubits, on hardware
    python scripts/cqt_qpu/calibration/time_of_flight.py --run --qubits q2 q4

For each qubit the readout probe (the platform's MZ native gate) is played detuned by
``--detuning`` from the resonator, so that the response does not depend on the resonator,
with the acquisition ``delay`` set to 0, and a RAW trace is acquired. As in the
``cal00_time_of_flight`` experiment of the Qblox Scheduler demos, the arrival of the signal
is the first of two consecutive samples of the magnitude, after removing the complex
baseline, which are 4 standard deviations of the baseline above it. The time of flight is
that sample plus the start of the trace (``delay`` plus, where the output of the module
has a delay compensation, the 340 ns of the driver; the QRC outputs have none).

It prints, for each qubit, the time of flight and the ``delay`` to write in the
``acquisition`` config of ``parameters.json`` (the time of flight minus the delay
compensation, on the grid of 4 ns), and the value shared by the qubits that are read
out together (the largest one). Nothing is written to the platform.

The probe is the one of the MZ gate (use ``--amplitude`` to lower it, the readout line must
not saturate), and the readout frequency of the platform is moved by ``--detuning`` only
for the measurement. The NCO propagation delay compensation that the Scheduler also sets
(``nco_prop_delay_comp``) is not touched by the driver, nor by this script.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qibolab import AcquisitionType, AveragingMode, create_platform
from qibolab._core.instruments.qblox.limits import FILTER_DELAY_COMPENSATION
from qibolab._core.instruments.qblox.utils import time_of_flights

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
DETUNING = 20e6  # Hz
NSHOTS = 1024
BASELINE_NS = 40  # samples before the arrival of the signal, taken as the baseline
GRID_NS = 4
OUTPUT = OUTPUT_DIR / "cqt_time_of_flight.png"


def arrival(trace: np.ndarray) -> tuple[int | None, np.ndarray]:
    """First sample of the signal in a complex trace: two consecutive ones above 4 sigma."""
    adjusted = trace - np.mean(trace[:BASELINE_NS])
    magnitude = np.abs(adjusted)
    baseline = magnitude[:BASELINE_NS]
    threshold = baseline.mean() + 4 * baseline.std()
    above = np.flatnonzero(magnitude > threshold)
    pairs = np.flatnonzero(np.diff(above) == 1)
    if len(pairs) == 0:
        return None, magnitude
    return int(above[pairs[0]]), magnitude


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--show", action="store_true", help="open the plot window")
    parser.add_argument(
        "--qubits", nargs="+", help="default: all the qubits with a MZ gate"
    )
    parser.add_argument("--detuning", type=float, default=DETUNING, help="Hz")
    parser.add_argument(
        "--amplitude", type=float, help="override the MZ probe amplitude"
    )
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    qubits = args.qubits or [
        q for q, gates in platform.natives.single_qubit.items() if gates.MZ is not None
    ]
    print(
        f"cluster {controller.address} | qubits {qubits} | detuning {args.detuning / 1e6:g} MHz"
    )
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    results = {}
    platform.connect()
    try:
        uncompensated = controller._qrc_channels()
        configs = platform.parameters.configs
        for name in qubits:
            qubit = platform.qubits[name]
            sequence = platform.natives.single_qubit[name].MZ.create_sequence()
            readout = next(e for _, e in sequence if hasattr(e, "probe"))
            if args.amplitude is not None:
                readout.probe = readout.probe.model_copy(
                    update={"amplitude": args.amplitude}
                )
            probe, acquisition = qubit.probe, qubit.acquisition
            original = (configs[probe], configs[acquisition])
            configs[probe] = original[0].model_copy(
                update={"frequency": original[0].frequency - args.detuning}
            )
            configs[acquisition] = original[1].model_copy(update={"delay": 0.0})
            try:
                res = platform.execute(
                    [sequence],
                    nshots=NSHOTS,
                    acquisition_type=AcquisitionType.RAW,
                    averaging_mode=AveragingMode.CYCLIC,
                )
            finally:
                configs[probe], configs[acquisition] = original
            iq = np.asarray(res[readout.acquisition.id], dtype=float).reshape(-1, 2)
            start = time_of_flights({acquisition: configs[acquisition]}, uncompensated)[
                acquisition
            ]
            compensation = (
                0.0 if acquisition in uncompensated else FILTER_DELAY_COMPENSATION
            )
            sample, magnitude = arrival(iq[:, 0] + 1j * iq[:, 1])
            results[name] = (start, compensation, sample, magnitude)
    finally:
        platform.disconnect()

    print(f"\n{'qubit':>6} {'tof (ns)':>9} {'delay to set (ns)':>18}")
    delays = {}
    for name, (start, compensation, sample, _) in results.items():
        if sample is None or sample <= BASELINE_NS:
            print(
                f"{name:>6}  signal not found, or already there at the start of the trace "
                f"({start:.0f} ns after the probe): the time of flight is shorter, or the "
                "probe is too weak"
            )
            continue
        tof = start + sample
        delay = round((tof - compensation) / GRID_NS) * GRID_NS
        delays[name] = delay
        note = (
            ""
            if delay >= 0
            else "  (negative: the signal arrives before the earliest window)"
        )
        print(f"{name:>6} {tof:>9.0f} {delay:>18.0f}{note}")
    if delays:
        print(
            f"\nlargest delay (qubits read out together): {max(delays.values()):.0f} ns"
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (start, _, sample, magnitude) in results.items():
        ax.plot(start + np.arange(len(magnitude)), magnitude, label=name)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("time after the start of the probe (ns)")
    ax.set_ylabel("|I + iQ| (baseline removed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print("Saved", OUTPUT)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
