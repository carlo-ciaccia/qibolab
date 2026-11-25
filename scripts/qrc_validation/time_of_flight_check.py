"""Time of flight of the QRC loopback, as seen through the driver.

Usage (from the repository root)::

    python scripts/qrc_validation/time_of_flight_check.py            # dry run
    python scripts/qrc_validation/time_of_flight_check.py --run      # on hardware

Uses the ``qblox_qrc_loopback`` platform (output O1 cabled to input I1; the slot is
read from ``QRC_SLOT``, the address from ``QRC_CLUSTER_IP``). A short flat probe is
played and a RAW trace is acquired for several values of the acquisition ``delay``
of the platform. In the trace, the edges of the pulse give the time between the start
of the probe and the arrival of the signal at the input, ``tof = start of the trace +
edge position``. Two things are checked:

* ``tof`` must not depend on ``delay`` (the window only moves), and
* the start of the trace, which the driver puts at ``max(delay, 4)`` ns after the probe
  (the 340 ns of the filter delay compensation are only added where the output has it,
  and the QRC outputs do not), is compared with ``tof``: a trace that starts after the
  signal arrived (``tof < start``) shows only the falling edge.

A short probe is used so that the falling edge, ``probe duration`` after the rising
one, can be seen when the rising edge is before the start of the trace.
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
    Pulse,
    PulseSequence,
    Readout,
    Rectangular,
    create_platform,
)
from qibolab._core.instruments.qblox.utils import time_of_flights

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "qblox_qrc_loopback"
PROBE_NS = 200
AMPLITUDE = 0.2
NSHOTS = 512
TRACE_NS = 16384  # RAW acquisitions are as long as the scope memory
DELAYS = (0, 100, 200, 300, 400)
SMOOTH = 8
OUTPUT = OUTPUT_DIR / "time_of_flight_check.png"


def sequence(amplitude: float) -> PulseSequence:
    readout = Readout(
        probe=Pulse(
            duration=PROBE_NS,
            amplitude=amplitude,
            relative_phase=0.0,
            envelope=Rectangular(),
        ),
        acquisition=Acquisition(duration=TRACE_NS),
    )
    return PulseSequence([("q0/acquisition", readout)])


def envelope(z: np.ndarray) -> np.ndarray:
    kernel = np.ones(SMOOTH) / SMOOTH
    return np.convolve(np.abs(z), kernel, mode="same")


def edges(z: np.ndarray) -> tuple[int | None, int | None, np.ndarray]:
    """Samples of the rising and of the falling edge (None if not in the trace)."""
    env = envelope(z)
    inner = env[SMOOTH:-SMOOTH]  # the moving average is wrong at both ends of the trace
    baseline = np.median(inner[TRACE_NS // 2 :])
    peak = inner[: TRACE_NS // 2].max()
    if peak < 5 * baseline:  # only noise: the pulse is over before the trace starts
        return None, None, env
    level = (baseline + peak) / 2
    above = inner > level
    if not above.any():
        return None, None, env
    if above[0]:  # the trace starts when the signal is already there
        rise, start = None, 0
    else:
        rise = start = int(np.argmax(above))
    below = np.flatnonzero(~above[start:])
    fall = int(start + below[0]) if len(below) else None
    shift = lambda value: None if value is None else value + SMOOTH
    return shift(rise), shift(fall), env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    print(f"Probe {PROBE_NS} ns, amplitude {args.amplitude}, delays {DELAYS} ns")
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    platform.connect()
    controller = platform.instruments["cluster0"]
    try:
        for delay in DELAYS:
            configs = platform.parameters.configs
            configs["q0/acquisition"] = configs["q0/acquisition"].model_copy(
                update={"delay": float(delay)}
            )
            seq = sequence(args.amplitude)
            res = platform.execute(
                [seq],
                nshots=NSHOTS,
                acquisition_type=AcquisitionType.RAW,
                averaging_mode=AveragingMode.CYCLIC,
            )
            acquisition = seq.acquisitions[0][1]
            z = np.asarray(res[acquisition.id], dtype=float).reshape(-1, 2)
            z = z[:, 0] + 1j * z[:, 1]
            rise, fall, env = edges(z)
            start = time_of_flights(
                {"q0/acquisition": configs["q0/acquisition"]},
                controller._qrc_channels(),
            )["q0/acquisition"]
            tof_rise = None if rise is None else start + rise
            tof_fall = None if fall is None else start + fall - PROBE_NS
            rows.append((delay, start, tof_rise, tof_fall))
            ax.plot(start + np.arange(1500), env[:1500], label=f"delay {delay} ns")
    finally:
        platform.disconnect()

    def show(value):
        return "-" if value is None else f"{value:.0f}"

    print(
        f"\n{'delay':>6} {'trace start':>12} {'tof (rise)':>11} {'tof (fall)':>11}   [ns after the start of the probe]"
    )
    for delay, start, rise, fall in rows:
        print(f"{delay:>6} {start:>12.0f} {show(rise):>11} {show(fall):>11}")
    tofs = [v for _, _, r, f in rows for v in (r, f) if v is not None]
    if tofs:
        spread = max(tofs) - min(tofs)
        print(
            f"\ntime of flight: {np.median(tofs):.0f} ns (spread {spread:.0f} ns over the delays)"
        )
    ax.set_xlabel(
        "time after the start of the probe (ns), as the driver puts the trace"
    )
    ax.set_ylabel("|I + iQ| (smoothed)")
    ax.legend()
    ax.set_title("QRC loopback: RAW trace against the acquisition delay")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print("Saved", OUTPUT)


if __name__ == "__main__":
    main()
