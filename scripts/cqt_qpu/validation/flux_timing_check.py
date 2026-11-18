"""Timing of a flux pulse (QCM) against a drive pulse (QRC) on one qubit.

Usage (from the repository root)::

    python scripts/cqt_qpu/validation/flux_timing_check.py                 # preview
    python scripts/cqt_qpu/validation/flux_timing_check.py --run           # on hardware

The pi pulse of the drive is played at a fixed time, on a QRC channel. A flux pulse, played
by a QCM, is swept in time against it, from before the drive to after it: when the two
overlap, the flux pulse detunes the qubit and the pi pulse does not excite it, so the readout
shows the ground state. The scan of the start of the flux pulse, ``tau``, relative to the
start of the drive, shows a window at which they overlap, whose centre is at

    tau = (drive length - flux length) / 2

if the two signals reach the qubit together, e.g. -25 ns for a 50 ns pi pulse and a 100 ns
flux pulse. The flux channel is slower than the QRC one (QCM output path 110 ns plus 340 ns
of delay compensation of the outputs, against 170 ns of the QRC), so without alignment the
flux pulse has to start about 280 ns earlier: the window of the figure is at -305 ns.

The scan is run without the alignment of the driver (``align_module_latencies = False``) and
with it. The centre of the window with the alignment should be at the value above, and the
shift between the two runs about 280 ns. What is left is the delay of the cables and analog
paths, which the driver cannot know: the script prints the value to give as
``channel_delays={flux channel: value}`` to remove it (in ns, positive if the flux signal is
later than the drive one). Add it to the cluster in ``platform.py``, and run it again to check
that the window is at the expected place.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
    create_platform,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
QRC_SLOTS = (2, 5)
MARGIN = 100.0
"""Time, in ns, before and after the pulses."""


def build(platform, args, taus: np.ndarray):
    """Drive, flux swept against it, and readout after both."""
    qubit = platform.qubits[args.qubit]
    natives = platform.natives.single_qubit[args.qubit]
    rx = natives.RX()
    drive_length = rx.duration
    origin = max(-taus.min(), 0.0) + MARGIN

    sequence = PulseSequence()
    sequence.append((qubit.drive, Delay(duration=origin)))
    sequence.extend(rx)
    flux_delay = Delay(duration=origin)
    sequence.append((qubit.flux, flux_delay))
    sequence.append(
        (
            qubit.flux,
            Pulse(
                duration=args.flux_length,
                amplitude=args.flux_amplitude,
                envelope=Rectangular(),
            ),
        )
    )
    end = origin + max(drive_length, taus.max() + args.flux_length) + MARGIN
    sequence.append((qubit.acquisition, Delay(duration=end)))
    mz = natives.MZ()
    sequence.extend(mz)

    sweeper = Sweeper(
        parameter=Parameter.duration, values=origin + taus, pulses=[flux_delay]
    )
    acquisition = next(e for _, e in mz if hasattr(e, "probe")).acquisition
    return sequence, [[sweeper]], acquisition, drive_length


def window(taus: np.ndarray, signal: np.ndarray) -> tuple[float, float]:
    """Centre and width of the window where the signal differs from its plateau."""
    edge = max(len(signal) // 10, 3)
    plateau = np.median(np.r_[signal[:edge], signal[-edge:]])
    deviation = np.abs(signal - plateau)
    inside = np.flatnonzero(deviation > 0.5 * deviation.max())
    first, last = taus[inside[0]], taus[inside[-1]]
    return float((first + last) / 2), float(last - first)


def latencies_preview(controller) -> dict:
    """Latencies of the channels, without a connection to the cluster."""
    modules = {
        slot: SimpleNamespace(is_qrc_type=slot in QRC_SLOTS)
        for slot in controller._channels_by_module
    }
    controller.__dict__["_modules"] = modules
    try:
        return controller._channel_latencies()
    finally:
        controller.__dict__.pop("_modules", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--qubit", default="q2")
    parser.add_argument("--flux-amplitude", type=float, default=0.2, help="normalized")
    parser.add_argument("--flux-length", type=float, default=100, help="ns")
    parser.add_argument("--tau-min", type=float, default=-700, help="ns")
    parser.add_argument("--tau-max", type=float, default=300, help="ns")
    parser.add_argument("--step", type=float, default=8, help="ns")
    parser.add_argument("--nshots", type=int, default=1024)
    parser.add_argument("--relaxation", type=float, help="ns")
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    taus = np.arange(args.tau_min, args.tau_max + args.step / 2, args.step)
    sequence, sweepers, acquisition, drive_length = build(platform, args, taus)
    flux = platform.qubits[args.qubit].flux
    expected = (drive_length - args.flux_length) / 2

    latencies = latencies_preview(controller)
    qrc = platform.qubits[args.qubit].drive
    print(
        f"cluster {controller.address} | {args.qubit}: drive {qrc} ({drive_length:.0f} ns), "
        f"flux {flux} ({args.flux_length:.0f} ns, amplitude {args.flux_amplitude})"
    )
    print(
        f"latency of the drive channel {latencies[qrc]:.0f} ns, of the flux channel "
        f"{latencies[flux]:.0f} ns: the driver delays the QRC channels by "
        f"{latencies[flux] - latencies[qrc]:.0f} ns"
    )
    print(
        f"{len(taus)} values of tau from {taus[0]:.0f} to {taus[-1]:.0f} ns; the window is "
        f"expected at {expected:.0f} ns with the alignment and at "
        f"{expected - (latencies[flux] - latencies[qrc]):.0f} ns without it"
    )
    if not args.run:
        print("Preview only. Use --run to execute on the cluster.")
        return

    run_dir = OUTPUT_DIR / f"flux_timing_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    platform.connect()
    signals = {}
    try:
        for label, align in (("without alignment", False), ("with alignment", True)):
            controller.align_module_latencies = align
            print(f"\nscan {label} ...", flush=True)
            res = platform.execute(
                [sequence],
                sweepers,
                nshots=args.nshots,
                relaxation_time=args.relaxation,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            )
            iq = np.asarray(res[acquisition.id], dtype=float).reshape(len(taus), 2)
            signals[label] = np.abs(iq[:, 0] + 1j * iq[:, 1])
    finally:
        controller.align_module_latencies = True
        platform.disconnect()

    fig, ax = plt.subplots(figsize=(9, 5))
    centres = {}
    for (label, signal), color in zip(signals.items(), ("tab:red", "tab:blue")):
        centre, width = window(taus, signal)
        centres[label] = centre
        ax.plot(taus, signal, ".-", color=color, ms=4, lw=0.8, label=label)
        ax.axvline(centre, color=color, ls="--", lw=0.8)
        print(f"{label}: window centre {centre:+.0f} ns, width {width:.0f} ns")
    ax.axvline(expected, color="k", ls=":", lw=1, label="expected with the alignment")
    ax.set_xlabel("start of the flux pulse relative to the start of the drive (ns)")
    ax.set_ylabel("readout amplitude (a.u.)")
    ax.set_title(f"{args.qubit}: flux pulse (QCM) against drive pulse (QRC)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = run_dir / "flux_timing.png"
    fig.savefig(path, dpi=160)
    np.savez(
        run_dir / "flux_timing.npz",
        taus=taus,
        **{k.replace(" ", "_"): v for k, v in signals.items()},
    )
    print("Saved", path)

    shift = centres["with alignment"] - centres["without alignment"]
    residual = expected - centres["with alignment"]
    print(
        f"\nshift of the window by the alignment: {shift:+.0f} ns "
        f"(the driver delays the QRC channels by {latencies[flux] - latencies[qrc]:.0f} ns)"
    )
    print(
        f"window with the alignment against the expected one: {-residual:+.0f} ns; "
        f"to remove it set channel_delays={{'{flux}': {round(residual / 4) * 4}}} "
        "(ns, in the Cluster of platform.py)"
    )


if __name__ == "__main__":
    main()
