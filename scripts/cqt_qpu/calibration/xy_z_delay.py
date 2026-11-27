"""XY-Z delay: arrival time of the flux line relative to the drive line of a qubit.

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/xy_z_delay.py                  # preview
    python scripts/cqt_qpu/calibration/xy_z_delay.py --run            # on hardware
    python scripts/cqt_qpu/calibration/xy_z_delay.py --synthetic 12   # analysis on computed data

The cryoscope (Qibocal) plays the flux pulse right after the first pi/2 pulse, so it assumes
that the flux step and the drive pulses reach the qubit with no skew. This script measures
the skew, with the same method as ``cal33_xy_z_delay.py`` of the Scheduler.

A flux step is played from ``t = 0`` and kept for ``--flux-length`` ns, and the qubit is moved
far from its idle frequency. A pi pulse of the drive is scanned in time through it, and
``tau`` is the time of its CENTRE relative to the start of the flux step (0 is aligned).
A pi pulse that falls before the step excites the qubit, one that falls after it finds the
qubit off resonance and does nothing. The population goes from 1 to 0 as ``tau`` grows, and
half of the rotation is done when the step arrives in the middle of the pi pulse. The step is
fitted with an error function (the signal is projected on its principal axis, so no
calibration of the readout is needed) and its centre is the skew::

    skew > 0: the flux step reaches the qubit later than the drive.

The width of the step is about the length of the pi pulse. A much broader one points to a
distorted flux pulse. The bottom of the scan should be flat: if not, increase the flux
amplitude, the qubit has to move much further than 1 / pi pulse length.

To remove the skew add it to the delay of the flux channel of the cluster in ``platform.py``,
``channel_delays={flux channel: old + skew}`` (the script prints the value, rounded to the 4 ns
grid of the driver), and run the script again: the skew should then be within about 2 ns of
zero.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

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
MARGIN = 40.0
"""Time, in ns, between the start of the sequence and the first pulse, and between the pi
pulse and the end of the flux step."""
SETTLE = 1000.0
"""Time, in ns, that the flux is idle again before the readout."""
GRID = 4.0
"""Grid, in ns, of the delays of the channels."""


def build(platform, args, taus: np.ndarray):
    """Flux step, pi pulse of the drive swept through it, and the readout."""
    qubit = platform.qubits[args.qubit]
    natives = platform.natives.single_qubit[args.qubit]
    rx = natives.RX()
    pi_length = rx.duration
    if args.flux_length < taus.max() + pi_length / 2 + MARGIN:
        raise ValueError(
            f"The flux step ({args.flux_length} ns) ends before the last pi pulse: "
            f"increase --flux-length above {taus.max() + pi_length / 2 + MARGIN:.0f} ns."
        )
    # the first pi pulse starts at `origin + tau_min - pi_length / 2`
    origin = MARGIN + pi_length / 2 - taus.min()

    drive_delay = Delay(duration=origin)
    sequence = PulseSequence()
    sequence.append((qubit.drive, drive_delay))
    sequence.extend(rx)
    sequence.append((qubit.flux, Delay(duration=origin)))
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
    sequence.append(
        (qubit.acquisition, Delay(duration=origin + args.flux_length + SETTLE))
    )
    mz = natives.MZ()
    sequence.extend(mz)

    sweeper = Sweeper(
        parameter=Parameter.duration,
        values=origin + taus - pi_length / 2,
        pulses=[drive_delay],
    )
    acquisition = next(e for _, e in mz if hasattr(e, "probe")).acquisition
    return sequence, [[sweeper]], acquisition, pi_length


def step(tau, centre, width, amplitude, offset):
    """Error-function step going down from ``offset + amplitude`` to ``offset``."""
    return offset + amplitude * 0.5 * erfc((tau - centre) / (np.sqrt(2) * width))


def principal_axis(iq: np.ndarray) -> np.ndarray:
    """Project the points ``(I, Q)`` on their axis of largest variance."""
    centred = iq - iq.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    signal = centred @ vt[0]
    # orientation: the population is high at the start of the scan
    n = max(len(signal) // 5, 1)
    return signal if signal[:n].mean() > signal[-n:].mean() else -signal


def fit_step(taus: np.ndarray, signal: np.ndarray) -> dict:
    """Fit the step. Returns the centre (the skew), its error and the width, in ns."""
    n = max(len(signal) // 5, 1)
    top, bottom = signal[:n].mean(), signal[-n:].mean()
    half = 0.5 * (top + bottom)
    crossing = np.flatnonzero(signal < half)
    guess = taus[crossing[0]] if len(crossing) else taus.mean()
    popt, pcov = curve_fit(
        step,
        taus,
        signal,
        p0=[guess, 10.0, top - bottom, bottom],
        bounds=(
            [taus.min(), 0.5, -np.inf, -np.inf],
            [taus.max(), np.ptp(taus), np.inf, np.inf],
        ),
    )
    residual = signal - step(taus, *popt)
    return {
        "skew": popt[0],
        "error": float(np.sqrt(pcov[0, 0])),
        "width": abs(popt[1]),
        "amplitude": popt[2],
        "offset": popt[3],
        "snr": abs(popt[2]) / max(residual.std(), 1e-12),
        "fit": step(taus, *popt),
    }


def synthetic(
    taus: np.ndarray, skew: float, pi_length: float, noise: float, seed: int = 1
) -> np.ndarray:
    """COMPUTED data, not a measurement: the response if the step arrived ``skew`` ns late.

    The rotation done by the pi pulse is the fraction of it that comes before the step,
    smoothed by the pulse length. ``skew`` and ``noise`` are invented here to have a known
    answer to check the analysis against.
    """
    rng = np.random.default_rng(seed)
    population = 0.5 * erfc((taus - skew) / (np.sqrt(2) * pi_length / 2.5))
    iq = np.stack([population * 3 + 1, population * -2 + 0.5], axis=1)
    return iq + rng.normal(0, noise, iq.shape)


def report(taus, iq, flux, old_delay, title):
    """Fit, print the result and the new delay, and return the figure."""
    signal = principal_axis(iq)
    result = fit_step(taus, signal)
    skew = result["skew"]
    new = round((old_delay + skew) / GRID) * GRID
    print(
        f"flux arrives {skew:+.1f} +- {result['error']:.1f} ns after the drive "
        f"(width of the step {result['width']:.1f} ns, signal to noise {result['snr']:.1f})"
    )
    if result["snr"] < 5:
        print(
            "WARNING: the step is not clear, check the flux amplitude and the window."
        )
    print(
        f"channel_delays for {flux}: {old_delay:+.0f} ns now, set {new:+.0f} ns "
        f"(skew rounded to the {GRID:.0f} ns grid), in the Cluster of platform.py"
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(taus, signal, ".", ms=5, label="data")
    ax.plot(taus, result["fit"], "-", lw=1, label="erf fit")
    ax.axvline(
        skew, color="tab:green", ls="--", label=f"50 % point (skew) {skew:+.1f} ns"
    )
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("centre of the pi pulse relative to the flux step (ns)")
    ax.set_ylabel("signal on the principal axis (a.u.)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument(
        "--synthetic",
        type=float,
        metavar="SKEW",
        help="analyse computed data with this skew (ns), no hardware: checks the analysis only",
    )
    parser.add_argument("--qubit", default="q2")
    parser.add_argument("--flux-amplitude", type=float, default=0.2, help="normalized")
    parser.add_argument("--flux-length", type=float, default=300, help="ns")
    parser.add_argument("--tau-min", type=float, default=-100, help="ns")
    parser.add_argument("--tau-max", type=float, default=100, help="ns")
    parser.add_argument("--step", type=float, default=4, help="ns, multiple of 4")
    parser.add_argument("--nshots", type=int, default=1024)
    parser.add_argument("--relaxation", type=float, help="ns")
    args = parser.parse_args()

    if args.step % GRID:
        parser.error(f"--step has to be a multiple of {GRID:.0f} ns")
    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    flux = platform.qubits[args.qubit].flux
    old_delay = controller.channel_delays.get(flux, 0.0)
    taus = np.arange(args.tau_min, args.tau_max + args.step / 2, args.step)
    sequence, sweepers, acquisition, pi_length = build(platform, args, taus)

    print(
        f"cluster {controller.address} | {args.qubit}: drive {platform.qubits[args.qubit].drive}, "
        f"flux {flux} (step of {args.flux_length:.0f} ns, amplitude {args.flux_amplitude}), "
        f"pi pulse {pi_length:.0f} ns"
    )
    print(
        f"{len(taus)} values of tau from {taus[0]:.0f} to {taus[-1]:.0f} ns, "
        f"alignment of the driver {'on' if controller.align_module_latencies else 'off'}, "
        f"current channel_delays {old_delay:+.0f} ns"
    )

    run_dir = OUTPUT_DIR / f"xy_z_delay_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.synthetic is not None:
        print(f"COMPUTED data with an invented skew of {args.synthetic:+.1f} ns")
        iq = synthetic(taus, args.synthetic, pi_length, noise=0.05)
        title = f"{args.qubit}: SYNTHETIC data, skew {args.synthetic:+.1f} ns"
    elif not args.run:
        print("Preview only. Use --run to execute on the cluster.")
        return
    else:
        platform.connect()
        try:
            res = platform.execute(
                [sequence],
                sweepers,
                nshots=args.nshots,
                relaxation_time=args.relaxation,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            )
        finally:
            platform.disconnect()
        iq = np.asarray(res[acquisition.id], dtype=float).reshape(len(taus), 2)
        title = f"XY-Z delay of {args.qubit}"

    fig, result = report(taus, iq, flux, old_delay, title)
    run_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "xy_z_delay.png", dpi=160)
    np.savez(run_dir / "xy_z_delay.npz", taus=taus, iq=iq, skew=result["skew"])
    print("Saved", run_dir / "xy_z_delay.png")


if __name__ == "__main__":
    main()
