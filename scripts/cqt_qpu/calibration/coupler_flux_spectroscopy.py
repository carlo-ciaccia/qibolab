"""q2 spectroscopy as a function of the flux of each coupler connected to it.

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/coupler_flux_spectroscopy.py            # dry run
    python scripts/cqt_qpu/calibration/coupler_flux_spectroscopy.py --run      # on hardware
    python scripts/cqt_qpu/calibration/coupler_flux_spectroscopy.py --run --show

The couplers of ``q2`` are ``tc1_2`` and ``tc2_5``. They are measured one after the
other, and the plot of each one is produced as soon as its measurement is done, while the
other coupler stays parked at its configured static offset (0 by default).

Each measurement is a single 2D hardware sweep, with the coupler flux as outer loop and
the frequency as inner loop. The flux is a long rectangular DC pulse, played from the
coupler flux channel, and covering the drive and the readout::

    flux   : |‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾|   amplitude swept
    drive  : |  settle  |‾‾‾ drive pulse ‾‾‾|                   frequency swept
    probe  :                                  |‾ readout ‾|

Rectangular pulses are compiled by the Qblox driver as AWG offsets, so they use no
waveform memory, whatever their duration. The offset is set back to 0 at the end of each
shot. The flux amplitude is normalized, 1 corresponds to 2.5 V.

By default the qubit is measured, sweeping the drive frequency (the resonance is a peak
or a dip of the integrated readout). With ``--target resonator`` the probe frequency is
swept instead, without any drive.
"""

import argparse
import os
import sys
from pathlib import Path

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
QUBIT = "q2"
COUPLERS = ["tc1_2", "tc2_5"]
NSHOTS = 1024
OUTPUT = OUTPUT_DIR


def build(platform, coupler: str, args):
    """Sequence and sweepers measuring ``QUBIT`` against the flux of ``coupler``."""
    qubit = platform.qubits[QUBIT]
    mz = platform.natives.single_qubit[QUBIT].MZ
    readout = mz.create_sequence()

    qubit_target = args.target == "qubit"
    freq_channel = qubit.drive if qubit_target else qubit.probe
    center = platform.parameters.configs[freq_channel].frequency
    freqs = center + np.arange(-args.span / 2, args.span / 2 + args.step / 2, args.step)

    # the drive goes first, after the time the flux needs to settle, then the readout
    sequence = PulseSequence()
    if qubit_target:
        sequence.append((qubit.drive, Delay(duration=args.settle)))
        sequence.append(
            (
                qubit.drive,
                Pulse(
                    duration=args.drive_duration,
                    amplitude=args.drive_amplitude,
                    envelope=Rectangular(),
                ),
            )
        )
    else:
        # the probe channel can only be used through readout events, so the settling
        # time is a delay of the acquisition channel, which the readout belongs to
        sequence.append((qubit.acquisition, Delay(duration=args.settle)))
    sequence |= readout

    # the flux pulse spans everything, its amplitude is a placeholder replaced by the sweep
    flux_pulse = Pulse(
        duration=sequence.duration, amplitude=0.0, envelope=Rectangular()
    )
    sequence.append((platform.couplers[coupler].flux, flux_pulse))

    fluxes = np.linspace(-args.flux_max, args.flux_max, args.flux_points)
    sweepers = [
        [Sweeper(parameter=Parameter.amplitude, values=fluxes, pulses=[flux_pulse])],
        [Sweeper(parameter=Parameter.frequency, values=freqs, channels=[freq_channel])],
    ]
    acquisition = next(e for _, e in readout if hasattr(e, "probe")).acquisition
    return sequence, sweepers, acquisition, fluxes, freqs, center


def plot(coupler, fluxes, freqs, center, z, args):
    amp = np.abs(z)
    # remove the dependence of the background on the flux, row by row
    contrast = amp - np.median(amp, axis=1, keepdims=True)
    extremum = freqs[np.abs(contrast).argmax(axis=1)]

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(fluxes, freqs / 1e9, contrast.T, shading="auto")
    ax.plot(fluxes, extremum / 1e9, "r.", ms=4, label="largest deviation")
    ax.axhline(center / 1e9, color="w", ls="--", lw=0.8, label="configured")
    fig.colorbar(mesh, label="|I + iQ| minus row median (a.u.)")
    ax.set_xlabel(f"{coupler} flux amplitude (normalized, 1 = 2.5 V)")
    ax.set_ylabel(f"{args.target} frequency (GHz)")
    ax.set_title(f"CQT_QPU {QUBIT} {args.target} spectroscopy vs {coupler} flux")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = OUTPUT / f"cqt_{QUBIT}_{coupler}_flux_spectroscopy.png"
    fig.savefig(path, dpi=150)
    print("Saved", path)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--show", action="store_true", help="open the plot windows")
    parser.add_argument("--target", choices=["qubit", "resonator"], default="qubit")
    parser.add_argument("--couplers", nargs="+", default=COUPLERS)
    parser.add_argument("--flux-max", type=float, default=0.05, help="normalized")
    parser.add_argument("--flux-points", type=int, default=21)
    parser.add_argument("--span", type=float, default=40e6, help="Hz")
    parser.add_argument("--step", type=float, default=0.5e6, help="Hz")
    parser.add_argument("--drive-amplitude", type=float, default=0.01)
    parser.add_argument("--drive-duration", type=float, default=2000, help="ns")
    parser.add_argument("--settle", type=float, default=1000, help="ns")
    parser.add_argument("--nshots", type=int, default=NSHOTS)
    parser.add_argument(
        "--relaxation", type=float, help="ns, the one of the platform by default"
    )
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]

    plans = {c: build(platform, c, args) for c in args.couplers}
    for coupler, (sequence, _, _, fluxes, freqs, center) in plans.items():
        print(
            f"cluster {controller.address} | {coupler} | {args.target} around "
            f"{center / 1e9:.5f} GHz | flux {fluxes[0]:+.3f}..{fluxes[-1]:+.3f} "
            f"({len(fluxes)} points) | {len(freqs)} frequencies | "
            f"{sequence.duration:.0f} ns per shot"
        )
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    platform.connect()
    figures = []
    try:
        for coupler, (sequence, sweepers, acq, fluxes, freqs, center) in plans.items():
            print(f"Measuring {coupler} ...")
            res = platform.execute(
                [sequence],
                sweepers,
                nshots=args.nshots,
                relaxation_time=args.relaxation,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            )
            iq = np.asarray(res[acq.id], dtype=float).reshape(
                len(fluxes), len(freqs), 2
            )
            z = iq[..., 0] + 1j * iq[..., 1]
            figures.append(plot(coupler, fluxes, freqs, center, z, args))
            if args.show:
                # draw this plot now, while the next coupler is being measured
                plt.show(block=False)
                plt.pause(0.1)
    finally:
        platform.disconnect()
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
