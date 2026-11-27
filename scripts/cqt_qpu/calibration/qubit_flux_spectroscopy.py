"""Pulsed flux spectroscopy of a qubit: its frequency against the flux pulse amplitude.

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/qubit_flux_spectroscopy.py                  # preview
    python scripts/cqt_qpu/calibration/qubit_flux_spectroscopy.py --run            # on hardware
    python scripts/cqt_qpu/calibration/qubit_flux_spectroscopy.py --synthetic      # analysis on computed data

Translation of ``cal07_pulsed_flux_qubit_spectroscopy.py`` of the Scheduler. For every
amplitude of a flux pulse, the drive frequency is swept and a drive pulse is played while the
flux is on, then the flux goes back to idle and the qubit is read out::

    flux   : |‾‾‾‾‾‾‾ amplitude swept ‾‾‾‾‾‾‾‾|                    (idle again for the readout)
    drive  :  |settle|‾‾ drive pulse (native RX), frequency swept ‾|
    probe  :                                      |settle| readout

The qubit frequency of every amplitude is the centre of a Lorentzian fitted to the readout
(peak or dip, found automatically), and a parabola is fitted to the frequencies.

The result is what the cryoscope of Qibocal needs as ``flux_coefficients``: the detuning
in GHz against the amplitude of the flux pulse, as the coefficients of a polynomial of degree 2
(highest power first, as ``np.polyval``), with the frequency of the qubit at amplitude 0
(the idle flux) as reference, so the last coefficient is 0. They are in the
``calibration.json`` of the Qibocal platform folder, under
``single_qubits > q2 > qubit > flux_coefficients``. The script prints the list and does not
write it (nothing in the repository does). Choose the amplitude of the cryoscope so that the
detuning is below 0.5 GHz (the Nyquist limit at 1 GSps), and inside the range scanned here.

The pulse amplitude is normalized, 1 is 2.5 V, and it is added to the static offset of the
flux channel. If the vertex of the parabola is not at amplitude 0 the qubit is not parked at
its sweet spot: the cryoscope of Qibocal assumes the drive is on resonance at amplitude 0.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
MIN_CONTRAST = 4.0
"""Minimum ratio between the contrast of a line and the noise of the row to use it."""


def build(platform, args):
    """Flux pulse swept in amplitude, drive swept in frequency, readout after both."""
    qubit = platform.qubits[args.qubit]
    natives = platform.natives.single_qubit[args.qubit]
    rx = natives.RX()
    mz = natives.MZ()

    center = platform.parameters.configs[qubit.drive].frequency
    freqs = center + np.arange(
        args.freq_start, args.freq_stop + args.freq_step / 2, args.freq_step
    )
    amplitudes = np.linspace(-args.flux_max, args.flux_max, args.flux_points)

    sequence = PulseSequence()
    sequence.append((qubit.drive, Delay(duration=args.settle)))
    sequence.extend(rx)
    flux_pulse = Pulse(
        duration=args.settle + rx.duration + args.tail,
        amplitude=0.0,
        envelope=Rectangular(),
    )
    sequence.append((qubit.flux, flux_pulse))
    sequence.append(
        (qubit.acquisition, Delay(duration=flux_pulse.duration + args.settle_readout))
    )
    sequence.extend(mz)

    sweepers = [
        [
            Sweeper(
                parameter=Parameter.amplitude, values=amplitudes, pulses=[flux_pulse]
            )
        ],
        [Sweeper(parameter=Parameter.frequency, values=freqs, channels=[qubit.drive])],
    ]
    acquisition = next(e for _, e in mz if hasattr(e, "probe")).acquisition
    return sequence, sweepers, acquisition, amplitudes, freqs, center


def lorentzian(x, x0, gamma, height, offset):
    return offset + height * gamma**2 / ((x - x0) ** 2 + gamma**2)


def line_center(freqs, row):
    """Centre of the Lorentzian (peak or dip) of one row, and the contrast / noise."""
    baseline = np.median(row)
    noise = 1.4826 * np.median(np.abs(row - baseline)) + 1e-12
    deviation = row - baseline
    k = np.argmax(np.abs(deviation))
    guess = [freqs[k], 5e6, deviation[k], baseline]
    span = freqs[-1] - freqs[0]
    try:
        popt, _ = curve_fit(
            lorentzian,
            freqs,
            row,
            p0=guess,
            bounds=(
                [freqs[0], 0.5e6, -np.inf, -np.inf],
                [freqs[-1], span, np.inf, np.inf],
            ),
        )
    except RuntimeError:
        return np.nan, 0.0
    return popt[0], abs(popt[2]) / noise


def analyse(amplitudes, freqs, amp):
    """Frequency of the qubit for every amplitude and the parabola through them.

    ``amp`` is the amplitude of the readout, shape (amplitudes, frequencies). The
    coefficients are of the detuning in GHz, referred to the fit at amplitude 0.
    """
    centres = np.empty(len(amplitudes))
    contrast = np.empty(len(amplitudes))
    for i, row in enumerate(amp):
        centres[i], contrast[i] = line_center(freqs, row)
    good = np.isfinite(centres) & (contrast > MIN_CONTRAST)
    if good.sum() < 5:
        raise RuntimeError(
            f"Only {good.sum()} of {len(good)} lines are clear: check the window of the "
            "frequency and the amplitude of the drive."
        )
    a, b, c = np.polyfit(amplitudes[good], centres[good] / 1e9, 2)
    rest = c  # frequency of the fit at amplitude 0, in GHz
    residual = centres[good] / 1e9 - np.polyval([a, b, c], amplitudes[good])
    vertex = -b / (2 * a) if a else np.nan
    return {
        "centres": centres,
        "good": good,
        "contrast": contrast,
        "coefficients": [a, b, 0.0],
        "frequency_at_zero": rest,
        "vertex": vertex,
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
    }


def synthetic(amplitudes, freqs, center, noise=0.15, seed=1):
    """COMPUTED data, not a measurement: dips on an invented parabola.

    The frequency is ``center - 1.3 GHz * amplitude**2`` and the lines are 8 MHz wide: they
    are invented here to have a known answer to check the analysis against.
    """
    rng = np.random.default_rng(seed)
    f = center - 1.3e9 * amplitudes**2
    z = 1 - 0.6 * 4e6**2 / ((freqs[None, :] - f[:, None]) ** 2 + 4e6**2)
    return z + rng.normal(0, noise * 0.2, z.shape), f


def plot(amplitudes, freqs, amp, result, title):
    fig, (ax_map, ax_fit) = plt.subplots(1, 2, figsize=(12, 5))
    contrast = amp - np.median(amp, axis=1, keepdims=True)
    mesh = ax_map.pcolormesh(amplitudes, freqs / 1e9, contrast.T, shading="auto")
    fig.colorbar(mesh, ax=ax_map, label="|I + iQ| minus row median (a.u.)")
    ax_map.plot(
        amplitudes[result["good"]],
        result["centres"][result["good"]] / 1e9,
        "w.",
        ms=5,
        label="Lorentzian centres",
    )
    ax_map.set_xlabel("flux pulse amplitude (normalized)")
    ax_map.set_ylabel("drive frequency (GHz)")
    ax_map.legend(loc="lower center")

    a, b, c = np.polyfit(
        amplitudes[result["good"]], result["centres"][result["good"]] / 1e9, 2
    )
    fine = np.linspace(amplitudes.min(), amplitudes.max(), 300)
    ax_fit.plot(
        amplitudes[result["good"]],
        result["centres"][result["good"]] / 1e9,
        "o",
        label="centres",
    )
    ax_fit.plot(fine, np.polyval([a, b, c], fine), "r-", label="parabola")
    ax_fit.axvline(
        result["vertex"], color="C4", ls="--", label=f"vertex {result['vertex']:+.3f}"
    )
    ax_fit.set_xlabel("flux pulse amplitude (normalized)")
    ax_fit.set_ylabel("qubit frequency (GHz)")
    ax_fit.grid(alpha=0.3)
    ax_fit.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="analyse computed data, no hardware: checks the analysis only",
    )
    parser.add_argument("--qubit", default="q2")
    parser.add_argument("--flux-max", type=float, default=0.45, help="normalized, +-")
    parser.add_argument("--flux-points", type=int, default=21)
    parser.add_argument(
        "--freq-start", type=float, default=-300e6, help="Hz, from drive"
    )
    parser.add_argument("--freq-stop", type=float, default=10e6, help="Hz, from drive")
    parser.add_argument("--freq-step", type=float, default=1.5e6, help="Hz")
    parser.add_argument(
        "--settle", type=float, default=200, help="ns, flux on before the drive"
    )
    parser.add_argument(
        "--tail", type=float, default=40, help="ns, flux on after the drive"
    )
    parser.add_argument(
        "--settle-readout",
        type=float,
        default=500,
        help="ns, flux off before the readout",
    )
    parser.add_argument("--nshots", type=int, default=100)
    parser.add_argument("--relaxation", type=float, help="ns")
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    sequence, sweepers, acquisition, amplitudes, freqs, center = build(platform, args)
    flux = platform.qubits[args.qubit].flux
    print(
        f"cluster {controller.address} | {args.qubit}: flux {flux} amplitude "
        f"{amplitudes[0]:+.3f}..{amplitudes[-1]:+.3f} ({len(amplitudes)} points), drive "
        f"{(center + args.freq_start) / 1e9:.4f}..{(center + args.freq_stop) / 1e9:.4f} GHz "
        f"({len(freqs)} points), {sequence.duration:.0f} ns per shot"
    )

    if args.synthetic:
        print("COMPUTED data with an invented parabola")
        amp, _ = synthetic(amplitudes, freqs, center)
        title = f"{args.qubit}: SYNTHETIC data"
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
        iq = np.asarray(res[acquisition.id], dtype=float).reshape(
            len(amplitudes), len(freqs), 2
        )
        amp = np.abs(iq[..., 0] + 1j * iq[..., 1])
        title = f"{args.qubit} pulsed flux spectroscopy"

    result = analyse(amplitudes, freqs, amp)
    a, b, _ = result["coefficients"]
    print(
        f"{result['good'].sum()} of {len(amplitudes)} lines used, rms of the fit "
        f"{result['residual_rms'] * 1e3:.2f} MHz, qubit at amplitude 0: "
        f"{result['frequency_at_zero']:.5f} GHz (configured {center / 1e9:.5f} GHz)"
    )
    print(
        f"vertex of the parabola at amplitude {result['vertex']:+.3f}, "
        f"maximum {np.polyval(np.polyfit(amplitudes[result['good']], result['centres'][result['good']] / 1e9, 2), result['vertex']):.5f} GHz"
    )
    if abs(result["vertex"]) > 0.02:
        print("WARNING: the qubit is not parked at its sweet spot.")
    print(
        f"flux_coefficients for {args.qubit} (detuning in GHz against amplitude), "
        f"to write in calibration.json: [{a:.6f}, {b:.6f}, 0.0]"
    )
    if args.synthetic:
        print(f"invented curvature {-1.3:.3f} GHz, fitted {a:.3f} GHz")

    run_dir = OUTPUT_DIR / f"qubit_flux_spectroscopy_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig = plot(amplitudes, freqs, amp, result, title)
    fig.savefig(run_dir / "qubit_flux_spectroscopy.png", dpi=160)
    np.savez(
        run_dir / "qubit_flux_spectroscopy.npz",
        amplitudes=amplitudes,
        freqs=freqs,
        amp=amp,
        centres=result["centres"],
        coefficients=result["coefficients"],
    )
    print("Saved", run_dir / "qubit_flux_spectroscopy.png")


if __name__ == "__main__":
    main()
