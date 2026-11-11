"""Compare the ways of setting a DC flux, using the qubit or its resonator as detector.

Usage (from the repository root)::

    python scripts/cqt_qpu/validation/flux_methods_comparison.py              # dry run, no hardware
    python scripts/cqt_qpu/validation/flux_methods_comparison.py --run        # on hardware
    python scripts/cqt_qpu/validation/flux_methods_comparison.py --run --target resonator --line tc1_2

There is no scope on the line, so the voltage is read through the qubit: the qubit
frequency is measured (drive frequency sweep, then readout) for the same set of flux
values, set in four different ways. If the flux is applied correctly and identically by
all of them, the four frequency-versus-flux curves coincide.

With ``--target resonator`` the resonator is measured instead, sweeping the probe
frequency without any drive, and its dip is followed. ``--line`` selects the flux line
being tuned: the qubit one (``q2``) or a coupler (``tc1_2``). The sweep is centred on the
configured frequency, or on ``--center``.

Methods (``--methods``):

``static``
    Python loop over the static ``DcConfig.offset`` (module output offset), one
    ``execute`` per flux value. Slow, but the simplest path, so it is the reference.
``offset-outer``
    A single hardware ``Parameter.offset`` sweep (sequencer ``set_awg_offs``), with the
    flux as the outer loop. Each value is held for the whole frequency sweep.
``offset-inner``
    The same sweep with the flux as the inner loop, as in the first notebook of Paul
    (the flux changes at every iteration, i.e. every ``relaxation_time``). Comparing it
    with ``offset-outer`` isolates the effect of the dwell time (settling of the line).
``pulse``
    A long rectangular flux pulse covering the drive and the readout, with a
    ``Parameter.amplitude`` sweep. The flux is back to the static value after each shot.
    A pulse rides on the static offset of the channel (``DcConfig.offset``), unlike the
    flux values of the other methods, which are absolute: the static offset is
    subtracted from the pulse amplitudes, so that all methods are compared at the same
    flux. Without this, a channel with a nonzero static offset (as the couplers of the
    CQT_QPU platform) would show the pulse curve shifted by that offset.

Each one is run with the flux values ascending and descending, to expose a hysteresis or
a slow settling, and the zero is always included in the values, to expose a constant
offset between methods.

For each method the qubit frequency is extracted row by row, and a parabola
``f = c0 + c1*v + c2*v**2`` is fitted. Against ``static``, the checks are:

- ``c0`` (frequency at zero flux): different if there is a constant shift at zero
- the vertex ``-c1/(2*c2)``: its shift is the offset along the flux axis
- the flux scale (square root of the ratio of the curvatures ``c2``): different from 1 if
  the flux scale differs between the two paths
- peak-to-peak of the curve: ~0 if the method does not move the flux at all, or only the
  first value works
- ascending versus descending RMS: hysteresis or settling

Choose ``--flux-max`` so that the qubit moves by a good fraction of the frequency span
(``--span``) at the largest flux, but does not leave it. A first run with
``--methods static`` is a cheap way to find it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Gaussian,
    Parameter,
    Pulse,
    PulseSequence,
    Rectangular,
    Sweeper,
    create_platform,
)
from qibolab._core.serialize import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
QUBIT = "q2"
METHODS = ["static", "offset-outer", "offset-inner", "pulse"]
OUTPUT = OUTPUT_DIR


def flux_channel(platform, line: str) -> str:
    """Flux channel of a qubit (``q2``) or of a coupler (``tc1_2``)."""
    if line in platform.couplers:
        return platform.couplers[line].flux
    return platform.qubits[line].flux


def base(platform, args):
    """Readout, after the flux settling time and the drive (qubit target only).

    Without any flux. The swept frequency is the one of the drive for the qubit, and the
    one of the probe for the resonator, which is measured without any drive.
    """
    qubit = platform.qubits[QUBIT]
    readout = platform.natives.single_qubit[QUBIT].MZ.create_sequence()
    sequence = PulseSequence()
    if args.target == "qubit":
        envelope = (
            Gaussian(rel_sigma=0.25) if args.shape == "gaussian" else Rectangular()
        )
        sequence.append((qubit.drive, Delay(duration=args.settle)))
        sequence.append(
            (
                qubit.drive,
                Pulse(
                    duration=args.drive_duration,
                    amplitude=args.drive_amplitude,
                    envelope=envelope,
                ),
            )
        )
        freq_channel = qubit.drive
    else:
        # the probe channel can only be used through readout events, so the settling
        # time is a delay of the acquisition channel, which the readout belongs to
        sequence.append((qubit.acquisition, Delay(duration=args.settle)))
        freq_channel = qubit.probe
    sequence |= readout
    acquisition = next(e for _, e in readout if hasattr(e, "probe")).acquisition
    center = args.center or platform.parameters.configs[freq_channel].frequency
    freqs = center + np.arange(-args.span / 2, args.span / 2 + args.step / 2, args.step)
    freq_sweeper = Sweeper(
        parameter=Parameter.frequency, values=freqs, channels=[freq_channel]
    )
    return sequence, acquisition, freqs, freq_sweeper


def plan(platform, args, method: str, values: np.ndarray):
    """Sequence and sweepers of a hardware method, and the transposition of its data."""
    sequence, acquisition, _, freq_sweeper = base(platform, args)
    flux = flux_channel(platform, args.line)
    if method == "pulse":
        pulse = Pulse(duration=sequence.duration, amplitude=0.0, envelope=Rectangular())
        sequence.append((flux, pulse))
        # a pulse is played on top of the static offset of the channel, while the
        # flux values of the other methods are absolute: the static offset is
        # subtracted, so that all of them are compared at the same flux
        static = platform.parameters.configs[flux].offset
        flux_sweeper = Sweeper(
            parameter=Parameter.amplitude, values=values - static, pulses=[pulse]
        )
    else:
        flux_sweeper = Sweeper(
            parameter=Parameter.offset, values=values, channels=[flux]
        )
    if method == "offset-inner":
        return sequence, [[freq_sweeper], [flux_sweeper]], acquisition, True
    return sequence, [[flux_sweeper], [freq_sweeper]], acquisition, False


def execute(platform, args, sequence, sweepers, acquisition, shape):
    res = platform.execute(
        [sequence],
        sweepers,
        nshots=args.nshots,
        relaxation_time=args.relaxation,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    iq = np.asarray(res[acquisition.id], dtype=float).reshape(*shape, 2)
    return iq[..., 0] + 1j * iq[..., 1]


def measure(platform, args, method: str, values: np.ndarray) -> np.ndarray:
    """Complex readout, shape (flux values, frequencies)."""
    freqs_n = len(base(platform, args)[2])
    if method == "static":
        flux = flux_channel(platform, args.line)
        original = platform.parameters.configs[flux]
        sequence, acquisition, _, freq_sweeper = base(platform, args)
        rows = []
        try:
            for value in values:
                platform.parameters.configs[flux] = replace(original, offset=value)
                rows.append(
                    execute(
                        platform,
                        args,
                        sequence,
                        [[freq_sweeper]],
                        acquisition,
                        (freqs_n,),
                    )
                )
        finally:
            platform.parameters.configs[flux] = original
        return np.array(rows)
    sequence, sweepers, acquisition, transpose = plan(platform, args, method, values)
    shape = (freqs_n, len(values)) if transpose else (len(values), freqs_n)
    z = execute(platform, args, sequence, sweepers, acquisition, shape)
    return z.T if transpose else z


def resonance(z: np.ndarray, freqs: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Frequency of the largest deviation of each row (NaN if not significant).

    The deviation is taken from the row median, the position is refined with a
    parabola through the three points around the extremum.
    """
    amp = np.abs(z)
    contrast = amp - np.median(amp, axis=1, keepdims=True)
    out = np.full(len(z), np.nan)
    for i, row in enumerate(contrast):
        k = int(np.abs(row).argmax())
        noise = 1.4826 * np.median(np.abs(row - np.median(row)))
        if noise == 0 or abs(row[k]) < threshold * noise:
            continue
        shift = 0.0
        if 0 < k < len(row) - 1:
            a, b, c = row[k - 1], row[k], row[k + 1]
            denominator = a - 2 * b + c
            if denominator != 0:
                shift = 0.5 * (a - c) / denominator
        out[i] = freqs[k] + shift * (freqs[1] - freqs[0])
    return out


def resonator_dip(
    z: np.ndarray, freqs: np.ndarray, threshold: float = 5.0
) -> np.ndarray:
    """Frequency of the minimum of the amplitude of each row (NaN if not a dip).

    The background of the transmission is not flat, so the largest deviation from the
    median, as for the qubit, would be found at the edge of the window. The amplitude is
    smoothed over three points, and a minimum at the edge of the window, or less deep
    than ``threshold`` times the noise, is not accepted. The position is refined with a
    parabola through the three points around the minimum.
    """
    amp = np.abs(z)
    out = np.full(len(z), np.nan)
    for i, row in enumerate(amp):
        smooth = np.convolve(row, np.ones(3) / 3, mode="same")
        smooth[[0, -1]] = row[[0, -1]]
        k = int(smooth[1:-1].argmin()) + 1
        noise = 1.4826 * np.median(np.abs(row - smooth))
        depth = np.median(smooth) - smooth[k]
        if noise == 0 or depth < threshold * noise:
            continue
        a, b, c = smooth[k - 1], smooth[k], smooth[k + 1]
        denominator = a - 2 * b + c
        shift = 0.5 * (a - c) / denominator if denominator != 0 else 0.0
        out[i] = freqs[k] + shift * (freqs[1] - freqs[0])
    return out


def extract(z: np.ndarray, freqs: np.ndarray, target: str) -> np.ndarray:
    """Frequency of the feature of each row: a qubit line, or a resonator dip."""
    return resonator_dip(z, freqs) if target == "resonator" else resonance(z, freqs)


def fit(values: np.ndarray, freq: np.ndarray) -> dict:
    ok = np.isfinite(freq)
    result = {"valid": int(ok.sum()), "ptp_mhz": np.nan}
    if ok.sum() < 3:
        return result
    result["ptp_mhz"] = float(np.ptp(freq[ok]) / 1e6)
    if ok.sum() < 5:
        return result
    c2, c1, c0 = np.polyfit(values[ok], freq[ok], 2)
    result |= {"c0_ghz": c0 / 1e9, "c2_mhz_per_unit2": c2 / 1e6}
    if c2 != 0:
        result["vertex"] = -c1 / (2 * c2)
    return result


def _mean_curve(directions: dict) -> np.ndarray:
    """Mean over the measured directions of the frequencies of one method."""
    return np.nanmean(list(directions.values()), axis=0)


def report(values, curves: dict) -> None:
    """Print the comparison; ``curves[method][direction]`` is the frequency array."""
    fits = {m: fit(values, _mean_curve(c)) for m, c in curves.items()}
    print(
        f"\n{'method':<14}{'valid':>6}{'p-p MHz':>10}{'c0 GHz':>12}"
        f"{'curv MHz/u^2':>14}{'vertex':>9}{'up-down MHz':>13}"
    )
    for m, f in fits.items():
        both = (
            np.isfinite(curves[m]["up"]) & np.isfinite(curves[m]["down"])
            if "down" in curves[m]
            else np.zeros(len(values), dtype=bool)
        )
        hyst = (
            float(
                np.sqrt(np.mean((curves[m]["up"][both] - curves[m]["down"][both]) ** 2))
                / 1e6
            )
            if both.any()
            else np.nan
        )
        print(
            f"{m:<14}{f['valid']:>6}{f['ptp_mhz']:>10.2f}{f.get('c0_ghz', np.nan):>12.5f}"
            f"{f.get('c2_mhz_per_unit2', np.nan):>14.1f}{f.get('vertex', np.nan):>9.4f}"
            f"{hyst:>13.2f}"
        )
    ref = fits.get("static")
    if ref is None or "c2_mhz_per_unit2" not in ref:
        print(
            "\nNo usable static reference (the qubit does not respond, or no peak found)."
        )
        return
    print("\nagainst static:")
    for m, f in fits.items():
        if m == "static" or "c2_mhz_per_unit2" not in f:
            continue
        # the curvature goes with the square of the flux scale
        ratio = f["c2_mhz_per_unit2"] / ref["c2_mhz_per_unit2"]
        scale = np.sqrt(ratio) if ratio > 0 else np.nan
        print(
            f"  {m:<14} c0 {1e3 * (f['c0_ghz'] - ref['c0_ghz']):+.3f} MHz | "
            f"flux scale x{scale:.3f} | "
            f"vertex shift {f.get('vertex', np.nan) - ref.get('vertex', np.nan):+.4f} "
            "(flux units)"
        )


def plot(values, curves, path: Path, target: str = "qubit") -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    reference = _mean_curve(curves["static"]) if "static" in curves else None
    for m, c in curves.items():
        line = ax1.plot(values, c["up"] / 1e9, "o-", ms=4, label=f"{m} up")[0]
        if reference is not None:
            ax2.plot(
                values, (c["up"] - reference) / 1e6, "o-", ms=4, color=line.get_color()
            )
        if "down" in c:
            ax1.plot(values, c["down"] / 1e9, "x--", ms=4, color=line.get_color())
            if reference is not None:
                ax2.plot(
                    values,
                    (c["down"] - reference) / 1e6,
                    "x--",
                    ms=4,
                    color=line.get_color(),
                )
    ax1.set_ylabel(f"{QUBIT} {target} frequency (GHz)")
    ax1.legend()
    ax2.set_ylabel("minus static (MHz)")
    ax2.set_xlabel("flux (normalized, 1 = 2.5 V); o: ascending, x: descending")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("Saved", path)


def plot_maps(values, freqs, data: dict, path: Path, target: str = "qubit") -> None:
    """Raw maps, one panel per method (ascending direction), with the extracted line.

    The colour is the amplitude minus the median of its row, to remove the dependence of
    the background on the frequency.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(data), figsize=(4.5 * len(data), 4.5), sharey=True, squeeze=False
    )
    for ax, (method, directions) in zip(axes[0], data.items()):
        z = directions["up"]
        amp = np.abs(z)
        contrast = amp - np.median(amp, axis=1, keepdims=True)
        mesh = ax.pcolormesh(values, freqs / 1e9, contrast.T, shading="auto")
        ax.plot(values, extract(z, freqs, target) / 1e9, "r.", ms=5)
        ax.set_title(method)
        ax.set_xlabel("flux (normalized, 1 = 2.5 V)")
        fig.colorbar(mesh, ax=ax, pad=0.01)
    axes[0, 0].set_ylabel(f"{QUBIT} {target} frequency (GHz)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("Saved", path)


def validate_offline(platform, args, values) -> None:
    """Compile every hardware method with the driver, without any instrument."""
    from qibolab._core.components import DcConfig
    from qibolab._core.execution_parameters import ExecutionParameters
    from qibolab._core.instruments.qblox.cluster import lo_configs
    from qibolab._core.instruments.qblox.sequence.sequence import compile
    from qibolab._core.instruments.qblox.validate import assert_channels_exclusion
    from qibolab._core.sweeper import normalize_sweepers

    controller = platform.instruments["cluster0"]
    configs = platform.parameters.configs
    for method in [m for m in args.methods if m != "static"]:
        sequence, sweepers, _, _ = plan(platform, args, method, values)
        sweepers = normalize_sweepers(
            sweepers,
            lo_configs(controller._los, configs),
            {k: v.offset for k, v in configs.items() if isinstance(v, DcConfig)},
        )
        options = ExecutionParameters(
            nshots=args.nshots,
            relaxation_time=args.relaxation,
            acquisition_type=AcquisitionType.INTEGRATION,
        )
        assert_channels_exclusion(sequence.align_to_delays(), controller._probes)
        programs = compile(
            sequence.align_to_delays(),
            sweepers,
            options,
            sampling_rate=1.0,
            merged_vzs=True,
        )
        print(f"  {method:<13} compiles: {sorted(programs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument(
        "--target",
        choices=["qubit", "resonator"],
        default="qubit",
        help="what is measured against the flux: the qubit line, or the resonator dip",
    )
    parser.add_argument(
        "--center",
        type=float,
        help="Hz, centre of the sweep, the configured one if absent",
    )
    parser.add_argument(
        "--line", default="q2", help="qubit or coupler flux line, e.g. q2, tc1_2"
    )
    parser.add_argument("--flux-max", type=float, default=0.05, help="normalized")
    parser.add_argument("--flux-points", type=int, default=9, help="odd, includes 0")
    parser.add_argument("--span", type=float, default=60e6, help="Hz")
    parser.add_argument("--step", type=float, default=1e6, help="Hz")
    parser.add_argument(
        "--shape", choices=["gaussian", "rectangular"], default="gaussian"
    )
    parser.add_argument("--drive-amplitude", type=float, default=0.02)
    parser.add_argument("--drive-duration", type=float, default=2000, help="ns")
    parser.add_argument(
        "--settle", type=float, default=1000, help="ns, flux before drive"
    )
    parser.add_argument("--relaxation", type=float, default=200_000, help="ns")
    parser.add_argument("--nshots", type=int, default=256)
    parser.add_argument(
        "--directions",
        choices=["both", "up"],
        default="both",
        help="ascending and descending flux, or only ascending (no hysteresis check)",
    )
    args = parser.parse_args()
    if args.flux_points % 2 == 0:
        parser.error("--flux-points must be odd, so that 0 is one of the values")

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    values = np.linspace(-args.flux_max, args.flux_max, args.flux_points)
    freqs = base(platform, args)[2]
    per_shot = base(platform, args)[0].duration + args.relaxation  # ns
    directions = ("up", "down") if args.directions == "both" else ("up",)
    runs = len(directions) * len(args.methods)
    seconds = runs * len(values) * len(freqs) * args.nshots * per_shot * 1e-9
    center = (freqs[0] + freqs[-1]) / 2
    print(
        f"line {args.line} ({flux_channel(platform, args.line)}) | {QUBIT} "
        f"{args.target} around {center / 1e9:.5f} GHz | "
        f"flux {values[0]:+.3f}..{values[-1]:+.3f} ({len(values)}) | "
        f"{len(freqs)} frequencies | methods {', '.join(args.methods)} "
        f"({' and '.join(directions)})"
    )
    print(
        f"estimated measurement time: {seconds / 60:.1f} min, plus about 5 s per static point"
    )
    print("offline compilation of the hardware methods:")
    validate_offline(platform, args, values)
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    platform.connect()
    data: dict[str, dict[str, np.ndarray]] = {}
    curves: dict[str, dict[str, np.ndarray]] = {}
    try:
        for method in args.methods:
            data[method], curves[method] = {}, {}
            for direction in directions:
                ordered = values if direction == "up" else values[::-1]
                print(f"{method} ({direction}) ...", flush=True)
                z = measure(platform, args, method, ordered)
                z = z if direction == "up" else z[::-1]  # back to the order of values
                data[method][direction] = z
                curves[method][direction] = extract(z, freqs, args.target)
    finally:
        platform.disconnect()

    stem = (
        f"cqt_flux_methods_{args.target}_{args.line}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    np.savez(
        OUTPUT / f"{stem}.npz",
        values=values,
        freqs=freqs,
        **{f"{m}_{d}": z for m, ds in data.items() for d, z in ds.items()},
    )
    (OUTPUT / f"{stem}.json").write_text(
        json.dumps(
            {m: {d: c.tolist() for d, c in ds.items()} for m, ds in curves.items()},
            indent=2,
        )
    )
    report(values, curves)
    plot(values, curves, OUTPUT / f"{stem}.png", args.target)
    plot_maps(values, freqs, data, OUTPUT / f"{stem}_maps.png", args.target)


if __name__ == "__main__":
    main()
