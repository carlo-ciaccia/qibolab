"""Cryoscope of the flux line of a qubit, with a flux pulse played as an AWG offset.

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/cryoscope.py                       # dry run, compiles offline
    python scripts/cqt_qpu/calibration/cryoscope.py --run                 # on hardware
    python scripts/cqt_qpu/calibration/cryoscope.py --run --variant all   # compare the variants
    python scripts/cqt_qpu/calibration/cryoscope.py --synthetic           # analysis on computed data

Same experiment as the cryoscope of Qibocal and ``cal20_cryoscope.py`` of the Scheduler: a
Ramsey sequence with a flux pulse of duration ``tau`` between two pi/2 pulses, whose second
one is X90 or Y90 (the two projections). The separation of the two pi/2 pulses is the same
for every ``tau``. The phase of the qubit is the angle of the two projections, its derivative
against ``tau`` is the detuning, and the settled detuning is checked against the
``flux_coefficients`` of ``calibration.json``.

What is different from Qibocal is the flux pulse. It is a ``Rectangular`` one, which the
Qblox driver plays as an AWG offset (a ``set_awg_offs``, a ``wait`` and a ``set_awg_offs``
back to 0), with no waveform memory whatever its duration. Its duration is swept by the
hardware, as the delay of ``t1.py``: the ``wait`` reads it from a register. Qibocal uses an
envelope with leading zeros (``PaddedRectangular``), which is stored as one waveform for
every duration and filled the waveform memory (16384 samples) for 64 durations plus
padding. An offset change needs 4 ns before the next operation, so ``tau`` starts at 4 ns,
as in ``cal20_cryoscope.py``.

The three variants play the same pulses and differ in how they are sent to the cluster:

- ``shared``: one execution, the two sequences (X and Y) share the flux pulse object, as
  Qibocal does. In the first run of Qibocal on q2 the X projection oscillated and the Y one
  stayed flat.
- ``distinct``: one execution, the two sequences have their own flux pulse.
- ``separate``: one execution for each projection.

The script prints the standard deviation of both projections: both have to oscillate by
about the same amount. A projection that stays flat is a driver or sequence problem, not a
physical one.
"""

import argparse
import os
import sys
import time
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
MIN_TAU = 4
"""Shortest flux pulse, in ns: an AWG offset needs 4 ns before the next operation."""
VARIANTS = ("shared", "distinct", "separate")


def build(platform, args, taus: np.ndarray):
    """Sequences of the two projections, their flux pulses and readout events."""
    qubit = platform.qubits[args.qubit]
    natives = platform.natives.single_qubit[args.qubit]
    first = natives.RX90()
    first_length = first.duration
    second_x, second_y = natives.RX90(), natives.R(theta=np.pi / 2, phi=np.pi / 2)
    mz_x, mz_y = natives.MZ(), natives.MZ()

    # time between the end of the first pi/2 pulse and the start of the second one
    gap = args.buffer + taus.max() + args.settle

    def sequence(second, mz, flux_pulse):
        seq = PulseSequence()
        seq.extend(first)
        seq.append((qubit.flux, Delay(duration=first_length + args.buffer)))
        seq.append((qubit.flux, flux_pulse))
        seq.append((qubit.drive, Delay(duration=gap)))
        seq.extend(second)
        seq.append(
            (qubit.acquisition, Delay(duration=first_length + gap + second.duration))
        )
        seq.extend(mz)
        return seq

    def pulse():
        return Pulse(
            duration=float(taus[0]), amplitude=args.amplitude, envelope=Rectangular()
        )

    shared = pulse()
    pulse_x, pulse_y = (
        (shared, shared) if args.variant == "shared" else (pulse(), pulse())
    )
    seq_x = sequence(second_x, mz_x, pulse_x)
    seq_y = sequence(second_y, mz_y, pulse_y)

    def acquisition(mz):
        return next(e for _, e in mz if hasattr(e, "probe")).acquisition

    ids = {"X": acquisition(mz_x).id, "Y": acquisition(mz_y).id}
    assert ids["X"] != ids["Y"], "the two readouts have the same id"
    return {"X": seq_x, "Y": seq_y}, {"X": pulse_x, "Y": pulse_y}, ids


def sweeper_for(pulses, taus):
    unique = list({p.id: p for p in pulses}.values())
    return [[Sweeper(parameter=Parameter.duration, values=taus, pulses=unique)]]


def compile_offline(platform, sequence, sweepers, args):
    """Compile with the Qblox driver, without any instrument."""
    from qibolab._core.components import DcConfig
    from qibolab._core.execution_parameters import ExecutionParameters
    from qibolab._core.instruments.qblox.cluster import lo_configs
    from qibolab._core.instruments.qblox.sequence.sequence import compile
    from qibolab._core.sweeper import normalize_sweepers

    controller = platform.instruments["cluster0"]
    configs = platform.parameters.configs
    sweepers = normalize_sweepers(
        sweepers,
        lo_configs(controller._los, configs),
        {k: v.offset for k, v in configs.items() if isinstance(v, DcConfig)},
    )
    options = ExecutionParameters(
        nshots=args.nshots,
        relaxation_time=args.relaxation or platform.settings.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    return compile(
        sequence.align_to_delays(),
        sweepers,
        options,
        sampling_rate=1.0,
        merged_vzs=True,
    )


def flux_program(programs, flux_channel) -> str:
    """The instructions of the flux channel, one line each."""
    program = programs[flux_channel].program
    names = [type(line.instruction).__name__ for line in program.elements]
    return ", ".join(f"{n} x{names.count(n)}" for n in dict.fromkeys(names))


def to_complex(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=float)
    return iq[..., 0] + 1j * iq[..., 1]


def phase_from_projections(x: np.ndarray, y: np.ndarray):
    """Phase (rad, unwrapped, 0 at the first point), centre and radius of the X/Y circle.

    ``x = c + r cos(phi)`` and ``y = c + r sin(phi)`` share the centre ``c`` (the readout
    levels) and the radius ``r``, so ``x^2 + y^2 = 2c (x + y) + (r^2 - 2c^2)`` is linear in
    its unknowns (as in ``cal20_cryoscope.py``).
    """
    design = np.column_stack([x + y, np.ones_like(x)])
    (p, q), *_ = np.linalg.lstsq(design, x**2 + y**2, rcond=None)
    centre = p / 2
    radius_sq = q + 2 * centre**2
    if not np.isfinite(radius_sq) or radius_sq <= 0:
        centre = float(np.mean(np.concatenate([x, y])))
        radius_sq = float(np.mean((x - centre) ** 2 + (y - centre) ** 2))
    phase = np.unwrap(np.arctan2(y - centre, x - centre))
    return phase - phase[0], float(centre), float(np.sqrt(radius_sq))


def project(zx: np.ndarray, zy: np.ndarray):
    """Rotate the complex signals of both projections on the real axis (principal axis)."""
    z = np.concatenate([zx, zy])
    centred = z - z.mean()
    angle = 0.5 * np.angle((centred**2).mean())
    rotated = (centred * np.exp(-1j * angle)).real
    return rotated[: len(zx)], rotated[len(zx) :]


def analyse(taus, zx, zy, coefficients=None, amplitude=None):
    """Phase, detuning and step response from the two projections."""
    sx, sy = project(zx, zy)
    phase, _, _ = phase_from_projections(sx, sy)
    detuning = np.gradient(phase, taus) / (2 * np.pi)  # GHz, taus in ns
    n_tail = max(3, len(taus) // 10)
    settled = float(np.mean(detuning[-n_tail:]))
    result = {
        "sx": sx,
        "sy": sy,
        "std_x": float(np.std(sx)),
        "std_y": float(np.std(sy)),
        "phase": phase,
        "detuning": detuning,
        "settled": settled,
        "step_response": detuning / settled if abs(settled) > 1e-6 else None,
        "max_jump": float(np.max(np.abs(np.diff(phase)))),
    }
    if coefficients is not None and amplitude is not None:
        expected = float(np.polyval(coefficients, amplitude))
        result["expected"] = expected
        # the amplitude that gives the settled detuning
        roots = np.roots(
            np.array(coefficients, dtype=float)
            - np.r_[np.zeros(len(coefficients) - 1), settled]
        )
        real = roots[np.isreal(roots)].real
        result["implied_amplitude"] = (
            float(real[np.argmin(np.abs(real - amplitude))]) if len(real) else np.nan
        )
    return result


def synthetic(
    taus,
    amplitude,
    coefficients,
    rise=6.0,
    overshoot=0.04,
    decay=40.0,
    noise=0.02,
    seed=1,
):
    """COMPUTED data, not a measurement: a line with an invented rise time and overshoot.

    The flux seen by the qubit is ``1 - exp(-t/rise) + overshoot * exp(-t/decay)``, so the
    detuning follows the flux coefficients and the phase is its integral. The readout levels
    are invented too: they are here to have a known answer for the analysis.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, taus[-1] + 1.0, 0.1)
    flux = amplitude * (1 - np.exp(-t / rise) + overshoot * np.exp(-t / decay))
    detuning = np.polyval(coefficients, flux)
    phase = np.cumsum(detuning) * 0.1 * 2 * np.pi
    phi = np.interp(taus, t, phase)
    zx = (1.0 + 0.5j) * (0.5 + 0.4 * np.cos(phi)) + rng.normal(0, noise, len(phi))
    zy = (1.0 + 0.5j) * (0.5 + 0.4 * np.sin(phi)) + rng.normal(0, noise, len(phi))
    return zx, zy


def plot(taus, result, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(taus, result["sx"], "o-", ms=3, label="X")
    axes[0].plot(taus, result["sy"], "s-", ms=3, label="Y")
    axes[0].set_title("projections")
    axes[1].plot(taus, result["phase"], "k-")
    axes[1].set_title("unwrapped phase (rad)")
    axes[2].plot(taus, result["detuning"] * 1e3, label="detuning (MHz)")
    if result["step_response"] is not None:
        ax2 = axes[2].twinx()
        ax2.plot(taus, result["step_response"], "g-", label="step response")
        ax2.axhline(1, color="k", ls="--", lw=0.8)
    axes[2].set_title("detuning and step response")
    for ax in axes:
        ax.set_xlabel("flux pulse duration (ns)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def report(label, taus, result):
    flat = min(result["std_x"], result["std_y"]) < 0.3 * max(
        result["std_x"], result["std_y"]
    )
    print(
        f"[{label}] std of the projections: X {result['std_x']:.4f}, Y {result['std_y']:.4f}"
        + ("  <-- ONE PROJECTION IS FLAT" if flat else "")
    )
    print(
        f"[{label}] settled detuning {result['settled'] * 1e3:.1f} MHz"
        + (
            f", expected {result['expected'] * 1e3:.1f} MHz from the flux coefficients"
            f", implied amplitude {result['implied_amplitude']:.3f}"
            if "expected" in result
            else ""
        )
        + f", largest phase step {result['max_jump'] / np.pi:.2f} pi (aliased above 0.8)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument(
        "--synthetic", action="store_true", help="analyse computed data"
    )
    parser.add_argument("--print-program", action="store_true")
    parser.add_argument("--qubit", default="q2")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="shared")
    parser.add_argument("--amplitude", type=float, default=0.3, help="normalized")
    parser.add_argument(
        "--tau-start", type=float, default=MIN_TAU, help="ns, at least 4"
    )
    parser.add_argument("--tau-stop", type=float, default=64, help="ns")
    parser.add_argument("--step", type=float, default=1, help="ns")
    parser.add_argument(
        "--buffer", type=float, default=10, help="ns, first pi/2 to the flux"
    )
    parser.add_argument(
        "--settle", type=float, default=100, help="ns, flux to the second pi/2"
    )
    parser.add_argument("--nshots", type=int, default=1000)
    parser.add_argument("--relaxation", type=float, help="ns")
    args = parser.parse_args()
    if args.tau_start < MIN_TAU:
        parser.error(f"--tau-start has to be at least {MIN_TAU} ns")

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    taus = np.arange(args.tau_start, args.tau_stop + args.step / 2, args.step)
    flux = platform.qubits[args.qubit].flux
    coefficients = None
    calibration = Path(PLATFORMS / PLATFORM / "calibration.json")
    if calibration.exists():
        import json

        coefficients = (
            json.loads(calibration.read_text())
            .get("single_qubits", {})
            .get(args.qubit, {})
            .get("qubit", {})
            .get("flux_coefficients")
        )
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    print(
        f"cluster {controller.address} | {args.qubit} flux {flux} | {len(taus)} durations "
        f"{taus[0]:.0f}..{taus[-1]:.0f} ns | amplitude {args.amplitude} | flux coefficients "
        f"{'found' if coefficients else 'not found'}"
    )

    run_dir = OUTPUT_DIR / f"cryoscope_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        coeffs = coefficients or [-2.34, 0.0, 0.0]
        zx, zy = synthetic(taus, args.amplitude, coeffs)
        result = analyse(taus, zx, zy, coeffs, args.amplitude)
        report("synthetic", taus, result)
        fig = plot(taus, result, f"{args.qubit}: SYNTHETIC data")
        fig.savefig(run_dir / "cryoscope_synthetic.png", dpi=130)
        print("Saved", run_dir / "cryoscope_synthetic.png")
        return

    plans = {}
    for variant in variants:
        args.variant = variant
        seqs, pulses, ids = build(platform, args, taus)
        sweepers = sweeper_for(pulses.values(), taus)
        plans[variant] = (seqs, pulses, ids, sweepers)
        programs = compile_offline(platform, seqs["X"], sweepers, args)
        print(f"[{variant}] flux program: {flux_program(programs, flux)}")
        if args.print_program:
            print(programs[flux].program.asm())

    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    options = {
        "nshots": args.nshots,
        "relaxation_time": args.relaxation,
        "acquisition_type": AcquisitionType.INTEGRATION,
        "averaging_mode": AveragingMode.CYCLIC,
    }
    platform.connect()
    try:
        for variant, (seqs, pulses, ids, sweepers) in plans.items():
            print(f"\nvariant {variant} ...", flush=True)
            if variant == "separate":
                res = {}
                for proj in ("X", "Y"):
                    sw = sweeper_for([pulses[proj]], taus)
                    res.update(platform.execute([seqs[proj]], sw, **options))
            else:
                res = platform.execute([seqs["X"], seqs["Y"]], sweepers, **options)
            zx = to_complex(
                np.asarray(res[ids["X"]], dtype=float).reshape(len(taus), 2)
            )
            zy = to_complex(
                np.asarray(res[ids["Y"]], dtype=float).reshape(len(taus), 2)
            )
            np.savez(run_dir / f"cryoscope_{variant}.npz", taus=taus, zx=zx, zy=zy)
            result = analyse(taus, zx, zy, coefficients, args.amplitude)
            report(variant, taus, result)
            fig = plot(taus, result, f"{args.qubit} cryoscope, variant {variant}")
            fig.savefig(run_dir / f"cryoscope_{variant}.png", dpi=130)
            plt.close(fig)
    finally:
        platform.disconnect()
    print("Saved", run_dir)


if __name__ == "__main__":
    main()
