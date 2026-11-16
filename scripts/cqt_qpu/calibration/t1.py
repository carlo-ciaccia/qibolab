"""T1 measurement of one qubit of the CQT_QPU platform.

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/t1.py                      # dry run, compiles offline
    python scripts/cqt_qpu/calibration/t1.py --run                # on hardware
    python scripts/cqt_qpu/calibration/t1.py --run --stop 200000  # waits up to 200 us

The sequence is the one of the ``qibocal`` T1 protocol: an RX pulse, a delay, then the
readout. The delay is a hardware sweeper (``Parameter.duration`` on a ``Delay`` of the
acquisition channel), played as a Q1ASM ``wait`` whose duration is read from a register.

A dry run compiles the sweep offline and checks that the swept ``wait`` is split in
chunks. The ``wait`` instruction takes at most 65535 ns, so a longer delay, read from a
register, used to wrap around and make the decay start over at about 65.5 us. The plot
marks that time.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
    create_platform,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
QUBIT = "q2"
OUTPUT = OUTPUT_DIR


def build(platform, qubit: str, delays: np.ndarray):
    """RX, swept delay, readout. Returns sequence, sweeper and the acquisition id."""
    natives = platform.natives.single_qubit[qubit]
    drive_channel, drive_pulse = natives.RX()[0]
    readout = natives.MZ()
    ro_channel = platform.qubits[qubit].acquisition

    delay = Delay(duration=0)
    sequence = PulseSequence()
    sequence.append((drive_channel, drive_pulse))
    sequence.append((ro_channel, Delay(duration=drive_pulse.duration)))
    sequence.append((ro_channel, delay))
    sequence |= readout

    sweeper = Sweeper(parameter=Parameter.duration, values=delays, pulses=[delay])
    acquisition = next(e for _, e in readout if hasattr(e, "probe")).acquisition
    return sequence, [[sweeper]], acquisition


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


MAX_WAIT = 2**16 - 1
"""Largest duration, in ns, accepted by the ``wait`` Q1ASM instruction."""


def check_waits(programs) -> None:
    """Check that every compiled wait is legal, and report the swept ones.

    An immediate wait must not exceed ``MAX_WAIT``. A wait read from a register, as the
    swept delay, is only safe if the compiled program splits it first, which the driver
    does by copying the register in the scratch one and looping over it.
    """
    from qibolab._core.instruments.qblox.q1asm.ast_ import Jlt, Register, Wait

    for name, sequence in programs.items():
        lines = sequence.program.lines
        for i, line in enumerate(lines):
            instr = line.instruction
            if not isinstance(instr, Wait):
                continue
            if isinstance(instr.duration, int):
                assert instr.duration <= MAX_WAIT, (name, instr)
            elif isinstance(instr.duration, Register):
                guarded = any(
                    isinstance(el.instruction, Jlt) for el in lines[i - 5 : i]
                )
                status = (
                    "split in chunks"
                    if guarded
                    else "NOT split, delays above 65.5 us wrap"
                )
                print(f"  {name}: wait R{instr.duration.number} is {status}")


def exp_decay(x, offset, amplitude, tau):
    return offset + amplitude * np.exp(-x / tau)


def fit(delays, z):
    """Project the signal on its principal axis and fit an exponential decay."""
    centered = z - z.mean()
    angle = 0.5 * np.angle((centered**2).mean())
    signal = (centered * np.exp(-1j * angle)).real
    p0 = [signal[-1], signal[0] - signal[-1], delays.max() / 3]
    popt, pcov = curve_fit(exp_decay, delays, signal, p0=p0, maxfev=10000)
    return signal, popt, np.sqrt(np.diag(pcov))


def plot(delays, signal, popt, perr, args):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delays / 1e3, signal, ".", label="data")
    fine = np.linspace(delays.min(), delays.max(), 400)
    ax.plot(fine / 1e3, exp_decay(fine, *popt), "r-", label="fit")
    # a delay stored in 16 bits would start over here
    ax.axvline(MAX_WAIT / 1e3, color="gray", ls="--", lw=0.8, label="65.5 us")
    ax.set_xlabel("delay before readout (us)")
    ax.set_ylabel("projected signal (a.u.)")
    ax.set_title(
        f"CQT_QPU {args.qubit} T1 = {popt[2] / 1e3:.2f} +- {perr[2] / 1e3:.2f} us"
    )
    ax.legend()
    fig.tight_layout()
    path = OUTPUT / f"cqt_{args.qubit}_t1.png"
    fig.savefig(path, dpi=150)
    print("Saved", path)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--show", action="store_true", help="open the plot window")
    parser.add_argument(
        "--print-program", action="store_true", help="print the compiled Q1ASM"
    )
    parser.add_argument("--qubit", default=QUBIT)
    parser.add_argument("--start", type=float, default=20, help="ns")
    parser.add_argument("--stop", type=float, default=200_000, help="ns")
    parser.add_argument("--step", type=float, default=4_000, help="ns")
    parser.add_argument("--nshots", type=int, default=1024)
    parser.add_argument(
        "--relaxation", type=float, help="ns, the one of the platform by default"
    )
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]

    delays = np.arange(args.start, args.stop, args.step)
    sequence, sweepers, acquisition = build(platform, args.qubit, delays)
    print(
        f"cluster {controller.address} | {args.qubit} | {len(delays)} delays "
        f"{delays[0] / 1e3:.1f}..{delays[-1] / 1e3:.1f} us"
    )

    programs = compile_offline(platform, sequence, sweepers, args)
    check_waits(programs)
    if args.print_program:
        for name, compiled in programs.items():
            print(f"--- compiled program of {name}")
            print(compiled.program.asm())

    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

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

    iq = np.asarray(res[acquisition.id], dtype=float).reshape(len(delays), 2)
    z = iq[:, 0] + 1j * iq[:, 1]
    np.savez(OUTPUT / f"cqt_{args.qubit}_t1.npz", delays=delays, z=z)
    signal, popt, perr = fit(delays, z)
    print(f"T1 = {popt[2] / 1e3:.2f} +- {perr[2] / 1e3:.2f} us")
    fig = plot(delays, signal, popt, perr, args)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
