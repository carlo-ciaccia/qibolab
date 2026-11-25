"""Time of flight of the readout of the CQT_QPU platform.

Usage (from the repository root)::

    python scripts/cqt_qpu/calibration/time_of_flight.py                       # dry run
    python scripts/cqt_qpu/calibration/time_of_flight.py --run                # all qubits, on hardware
    python scripts/cqt_qpu/calibration/time_of_flight.py --run --qubits q2 q4

For each qubit the readout probe (the platform's MZ native gate) is played detuned by
``--detuning`` from the resonator, so that the response does not depend on the resonator,
with the acquisition ``delay`` set to 0, and a RAW trace is acquired. The arrival of the
signal is where a window as long as the probe has the largest power (a matched filter for
the rectangular probe, which works when the signal is below the noise of a single sample),
if that is 6 standard deviations above what the same filter gives on the tail of the trace.
The Scheduler demo uses a threshold on single samples, which a noise spike can pass and a
weak signal does not reach. The time of flight is
that sample plus the start of the trace (``delay`` plus, where the output of the module
has a delay compensation, the 340 ns of the driver; the QRC outputs have none).

It prints, for each qubit, the time of flight and the ``delay`` to write in the
``acquisition`` config of ``parameters.json`` (the time of flight minus the delay
compensation, on the grid of 4 ns), and the value shared by the qubits that are read
out together (the largest one). Nothing is written to the platform.

The Scheduler demo measures it with the output attenuation of the readout at 6-10 dB, the input
attenuation at 30-35 dB (the Garnet hardware config), a probe of amplitude 0.7 and a detuning
of 50 MHz, while the platform has 0 dB on both and a probe of 0.09: an input without
attenuation is saturated by the noise of the line, and only noise is seen. Use
``--input-attenuation``, ``--output-attenuation`` and ``--amplitude`` to measure the way the
demo does (only for the measurement, nothing is written to the platform).

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
from qibolab._core.instruments.qblox.limits import FILTER_DELAY_COMPENSATION
from qibolab._core.instruments.qblox.utils import time_of_flights

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
DETUNING = 50e6  # Hz, as in the Qblox Scheduler demo
NSHOTS = 1024
GRID_NS = 4
OUTPUT = OUTPUT_DIR / "cqt_time_of_flight.png"


LOOPBACK_PROBE_NS = 1000
LOOPBACK_AMPLITUDE = 0.2
SMOOTH_NS = 8
TAIL_NS = 4096
SEARCH_NS = 3000
SIGMAS = 6


def readout_sequence(platform, name: str) -> PulseSequence:
    """The MZ gate of the qubit, or a flat probe if the platform has no native gates."""
    natives = platform.natives.single_qubit.get(name)
    if natives is not None and natives.MZ is not None:
        return natives.MZ.create_sequence()
    readout = Readout(
        probe=Pulse(
            duration=LOOPBACK_PROBE_NS,
            amplitude=LOOPBACK_AMPLITUDE,
            relative_phase=0.0,
            envelope=Rectangular(),
        ),
        acquisition=Acquisition(duration=LOOPBACK_PROBE_NS),
    )
    return PulseSequence([(platform.qubits[name].acquisition, readout)])


def arrival(trace: np.ndarray, duration: int) -> tuple[int | None, np.ndarray]:
    """Sample where a probe of ``duration`` ns starts in a complex trace.

    A threshold on single samples fails when the signal is below the noise of a sample, as
    in a readout line with the input attenuated. The probe is a rectangle of known length,
    so the arrival is where the power of ``duration`` consecutive samples is the largest
    (matched filter, the position of the edge is then known to a few ns even at a low signal
    to noise). The noise is measured on the tail of the trace, long after the probe: the
    arrival is accepted if the score is ``SIGMAS`` standard deviations above what the same
    filter gives on the tail. The returned magnitude is smoothed over ``SMOOTH_NS``.
    """
    tail = trace[-TAIL_NS:]
    adjusted = trace - tail.mean()
    power = np.abs(adjusted) ** 2
    magnitude = np.convolve(
        np.abs(adjusted), np.ones(SMOOTH_NS) / SMOOTH_NS, mode="same"
    )
    csum = np.concatenate([[0.0], np.cumsum(power)])
    score = (
        csum[duration:] - csum[:-duration]
    ) / duration  # starts 0 .. len - duration
    null = score[len(score) - (TAIL_NS - duration) :]  # windows inside the tail
    window = score[:SEARCH_NS]
    best = int(np.argmax(window))
    if window[best] < null.mean() + SIGMAS * null.std():
        return None, magnitude
    return best, magnitude


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
    parser.add_argument(
        "--profile", type=int, default=2000, help="ns of the printed profile"
    )
    parser.add_argument("--nshots", type=int, default=NSHOTS)
    parser.add_argument(
        "--platform",
        default=PLATFORM,
        help="e.g. qblox_qrc_loopback (QRC_SLOT, QRC_CLUSTER_IP), with a flat probe",
    )
    parser.add_argument(
        "--input-attenuation", type=float, help="dB of the QRC input, default: platform"
    )
    parser.add_argument(
        "--output-attenuation",
        type=float,
        help="dB of the QRC output of the probe, default: platform",
    )
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(args.platform)
    controller = platform.instruments["cluster0"]
    channels = platform.channels
    qubits = args.qubits or [
        q
        for q in platform.qubits
        if q not in platform.natives.single_qubit
        or platform.natives.single_qubit[q].MZ is not None
    ]
    qubits = [q for q in qubits if platform.qubits[q].acquisition in channels]
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
            sequence = readout_sequence(platform, name)
            readout = next(e for _, e in sequence if hasattr(e, "probe"))
            if args.amplitude is not None:
                louder = readout.model_copy(
                    update={
                        "probe": readout.probe.model_copy(
                            update={"amplitude": args.amplitude}
                        )
                    }
                )
                sequence = type(sequence)(
                    [(ch, louder if ev is readout else ev) for ch, ev in sequence]
                )
                readout = louder
            probe, acquisition = qubit.probe, qubit.acquisition
            original = (configs[probe], configs[acquisition])
            configs[probe] = original[0].model_copy(
                update={"frequency": original[0].frequency - args.detuning}
            )
            acq_update = {"delay": 0.0}
            if args.input_attenuation is not None:
                acq_update["attenuation"] = args.input_attenuation
            configs[acquisition] = original[1].model_copy(update=acq_update)
            lo = platform.channels[probe].lo
            original_lo = configs[lo]
            if args.output_attenuation is not None:
                # the attenuation of the output follows the power of the LO: -power
                configs[lo] = original_lo.model_copy(
                    update={"power": -args.output_attenuation}
                )
            # the start of the trace is the one of the delay of this measurement (0)
            start = time_of_flights({acquisition: configs[acquisition]}, uncompensated)[
                acquisition
            ]
            try:
                res = platform.execute(
                    [sequence],
                    nshots=args.nshots,
                    relaxation_time=max(platform.settings.relaxation_time, 100_000),
                    acquisition_type=AcquisitionType.RAW,
                    averaging_mode=AveragingMode.CYCLIC,
                )
            finally:
                configs[probe], configs[acquisition] = original
                configs[lo] = original_lo
            iq = np.asarray(res[readout.acquisition.id], dtype=float).reshape(-1, 2)
            z = iq[:, 0] + 1j * iq[:, 1]
            compensation = (
                0.0 if acquisition in uncompensated else FILTER_DELAY_COMPENSATION
            )
            sample, magnitude = arrival(z, int(readout.probe.duration))
            peak = int(np.argmax(magnitude))
            print(
                f"{name}: {len(z)} samples from {start:.0f} ns after the probe | baseline "
                f"{np.abs(z[-TAIL_NS:] - z[-TAIL_NS:].mean()).mean():.2e} (tail) | peak "
                f"{magnitude[peak]:.2e} at sample {peak} | arrival sample {sample}"
            )
            np.save(OUTPUT_DIR / f"cqt_time_of_flight_{name}.npy", z)
            binned = magnitude[: args.profile // 50 * 50].reshape(-1, 50).mean(axis=1)
            print(
                f"{name}: mean |signal| per 50 ns from the start of the trace: "
                + " ".join(f"{b:.2f}" for b in binned)
            )
            results[name] = (start, compensation, sample, magnitude)
    finally:
        platform.disconnect()

    print(f"\n{'qubit':>6} {'tof (ns)':>9} {'delay to set (ns)':>18}")
    delays = {}
    for name, (start, compensation, sample, _) in results.items():
        if sample is None or sample < 0:
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
