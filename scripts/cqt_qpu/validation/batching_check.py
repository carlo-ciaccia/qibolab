"""Check the batching of long multiplexed sequences on the Qblox cluster.

Usage (from the repository root)::

    python scripts/cqt_qpu/validation/batching_check.py                      # preview only
    python scripts/cqt_qpu/validation/batching_check.py --depths 200 1000 2500
    python scripts/cqt_qpu/validation/batching_check.py --run --depths 1000

Why: a multiplexed benchmark of several qubits at a large depth was rejected by the driver
with an error about too many pulses. The estimated number of program lines was summed over
all the channels, but each channel runs its own program on its own sequencer, whose
instruction memory is the limit: 12288 instructions for the readout sequencers of a QRC,
16384 for its control ones. The estimate is now made for each channel.

The sequence of each depth ``D`` is, for every qubit at the same time, ``2 D`` RX gates (they
return the qubit to its ground state), then the readout. Together with it, a ground-state
and a pi-pulse reference readout are sent, so the driver has three sequences to batch, as
in a benchmark with several circuits. Without ``--run`` nothing touches the cluster. The
preview, for each depth:

- the estimated lines of the longest channel, with the previous rule (the sum over all the
  channels of the QRCs, compared with 12288) to show whether it would have raised;
- the batches that the driver makes with the new rule;
- the lines of the Q1ASM programs actually compiled for each channel, against the limit of
  its sequencer, i.e. whether the upload fits.

With ``--run`` each depth is executed. It passes if the cluster accepts the programs and
returns every readout. The projection of the final state of each qubit on the line from the
ground to the pi-pulse reference is printed for information: it is about 0 if the gates
return the qubit to the ground state, and decoherence during a long sequence moves it.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    PulseSequence,
    create_platform,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import PLATFORMS

PLATFORM = "CQT_QPU"
QRC_SLOTS = (2, 5)
"""Slots of the QRC modules of the platform (they also have inputs)."""
OLD_LIMIT = 12288
"""Limit of the previous rule, for any module with inputs."""
ALL_QUBITS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q9", "q10"]


def readout(natives, acquisition, sequence: PulseSequence, wait: float):
    """Append a readout, after ``wait`` ns, and return the id of its acquisition."""
    mz = natives.MZ()
    if wait > 0:
        sequence.append((acquisition, Delay(duration=wait)))
    sequence.extend(mz)
    return next(e for _, e in mz if hasattr(e, "probe")).acquisition.id


def circuits(platform, qubits: list[str], depth: int):
    """Ground and pi-pulse references, and the deep sequence, with the readout ids."""
    ground, excited, deep = PulseSequence(), PulseSequence(), PulseSequence()
    ids: dict[str, tuple] = {}
    for q in qubits:
        natives = platform.natives.single_qubit[q]
        acquisition = platform.qubits[q].acquisition
        rx_duration = natives.RX().duration

        g = readout(natives, acquisition, ground, 0)
        excited.extend(natives.RX())
        e = readout(natives, acquisition, excited, rx_duration)
        for _ in range(2 * depth):
            deep.extend(natives.RX())
        d = readout(natives, acquisition, deep, 2 * depth * rx_duration)
        ids[q] = (g, e, d)
    return [ground, excited, deep], ids


def channel_slots(controller) -> dict:
    from qibolab._core.instruments.qblox.config import PortAddress

    return {
        ch: PortAddress.from_path(channel.path).slot
        for ch, channel in controller.channels.items()
    }


def offline_limits(controller) -> dict:
    """Instruction memory of the QRC channels, without a connection to the cluster."""
    # the driver reads the module types from the connected cluster, so they are given here
    modules = {
        slot: SimpleNamespace(is_qrc_type=slot in QRC_SLOTS)
        for slot in controller._channels_by_module
    }
    controller.__dict__["_modules"] = modules
    try:
        return controller._instruction_limits()
    finally:
        controller.__dict__.pop("_modules", None)


def preview(platform, controller, sequences, args, depth: int) -> bool:
    """Batches and compiled lines for the sequences. Returns whether they fit."""
    from qibolab._core.execution_parameters import ExecutionParameters
    from qibolab._core.instruments.qblox import batching
    from qibolab._core.instruments.qblox.sequence.sequence import compile

    slots = channel_slots(controller)
    qrm = {ch for ch, slot in slots.items() if slot in QRC_SLOTS}
    qcm = set(slots) - qrm
    limits = offline_limits(controller)
    relaxation = args.relaxation or platform.settings.relaxation_time
    options = ExecutionParameters(
        nshots=args.nshots,
        relaxation_time=relaxation,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    deep = sequences[-1]
    pulses = {ch: len(p) for ch, p in deep.by_channel.items()}
    per_channel = batching._lines_per_channel(
        deep, {ch: 1 for ch in pulses if ch in qrm | qcm}
    )
    worst = max(per_channel.items(), key=lambda kv: kv[1])
    old = 50 + math.ceil(1.6 * sum(n for ch, n in pulses.items() if ch in qrm))
    print(f"\n--- depth {depth}: {2 * depth} RX per qubit, {len(args.qubits)} qubits")
    print(
        f"  estimated lines: longest channel {worst[0]} {50 + worst[1]}; "
        f"previous rule (sum over the QRC channels) {old} against {OLD_LIMIT}: "
        f"{'RAISED' if old > OLD_LIMIT else 'no error'}"
    )

    try:
        batches = batching.batch_sequences_by_cluster_memory_limits(
            sequences, [], options, qcm, qrm, limits
        )
    except ValueError as error:
        print(f"  new rule: RAISES\n{error}")
        return False
    print(f"  new rule: {len(batches)} batch(es) for {len(sequences)} sequences")

    fits = True
    for index, batch in enumerate(batches):
        programs = compile(
            batch.align_to_delays(), [], options, sampling_rate=1.0, merged_vzs=True
        )
        lines = {ch: len(seq.program.lines) for ch, seq in programs.items()}
        fullest = sorted(
            lines.items(), key=lambda kv: kv[1] / limits.get(kv[0], 16384), reverse=True
        )[:3]
        for ch, n in fullest:
            limit = limits.get(ch, 16384)
            over = n > limit
            fits &= not over
            print(
                f"  batch {index}: {ch} {n} lines of {limit} "
                f"({100 * n / limit:.0f}%){' EXCEEDS' if over else ''}"
            )
    print(f"  upload: {'fits' if fits else 'DOES NOT FIT'}")
    return fits


def run(platform, sequences, ids, args, depth: int) -> bool:
    start = time.monotonic()
    results = platform.execute(
        sequences,
        nshots=args.nshots,
        relaxation_time=args.relaxation,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    print(f"  executed in {time.monotonic() - start:.1f} s")

    def value(uid) -> complex:
        iq = np.asarray(results[uid], dtype=float).reshape(2)
        return iq[0] + 1j * iq[1]

    ok = True
    for q, (g, e, d) in ids.items():
        zg, ze, zd = value(g), value(e), value(d)
        projection = ((zd - zg) * np.conj(ze - zg)).real / abs(ze - zg) ** 2
        finite = np.isfinite([zg, ze, zd]).all() and abs(ze - zg) > 0
        ok &= bool(finite)
        print(
            f"  {q}: readouts returned, final state projected on ground (0) / pi (1): "
            f"{projection:+.2f}{'' if finite else '  NO SIGNAL'}"
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--qubits", nargs="+", default=ALL_QUBITS)
    parser.add_argument("--depths", nargs="+", type=int, default=[200, 1000])
    parser.add_argument("--nshots", type=int, default=500)
    parser.add_argument(
        "--relaxation", type=float, help="ns, the one of the platform by default"
    )
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    print(f"cluster {controller.address} | qubits {args.qubits} | depths {args.depths}")

    plans = {}
    for depth in args.depths:
        sequences, ids = circuits(platform, args.qubits, depth)
        plans[depth] = (
            sequences,
            ids,
            preview(platform, controller, sequences, args, depth),
        )
    if not args.run:
        print("\nPreview only. Use --run to execute on the cluster.")
        return

    platform.connect()
    summary = {}
    try:
        for depth, (sequences, ids, fits) in plans.items():
            print(f"\n=== running depth {depth} ===", flush=True)
            try:
                summary[depth] = run(platform, sequences, ids, args, depth)
            except Exception as error:  # noqa: BLE001 - report which depth fails, and why
                print(f"  FAILED: {type(error).__name__}: {error}")
                summary[depth] = False
                print(
                    "  (the preview said the upload "
                    + ("fits" if fits else "does not fit")
                    + ")"
                )
    finally:
        platform.disconnect()
    print("\nSummary:", {d: ("ok" if ok else "FAILED") for d, ok in summary.items()})


if __name__ == "__main__":
    main()
