"""Readout fidelity against the acquisition delay of the CQT_QPU platform.

Usage (from the repository root)::

    python scripts/cqt_qpu/validation/readout_delay_fidelity.py                 # dry run
    python scripts/cqt_qpu/validation/readout_delay_fidelity.py --run           # q4
    python scripts/cqt_qpu/validation/readout_delay_fidelity.py --run --qubits q3 q4 q9 q10 --multiplexed
    python scripts/cqt_qpu/validation/readout_delay_fidelity.py --run --delays 0 100 200 300 324 400

Why: the ``delay`` of an ``acquisition`` config in ``parameters.json`` is the time between
the start of the probe and the start of the integration window. On the QRC modules the
driver used to add 340 ns to it, which started the window 340 ns after the signal arrived
and cost a few percent of readout fidelity, so the best delay on those drivers is
about 340 ns shorter than the time of flight (use ``--delays`` to include the values you
want to compare, the current ``delay`` of each qubit is always added). This script measures
how much the fidelity depends on it, with the driver you have installed.

For each delay, the platform ``delay`` of the qubits is set (the files are not modified),
and ``--repeats`` times the ground and the excited state are read out as integrated single
shots (the delays are visited in turn in every repeat, so a drift of the readout, or the
phase of the IQ plane after a cluster reset, does not decide which delay looks better).
The classification does not depend on the threshold of the platform, which is valid for
one delay only: the axis is the line between the centroids of the two states, and the
threshold on that axis is fitted on half of the shots and the fidelity counted on the
other half (the assignment fidelity ``1 - (P(e|g) + P(g|e)) / 2``). The separation of the
blobs over their spread is printed as well. The best delay of each qubit is the one with the
highest mean fidelity: write it, on the grid of 4 ns, in ``parameters.json`` and repeat the
single-shot calibration (rotation and threshold) for it.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qibolab import AcquisitionType, AveragingMode, create_platform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
DELAYS = tuple(range(0, 641, 40))
"""Default delays, in ns."""


def sequences(platform, qubits: list[str], multiplexed: bool):
    """Ground and excited readout sequences, with the acquisition ids per qubit."""
    groups = [qubits] if multiplexed else [[q] for q in qubits]
    plans = []
    for group in groups:
        ground, excited = None, None
        ids = {}
        for q in group:
            natives = platform.natives.single_qubit[q]
            g = natives.MZ()
            e = natives.RX() | natives.MZ()
            ground = g if ground is None else ground | g
            excited = e if excited is None else excited | e
            ids[q] = (
                next(ev for _, ev in g if hasattr(ev, "probe")).acquisition.id,
                next(ev for _, ev in e if hasattr(ev, "probe")).acquisition.id,
            )
        plans.append((group, [ground, excited], ids))
    return plans


def assignment(ground: np.ndarray, excited: np.ndarray) -> tuple[float, float]:
    """Held-out assignment fidelity along the line between the centroids, and d'.

    The threshold is the one that classifies best the even shots, the fidelity is
    counted on the odd ones.
    """
    axis = excited.mean() - ground.mean()
    axis /= abs(axis)
    centre = (excited.mean() + ground.mean()) / 2
    g = ((ground - centre) * np.conj(axis)).real
    e = ((excited - centre) * np.conj(axis)).real
    train_g, test_g = g[::2], g[1::2]
    train_e, test_e = e[::2], e[1::2]
    candidates = np.sort(np.concatenate([train_g, train_e]))
    scores = [(np.mean(train_g < t) + np.mean(train_e >= t)) / 2 for t in candidates]
    threshold = candidates[int(np.argmax(scores))]
    fidelity = (np.mean(test_g < threshold) + np.mean(test_e >= threshold)) / 2
    spread = np.sqrt((g.var() + e.var()) / 2)
    return float(fidelity), float(abs(e.mean() - g.mean()) / spread)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--qubits", nargs="+", default=["q4"])
    parser.add_argument("--multiplexed", action="store_true", help="read all together")
    parser.add_argument("--delays", nargs="+", type=float, help="ns, default 0-640")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nshots", type=int, default=2000)
    parser.add_argument("--relaxation", type=float, default=100e3, help="ns")
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    configs = platform.parameters.configs
    qubits = args.qubits
    current = {q: configs[platform.qubits[q].acquisition].delay for q in qubits}
    delays = sorted({*(args.delays or DELAYS), *current.values()})
    plans = sequences(platform, qubits, args.multiplexed)
    print(
        f"cluster {controller.address} | qubits {qubits} | current delay (ns) {current}"
        f" | {len(delays)} delays {delays[0]:g}-{delays[-1]:g} ns x {args.repeats} repeats"
        f" x {args.nshots} shots"
    )
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    rows = []
    platform.connect()
    try:
        for repeat in range(args.repeats):
            for delay in delays:
                for q in qubits:
                    acq = platform.qubits[q].acquisition
                    configs[acq] = configs[acq].model_copy(update={"delay": delay})
                for group, seqs, ids in plans:
                    raw = platform.execute(
                        seqs,
                        nshots=args.nshots,
                        relaxation_time=args.relaxation,
                        acquisition_type=AcquisitionType.INTEGRATION,
                        averaging_mode=AveragingMode.SINGLESHOT,
                    )
                    for q in group:
                        gid, eid = ids[q]
                        gz, ez = (np.asarray(raw[i], dtype=float) for i in (gid, eid))
                        gz = gz[..., 0] + 1j * gz[..., 1]
                        ez = ez[..., 0] + 1j * ez[..., 1]
                        fidelity, dprime = assignment(gz, ez)
                        rows.append((repeat, q, delay, fidelity, dprime))
                        print(
                            f"repeat {repeat} {q} delay {delay:5.0f} ns: "
                            f"fidelity {fidelity:.4f}, separation/spread {dprime:.2f}"
                        )
    finally:
        platform.disconnect()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    table = OUTPUT_DIR / f"readout_delay_fidelity_{stamp}.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["repeat", "qubit", "delay_ns", "fidelity", "separation_over_spread"]
        )
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    print(
        "\nqubit | current delay | best delay | fidelity at current / at best (mean over repeats)"
    )
    for q in qubits:
        mean = {}
        for column, ax in ((3, axes[0]), (4, axes[1])):
            values = np.array(
                [[r[column] for r in rows if r[1] == q and r[2] == d] for d in delays]
            )
            ax.errorbar(
                delays, values.mean(axis=1), values.std(axis=1), marker="o", label=q
            )
            mean[column] = values.mean(axis=1)
        best = delays[int(np.argmax(mean[3]))]
        at_current = mean[3][delays.index(current[q])]
        print(
            f"{q:>5} | {current[q]:>13.0f} | {best:>10.0f} | "
            f"{at_current:.4f} / {mean[3].max():.4f}"
        )
        axes[0].axvline(current[q], ls=":", color="gray")
    axes[0].set_ylabel("assignment fidelity (held-out)")
    axes[1].set_ylabel("separation / spread")
    axes[1].set_xlabel("acquisition delay (ns)")
    axes[0].legend()
    fig.suptitle(
        "CQT_QPU readout against the acquisition delay (dotted: current delay)"
    )
    fig.tight_layout()
    figure = table.with_suffix(".png")
    fig.savefig(figure, dpi=150)
    print("Saved", table, "and", figure)


if __name__ == "__main__":
    main()
