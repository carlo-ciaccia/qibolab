"""Track the phase and the position of the IQ blobs of the CQT_QPU readout over time.

Usage (from the repository root)::

    python scripts/cqt_qpu/validation/iq_phase_drift.py                          # dry run
    python scripts/cqt_qpu/validation/iq_phase_drift.py --run                    # q4, 1 hour
    python scripts/cqt_qpu/validation/iq_phase_drift.py --run --qubits q3 q4 q9 q10 --multiplexed
    python scripts/cqt_qpu/validation/iq_phase_drift.py --run --mode no-reset --duration 240

Why: the blobs of a qubit (ground and excited state) change position between
measurements, and it is not known whether it is a slow drift or a jump, for example caused
by the reset of the cluster that every ``execute`` does. Every ``--interval`` seconds the
ground and the excited states are read out as integrated single shots, for
``--duration`` minutes, and for each qubit the script records, with the time:

- the centroid of each blob, its phase and its amplitude (the phase is relative to the
  first sample in the figure),
- the angle of the line from the ground to the excited blob, which does not depend on the
  phase offset of the ground blob alone,
- the distance between the blobs over their spread.

The data are appended to a ``.csv`` file as they are taken (so a stop with Ctrl-C, or a
crash, keeps them), and a figure is saved at the end and every ``--plot-every`` samples,
with the phase against time and the blobs in the IQ plane coloured by time. The summary
counts the jumps, i.e. the samples whose phase differs from the previous one by more than
``--jump`` degrees, and prints the slope of what remains (the drift, in degrees per hour).

``--mode`` is what the driver does between the samples, as in
``readout_classification_check.py`` (``reset``: as normal, the cluster is reset by every
execute; ``no-reset``: the cluster is reset once, at the start). Run it in both to tell a jump
at each reset from a drift in time. Nothing is written to the platform files.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUTPUT_DIR, PLATFORMS
from readout_classification_check import MODES, driver_mode, sequences

PLATFORM = "CQT_QPU"
COLUMNS = [
    "time_s",
    "sample",
    "qubit",
    "ground_re",
    "ground_im",
    "excited_re",
    "excited_im",
    "spread",
]


def centroids(raw, ids) -> tuple[complex, complex, float]:
    gid, eid = ids
    values = []
    for i in (gid, eid):
        z = np.asarray(raw[i], dtype=float)
        values.append(z[..., 0] + 1j * z[..., 1])
    ground, excited = values
    spread = (np.std(ground.real) + np.std(ground.imag)) / 2
    spread += (np.std(excited.real) + np.std(excited.imag)) / 2
    return complex(ground.mean()), complex(excited.mean()), float(spread / 2)


def load(path: Path) -> dict[str, np.ndarray]:
    """Samples of each qubit from the csv file: ``{qubit: array of rows}``."""
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for q in dict.fromkeys(r["qubit"] for r in rows):
        sel = [r for r in rows if r["qubit"] == q]
        result[q] = np.array(
            [[float(r[c]) for c in COLUMNS if c not in ("qubit",)] for r in sel]
        )
    return result


def wrap(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180) % 360 - 180


def analyse(data: np.ndarray, jump_deg: float) -> dict:
    """Phases (deg, relative to the first sample), jumps and drift of one qubit."""
    t = data[:, 0]
    ground = data[:, 2] + 1j * data[:, 3]
    excited = data[:, 4] + 1j * data[:, 5]
    phase = wrap(np.degrees(np.angle(ground)) - np.degrees(np.angle(ground[0])))
    axis = wrap(
        np.degrees(np.angle(excited - ground))
        - np.degrees(np.angle(excited[0] - ground[0]))
    )
    steps = wrap(np.diff(phase))
    jumps = np.flatnonzero(np.abs(steps) > jump_deg) + 1
    # the phase without the jumps: what is left is a drift
    smooth = np.cumsum(np.where(np.abs(steps) > jump_deg, 0.0, steps))
    smooth = np.concatenate([[0.0], smooth])
    slope = np.polyfit(t / 3600, smooth, 1)[0] if len(t) > 2 else float("nan")
    return {
        "t": t,
        "ground": ground,
        "excited": excited,
        "phase": phase,
        "axis": axis,
        "amplitude": np.abs(ground),
        "separation": np.abs(excited - ground) / data[:, 6],
        "jumps": jumps,
        "drift": float(slope),
    }


def plot(table: Path, jump_deg: float, mode: str) -> Path:
    data = load(table)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for q, rows in data.items():
        a = analyse(rows, jump_deg)
        minutes = a["t"] / 60
        axes[0, 0].plot(minutes, a["phase"], ".-", label=q)
        axes[0, 1].plot(minutes, a["axis"], ".-", label=q)
        axes[1, 0].plot(minutes, a["separation"], ".-", label=q)
        scatter = axes[1, 1].scatter(
            a["ground"].real, a["ground"].imag, c=minutes, s=14, cmap="viridis"
        )
        axes[1, 1].scatter(
            a["excited"].real,
            a["excited"].imag,
            c=minutes,
            s=14,
            cmap="viridis",
            marker="s",
        )
    axes[0, 0].set_ylabel("phase of the ground blob vs the first (deg)")
    axes[0, 1].set_ylabel("angle of ground -> excited vs the first (deg)")
    axes[1, 0].set_ylabel("distance between blobs / spread")
    for ax in axes.flat[:3]:
        ax.set_xlabel("time (min)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[1, 1].set_xlabel("I")
    axes[1, 1].set_ylabel("Q")
    axes[1, 1].set_title("centroids (circles ground, squares excited), colour = time")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    fig.colorbar(scatter, ax=axes[1, 1], label="time (min)")
    fig.suptitle(f"IQ blobs of the CQT_QPU readout over time (mode {mode})")
    fig.tight_layout()
    path = table.with_suffix(".png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary(table: Path, jump_deg: float) -> None:
    for q, rows in load(table).items():
        a = analyse(rows, jump_deg)
        print(
            f"{q}: {len(rows)} samples over {a['t'][-1] / 60:.1f} min | phase range "
            f"{np.ptp(a['phase']):.1f} deg, std {np.std(a['phase']):.1f} deg | "
            f"{len(a['jumps'])} jumps above {jump_deg:g} deg "
            f"(samples {a['jumps'].tolist()}) | drift without the jumps "
            f"{a['drift']:+.2f} deg/h | amplitude {a['amplitude'].mean():.4f} "
            f"+- {a['amplitude'].std():.4f} | blobs apart by "
            f"{a['separation'].mean():.1f} +- {a['separation'].std():.1f} spreads"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--qubits", nargs="+", default=["q4"])
    parser.add_argument("--multiplexed", action="store_true", help="read all together")
    parser.add_argument("--mode", choices=MODES, default="reset")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds")
    parser.add_argument("--duration", type=float, default=60.0, help="minutes")
    parser.add_argument("--nshots", type=int, default=500)
    parser.add_argument("--relaxation", type=float, default=100e3, help="ns")
    parser.add_argument("--jump", type=float, default=20.0, help="degrees")
    parser.add_argument("--plot-every", type=int, default=20, help="samples")
    parser.add_argument("--summarize", type=Path, help="analyse a csv file and exit")
    args = parser.parse_args()

    if args.summarize:
        summary(args.summarize, args.jump)
        print("Saved", plot(args.summarize, args.jump, "from file"))
        return

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    samples = int(args.duration * 60 / args.interval) + 1
    plans = sequences(platform, args.qubits, args.multiplexed)
    print(
        f"cluster {controller.address} | qubits {args.qubits} | mode {args.mode} | "
        f"{samples} samples every {args.interval:g} s ({args.duration:g} min), "
        f"{args.nshots} shots"
    )
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    table = OUTPUT_DIR / f"iq_phase_drift_{args.mode}_{stamp}.csv"
    start = time.monotonic()
    platform.connect()
    try:
        with driver_mode(args.mode), table.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for index in range(samples):
                target = start + index * args.interval
                time.sleep(max(0.0, target - time.monotonic()))
                for group, seqs, ids in plans:
                    raw = platform.execute(
                        seqs,
                        nshots=args.nshots,
                        relaxation_time=args.relaxation,
                        acquisition_type=AcquisitionType.INTEGRATION,
                        averaging_mode=AveragingMode.SINGLESHOT,
                    )
                    now = time.monotonic() - start
                    for q in group:
                        ground, excited, spread = centroids(raw, ids[q])
                        writer.writerow(
                            [
                                now,
                                index,
                                q,
                                ground.real,
                                ground.imag,
                                excited.real,
                                excited.imag,
                                spread,
                            ]
                        )
                        print(
                            f"{now / 60:7.2f} min {q}: ground {abs(ground):.4f} at "
                            f"{np.degrees(np.angle(ground)):7.1f} deg, excited at "
                            f"{np.degrees(np.angle(excited)):7.1f} deg, apart by "
                            f"{abs(excited - ground) / spread:.1f} spreads"
                        )
                handle.flush()
                if (index + 1) % args.plot_every == 0:
                    plot(table, args.jump, args.mode)
    except KeyboardInterrupt:
        print("\nStopped, keeping the samples taken")
    finally:
        platform.disconnect()

    summary(table, args.jump)
    print("Saved", table, "and", plot(table, args.jump, args.mode))


if __name__ == "__main__":
    main()
