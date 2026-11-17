"""Check the state classification of the CQT_QPU readout, repeating the same executes.

Usage (from the repository root)::

    python scripts/cqt_qpu/validation/readout_classification_check.py            # dry run
    python scripts/cqt_qpu/validation/readout_classification_check.py --run      # q4
    python scripts/cqt_qpu/validation/readout_classification_check.py --run --qubits q3 q4 q9 q10 --multiplexed
    python scripts/cqt_qpu/validation/readout_classification_check.py --run --input-attenuation 30

Why: the state classification of q4 sometimes gives a wrong result, e.g. 1.0 for a pi/2
pulse, and the threshold of an SSRO sometimes does not work in randomized benchmarking.
The driver sends the platform ``iq_angle`` and ``threshold`` to the sequencer, and the
cluster is reset at every execute. This script tells apart the possible causes:

- a moving IQ cloud (the reset changes the phase of the LO): the centroids and the angle
  between them change from one execute to the other;
- a threshold, or angle, not separating the two blobs of the qubit: the classified
  fractions are wrong at every execute, also offline, with a stable cloud;
- an overrange of the input: the system status reports a flag after the execute.

Each iteration does two executes, which reset the cluster each: the ground and excited
readouts as integrated single shots (the cloud), then the same sequences classified by
the hardware. The cloud is classified offline too, with both signs of the rotation, to
show which convention the hardware follows. Nothing is written to the platform files:
``--input-attenuation`` only changes the loaded configuration.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qibolab import AcquisitionType, AveragingMode, create_platform

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import OUTPUT_DIR, PLATFORMS

PLATFORM = "CQT_QPU"
OUTPUT = OUTPUT_DIR


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


def rotate(z: np.ndarray, angle: float, sign: int) -> np.ndarray:
    return z * np.exp(-1j * sign * angle)


def classify(z: np.ndarray, angle: float, threshold: float, sign: int) -> float:
    """Fraction of shots above the threshold after rotating by ``sign * angle``."""
    return float(np.mean(rotate(z, angle, sign).real > threshold))


def stats(z: np.ndarray) -> tuple[complex, float]:
    return complex(z.mean()), float(np.std(z.real) + np.std(z.imag)) / 2


def status_flags(controller) -> str:
    """System status of the cluster, which latches flags such as the ADC overrange."""
    status = controller._cluster.get_system_status()
    return str(status)


def iteration(platform, controller, plans, args, index: int) -> list[dict]:
    rows = []
    for group, seqs, ids in plans:
        raw = platform.execute(
            seqs,
            nshots=args.nshots,
            relaxation_time=args.relaxation,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLESHOT,
        )
        flags_cloud = status_flags(controller)
        hw = platform.execute(
            seqs,
            nshots=args.nshots,
            relaxation_time=args.relaxation,
            acquisition_type=AcquisitionType.DISCRIMINATION,
            averaging_mode=AveragingMode.CYCLIC,
        )
        flags_hw = status_flags(controller)
        for q in group:
            gid, eid = ids[q]
            gz = np.asarray(raw[gid], dtype=float)
            ez = np.asarray(raw[eid], dtype=float)
            gz = gz[..., 0] + 1j * gz[..., 1]
            ez = ez[..., 0] + 1j * ez[..., 1]
            acq = platform.parameters.configs[platform.qubits[q].acquisition]
            (gc, gs), (ec, es) = stats(gz), stats(ez)
            rows.append(
                {
                    "iteration": index,
                    "qubit": q,
                    "ground": gc,
                    "excited": ec,
                    "spread": (gs + es) / 2,
                    "angle": float(np.angle(ec - gc)),
                    "separation": float(abs(ec - gc)),
                    "hw_ground": float(np.mean(hw[gid])),
                    "hw_excited": float(np.mean(hw[eid])),
                    "off": {
                        s: (
                            classify(gz, acq.iq_angle, acq.threshold, s),
                            classify(ez, acq.iq_angle, acq.threshold, s),
                        )
                        for s in (+1, -1)
                    },
                    "flags": (flags_cloud, flags_hw),
                }
            )
    return rows


def report(rows: list[dict]) -> None:
    for q in sorted({r["qubit"] for r in rows}):
        qrows = [r for r in rows if r["qubit"] == q]
        print(
            f"\n{q}: iteration | ground | excited | angle (deg) | sep/spread | hardware g/e | offline g/e (+angle, -angle) | status"
        )
        for r in qrows:
            sep = r["separation"] / r["spread"] if r["spread"] else float("nan")
            print(
                f"  {r['iteration']:3d} | {r['ground']:.4f} | {r['excited']:.4f} | "
                f"{np.degrees(r['angle']):7.1f} | {sep:5.1f} | "
                f"{r['hw_ground']:.2f}/{r['hw_excited']:.2f} | "
                f"{r['off'][1][0]:.2f}/{r['off'][1][1]:.2f}, "
                f"{r['off'][-1][0]:.2f}/{r['off'][-1][1]:.2f} | {r['flags'][1]}"
            )
        angles = np.degrees([r["angle"] for r in qrows])
        grounds = np.array([r["ground"] for r in qrows])
        print(
            f"  angle: mean {angles.mean():.1f} deg, std {angles.std():.1f} deg; "
            f"ground centroid moves by up to {np.abs(grounds - grounds.mean()).max():.4f}"
        )
        bad = [
            r["iteration"]
            for r in qrows
            if not (r["hw_ground"] < 0.5 < r["hw_excited"])
        ]
        print(f"  executes where the hardware does not separate g from e: {bad}")


def plot(rows: list[dict], args) -> Path:
    qubits = sorted({r["qubit"] for r in rows})
    fig, axes = plt.subplots(
        1, len(qubits), figsize=(5 * len(qubits), 4.5), squeeze=False
    )
    for ax, q in zip(axes[0], qubits):
        for r in (r for r in rows if r["qubit"] == q):
            ax.plot(r["ground"].real, r["ground"].imag, "b.", alpha=0.6)
            ax.plot(r["excited"].real, r["excited"].imag, "r.", alpha=0.6)
        ax.set_title(f"{q}: centroids of ground (blue) and excited (red)")
        ax.set_xlabel("I (a.u. per ns)")
        ax.set_ylabel("Q")
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    path = OUTPUT / "cqt_readout_classification_check.png"
    fig.savefig(path, dpi=150)
    print("Saved", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--qubits", nargs="+", default=["q4"])
    parser.add_argument("--multiplexed", action="store_true", help="read all together")
    parser.add_argument("--executes", type=int, default=20, help="iterations")
    parser.add_argument("--nshots", type=int, default=500)
    parser.add_argument("--relaxation", type=float, default=100e3, help="ns")
    parser.add_argument(
        "--input-attenuation",
        type=float,
        help="dB, overrides the one of the platform for the readout of these qubits",
    )
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]

    for q in args.qubits:
        channel = platform.qubits[q].acquisition
        if args.input_attenuation is not None:
            config = platform.parameters.configs[channel]
            platform.parameters.configs[channel] = config.model_copy(
                update={"attenuation": args.input_attenuation}
            )
        config = platform.parameters.configs[channel]
        print(
            f"{q}: {channel} threshold {config.threshold:.5f} "
            f"iq_angle {np.degrees(config.iq_angle):.1f} deg "
            f"input attenuation {config.attenuation} dB"
        )

    plans = sequences(platform, args.qubits, args.multiplexed)
    print(
        f"cluster {controller.address} | {len(plans)} readout group(s) | "
        f"{args.executes} iterations x 2 executes (each resets the cluster)"
    )
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    platform.connect()
    rows: list[dict] = []
    try:
        for i in range(args.executes):
            rows += iteration(platform, controller, plans, args, i)
            print(f"iteration {i} done", flush=True)
    finally:
        platform.disconnect()
    report(rows)
    plot(rows, args)


if __name__ == "__main__":
    main()
