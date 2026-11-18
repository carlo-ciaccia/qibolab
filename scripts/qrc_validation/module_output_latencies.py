"""Output path latency of every module of the cluster, as reported by the cluster itself.

Self-contained, it only needs ``qblox_instruments`` (tested with 1.4.1).

    python scripts/qrc_validation/module_output_latencies.py --ip 192.168.0.80
    python scripts/qrc_validation/module_output_latencies.py --ip dummy --dummy   # syntax check

Why: a QCM flux channel has to start about 260 ns before the QRC channels to reach the
qubit at the same time, i.e. the QCM output path is slower than the QRC one. Qblox gives an
output path of 110 ns for QCM, QCM-RF and QRM-RF, and 170 ns for the QRC. The Qibolab driver
also enables the delay compensation of the pre-distortion filters on every non-QRC output
(the filters are off, but the output is delayed as if they were on), which is assumed to
cost about 340 ns. This script reads ``module.out{n}_latency()`` (reported in seconds, printed
here in ns) for each output in four states, to check these numbers. On the cluster at 192.168.0.80
the parameter was 0 for the bypassed filters and 340 ns for delay_comp and enabled, on the QCM
and the QRC alike, so it reports the latency of the filters only, not the output path:

    reset      right after ``cluster.reset()``
    bypassed   all the filters of the output set to "bypassed"
    delay_comp all the filters set to "delay_comp", as the driver does
    enabled    all the filters set to "enabled"

A state the module does not accept (the QRC filters only take "bypassed" and "delay_comp")
is printed as n/a. The output path itself (about 110 ns for a QCM, QCM-RF or QRM-RF, 170 ns
for a QRC, from the Qblox documentation) is not part of this parameter, so it adds to the
numbers printed here. The cluster is reset first and at the end, so use it when nobody else
is on it. No sequence is played, and no output is enabled.
"""

import argparse

from qblox_instruments import Cluster, ClusterType

STATES = ("bypassed", "delay_comp", "enabled")
FILTERS = ("fir_config", "exp0_config", "exp1_config", "exp2_config", "exp3_config")


def outputs(module) -> list[int]:
    return [n for n in range(6) if f"out{n}_latency" in module.parameters]


def kind(module) -> str:
    for flag, name in (
        ("is_qrc_type", "QRC"),
        ("is_qcm_type", "QCM"),
        ("is_qrm_type", "QRM"),
    ):
        if getattr(module, flag, False):
            return name + ("-RF" if module.is_rf_type and name != "QRC" else "")
    return "?"


def set_filters(module, n: int, state: str) -> None:
    for name in FILTERS:
        parameter = f"out{n}_{name}"
        if parameter in module.parameters:
            module.parameters[parameter].set(state)


def read(module, n: int) -> float | str:
    try:
        # the cluster reports seconds
        return round(module.parameters[f"out{n}_latency"]() * 1e9, 1)
    except Exception as error:  # noqa: BLE001 - report what the module refuses
        return f"error: {error}"[:30]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ip", required=True)
    parser.add_argument("--dummy", action="store_true", help="dummy cluster, no data")
    args = parser.parse_args()

    dummy = (
        {1: ClusterType.CLUSTER_QCM, 2: ClusterType.CLUSTER_QRC} if args.dummy else None
    )
    cluster = Cluster("latencies", identifier=args.ip, dummy_cfg=dummy)
    cluster.reset()

    print(f"{'slot':>4} {'module':<7} {'out':>3} {'reset':>10}", end="")
    print("".join(f"{state:>12}" for state in STATES), "  (ns)")
    rows = []
    for module in cluster.modules:
        if not outputs(module):
            continue
        for n in outputs(module):
            row = {"slot": module.slot_idx, "kind": kind(module), "out": n}
            row["reset"] = read(module, n)
            for state in STATES:
                try:
                    set_filters(module, n, state)
                except ValueError:
                    # e.g. the QRC has no filters to enable, only the delay compensation
                    row[state] = "n/a"
                    continue
                row[state] = read(module, n)
            rows.append(row)
            print(
                f"{row['slot']:>4} {row['kind']:<7} {n + 1:>3} {row['reset']!s:>10}"
                + "".join(f"{row[state]!s:>12}" for state in STATES)
            )
    cluster.reset()
    cluster.close()

    print("\nPer module type (bypassed -> delay_comp), distinct values in ns:")
    for name in sorted({r["kind"] for r in rows}):
        values = {
            (r["bypassed"], r["delay_comp"], r["enabled"])
            for r in rows
            if r["kind"] == name
        }
        print(f"  {name}: {sorted(values, key=str)}")


if __name__ == "__main__":
    main()
