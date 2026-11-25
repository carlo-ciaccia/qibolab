"""Dump the parameters of one module of a cluster, and compare two dumps.

Usage (from the repository root)::

    python scripts/qrc_validation/dump_module.py --address 192.168.0.80 --slot 2 --out qbs.json
    python scripts/qrc_validation/dump_module.py --address 192.168.0.80 --slot 2 --out qibolab.json
    python scripts/qrc_validation/dump_module.py --diff qbs.json qibolab.json

Why: the same measurement (the time of flight of one readout line) is clean with the Qblox
Scheduler and noisy with the Qibolab driver. Both leave the module configured the way they
last ran, so reading every parameter of the module (and of its sequencers) right after each
of them, and listing what differs, finds the setting that is not the same (attenuations,
LO, NCO, scope averaging, input gain, the sequencers' connections...).

Only reads: nothing is written to the cluster. Run the dump right after the measurement,
before anything else uses the cluster. Without ``--address`` the value of
``QRC_CLUSTER_IP`` is used.
"""

import argparse
import json
import os
import re
from pathlib import Path

NOISE = re.compile(
    r"temperature|status|state|time$|_time\b|uptime|count|cnt|sync|"
    r"program|sequence|ready|armed|idn|serial|version|timestamp|date",
    re.IGNORECASE,
)
"""Parameters that change from run to run and say nothing about the configuration."""


def flatten(node: dict, prefix: str = "") -> dict[str, object]:
    """``{path: value}`` of a snapshot of qblox_instruments (parameters and submodules)."""
    result = {}
    for name, parameter in (node.get("parameters") or {}).items():
        if isinstance(parameter, dict) and "value" in parameter:
            result[f"{prefix}{name}"] = parameter["value"]
    for section in ("submodules", "instruments"):
        for name, child in (node.get(section) or {}).items():
            result.update(flatten(child, f"{prefix}{name}."))
    return result


def dump(address: str, slot: int, out: Path) -> None:
    from qblox_instruments import Cluster

    cluster = Cluster("dump", address)
    try:
        module = getattr(cluster, f"module{slot}")
        values = flatten(module.snapshot(update=True))
    finally:
        cluster.close()
    out.write_text(json.dumps(values, indent=1, default=str), encoding="utf-8")
    print(f"{len(values)} parameters of module {slot} saved in {out}")


def diff(first: Path, second: Path, everything: bool) -> None:
    a = json.loads(first.read_text(encoding="utf-8"))
    b = json.loads(second.read_text(encoding="utf-8"))
    rows = []
    for key in sorted(a.keys() | b.keys()):
        if a.get(key) == b.get(key):
            continue
        if not everything and NOISE.search(key):
            continue
        rows.append((key, a.get(key, "<missing>"), b.get(key, "<missing>")))
    print(f"{'parameter':<60} {first.name:>22} {second.name:>22}")
    for key, x, y in rows:
        print(f"{key:<60} {str(x)[:22]:>22} {str(y)[:22]:>22}")
    print(f"\n{len(rows)} parameters differ ({len(a)} and {len(b)} in the dumps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default=os.environ.get("QRC_CLUSTER_IP"))
    parser.add_argument("--slot", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    parser.add_argument("--all", action="store_true", help="also the usual noise")
    args = parser.parse_args()
    if args.diff:
        diff(*args.diff, args.all)
    elif args.address and args.slot is not None and args.out:
        dump(args.address, args.slot, args.out)
    else:
        parser.error("give --address, --slot and --out, or --diff FIRST SECOND")


if __name__ == "__main__":
    main()
