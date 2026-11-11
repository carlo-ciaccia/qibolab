"""CQT_QPU: IQM Garnet-style wiring on a Qblox cluster, QRC readout and drive.

Wiring, from the ``fieldlab`` hardware config and the "Qblox - IQM Garnet" table:

- readout: QRC in slot 2, paired IO port 1 (O1 / I1, readout line A) for ``q2``,
  LO ``qrc_readout/out0_in0``
- drive: QRC in slot 5, output O6 (``XY2``) for ``q2``, LO ``qrc_drive/out5``
- flux: QCM baseband modules in slots 8 to 18, one output (channel) per line. These
  are the flux lines of the tunable couplers (``TC a-b``, ids ``tc{a}_{b}``), and of
  the qubits ``q2``, ``q4``, ``q6`` and ``q10``

Only ``q2`` has readout and drive channels for now. The flux lines of the other
qubits (``q4``, ``q6`` and ``q10``) are defined without them, since the rest of their
lines are not part of this platform yet.

Qibolab ports are 1-based, while the QRC hardware parameters are 0-based, e.g. O6 is
driven by LO ``out5``. The cluster address is read from the ``QRC_CLUSTER_IP``
environment variable, and defaults to the one in the fieldlab config.
"""

import os

from qibolab import Hardware, Qubit
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.platform import map_ports

READOUT_SLOT = 2
DRIVE_SLOT = 5
DEFAULT_ADDRESS = "192.168.0.80"

FLUX_LINES = {
    # slot: {QCM output (1-based): element}
    8: {1: "q2", 2: "q6", 3: "q10", 4: "tc3_4"},
    10: {2: "tc6_11", 3: "q4", 4: "tc5_10"},
    12: {1: "tc1_2", 2: "tc3_8", 3: "tc7_12", 4: "tc6_7"},
    14: {1: "tc2_5", 2: "tc8_9", 3: "tc1_4", 4: "tc9_14"},
    16: {2: "tc4_5", 3: "tc10_11", 4: "tc4_9"},
    18: {2: "tc5_6", 3: "tc9_10", 4: "tc10_15"},
}

QRC = {
    2: {
        "io1": ["q1", "q2", "q5", "q6", "q7"],
        2: ["q4"],
        4: ["q10"],
        5: ["q9"],
        6: ["q7"],
    },
    5: {
        "io1": ["q3", "q4", "q9", "q10"],
        2: ["q1"],
        3: ["q5"],
        4: ["q6"],
        5: ["q3"],
        6: ["q2"],
    },
}

"""Flux lines of the QCM modules, from the Garnet wiring table.

Elements named ``tc{a}_{b}`` are tunable couplers (``TC a-b``), the others are qubits.
"""
QUBITS = [f"q{k}" for k in range(1, 11) if k != 8]


def _elements() -> tuple[dict[str, Qubit], dict[str, Qubit]]:
    names = [e for lines in FLUX_LINES.values() for e in lines.values()]
    qubits = {
        qid: Qubit.default(
            qid,
            channels=["probe", "acquisition", "drive", "flux"]
            if qid in names
            else ["probe", "acquisition", "drive"],
        )
        for qid in QUBITS
    }
    couplers = {name: Qubit.coupler(name) for name in names if name.startswith("tc")}
    return qubits, couplers


def create() -> Hardware:
    """Create the platform: QRC readout and drive, QCM flux lines."""
    qubits, couplers = _elements()
    cluster = {f"qrc{slot}": (slot, port_map) for slot, port_map in QRC.items()} | {
        f"qcm{slot}": (slot, {port: [name] for port, name in lines.items()})
        for slot, lines in FLUX_LINES.items()
    }
    channels = map_ports(cluster, qubits, couplers)

    for slot, port_map in QRC.items():
        for port, qid_array in port_map.items():
            if port == "io1":
                feedline_lo = f"qrc{slot}/out0_in0"
                for qid in qid_array:
                    channels[qubits[qid].probe] = channels[
                        qubits[qid].probe
                    ].model_copy(update={"lo": feedline_lo})
            else:
                qid = qid_array[0]
                channels[qubits[qid].drive] = channels[qubits[qid].drive].model_copy(
                    update={"lo": f"qrc{slot}/out{port - 1}"}
                )

    controller = Cluster(
        name="cluster0",
        address=os.environ.get("QRC_CLUSTER_IP", DEFAULT_ADDRESS),
        channels=channels,
    )
    return Hardware(
        instruments={"cluster0": controller}, qubits=qubits, couplers=couplers
    )
