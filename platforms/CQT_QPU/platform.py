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
"""Flux lines of the QCM modules, from the Garnet wiring table.

Elements named ``tc{a}_{b}`` are tunable couplers (``TC a-b``), the others are qubits.
"""


def _elements() -> tuple[dict[str, Qubit], dict[str, Qubit]]:
    names = [e for lines in FLUX_LINES.values() for e in lines.values()]
    qubits = {
        "q2": Qubit.default("q2", channels=["probe", "acquisition", "drive", "flux"])
    } | {
        name: Qubit.default(name, channels=["flux"])
        for name in names
        if not name.startswith("tc") and name != "q2"
    }
    couplers = {name: Qubit.coupler(name) for name in names if name.startswith("tc")}
    return qubits, couplers


def create() -> Hardware:
    """Create the platform: QRC readout and drive, QCM flux lines."""
    qubits, couplers = _elements()
    cluster = {
        "qrc_readout": (READOUT_SLOT, {"io1": ["q2"]}),
        "qrc_drive": (DRIVE_SLOT, {6: ["q2"]}),
    } | {
        f"qcm{slot}": (slot, {port: [name] for port, name in lines.items()})
        for slot, lines in FLUX_LINES.items()
    }
    channels = map_ports(cluster, qubits, couplers)

    qubit = qubits["q2"]
    channels[qubit.probe] = channels[qubit.probe].model_copy(
        update={"lo": "qrc_readout/out0_in0"}
    )
    channels[qubit.drive] = channels[qubit.drive].model_copy(
        update={"lo": "qrc_drive/out5"}
    )

    controller = Cluster(
        name="cluster0",
        address=os.environ.get("QRC_CLUSTER_IP", DEFAULT_ADDRESS),
        channels=channels,
    )
    return Hardware(
        instruments={"cluster0": controller}, qubits=qubits, couplers=couplers
    )
