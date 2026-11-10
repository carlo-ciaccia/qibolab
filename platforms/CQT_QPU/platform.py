"""CQT_QPU: one transmon controlled with two QRC modules (no flux line).

Wiring, from the ``fieldlab`` hardware config (flux line deliberately left out, to
test the QRC only):

- readout: QRC in slot 2, paired IO port 1 (O1 / I1), LO ``qrc_readout/out0_in0``
- drive: QRC in slot 5, output O6, LO ``qrc_drive/out5``

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


def create() -> Hardware:
    """Create the platform: readout on slot 2 io1, drive on slot 5 o6."""
    qubits = {"q2": Qubit.default("q2", channels=["probe", "acquisition", "drive"])}
    channels = map_ports(
        {
            "qrc_readout": (READOUT_SLOT, {"io1": ["q2"]}),
            "qrc_drive": (DRIVE_SLOT, {6: ["q2"]}),
        },
        qubits,
    )

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
    return Hardware(instruments={"cluster0": controller}, qubits=qubits)
