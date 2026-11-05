"""Qblox QRC readout test platform: one QRC in slot 2, paired IO port 1.

Intended for a loopback test (output O1 cabled to input I1, possibly through
attenuators), to check that the QRC works end to end within qibolab.

Cluster layout (from the lab photo): CMM in slot 0, QRC in slot 2, QCMs in slots
8-16. Only the QRC is used here. The cluster address is read from the
``QRC_CLUSTER_IP`` environment variable.
"""

import os

from qibolab import Hardware, Qubit
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.platform import map_ports

QRC_SLOT = 2
DEFAULT_ADDRESS = "10.10.200.126"


def create() -> Hardware:
    """Create the platform using the QRC in slot 2, paired IO port 1."""
    qubits = {"q0": Qubit.default("q0", channels=["probe", "acquisition"])}
    channels = map_ports({"qrc": (QRC_SLOT, {"io1": ["q0"]})}, qubits)

    # qibolab ports are 1-based ("io1" -> paths "2/o1", "2/i1"), while the QRC
    # hardware parameters are 0-based: "io1" is driven by LO "qrc/out0_in0".
    probe = qubits["q0"].probe
    channels[probe] = channels[probe].model_copy(update={"lo": "qrc/out0_in0"})

    controller = Cluster(
        name="cluster0",
        address=os.environ.get("QRC_CLUSTER_IP", DEFAULT_ADDRESS),
        channels=channels,
    )
    return Hardware(instruments={"cluster0": controller}, qubits=qubits)
