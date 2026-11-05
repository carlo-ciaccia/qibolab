"""Qblox QRC readout-only platform for one qubit."""

from qibolab import Hardware, Qubit
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.platform import map_ports


def create() -> Hardware:
    """Create the platform using QRC slot 12, paired IO port 1."""
    qubits = {"q0": Qubit.default("q0", channels=["probe", "acquisition"])}
    channels = map_ports({"qrc": (12, {"io1": ["q0"]})}, qubits)

    # NOTE: qibolab ports are 1-based ("io1" -> paths "12/o1", "12/i1"), while the QRC
    # hardware parameters are 0-based, so "io1" is driven by LO "qrc/out0_in0".
    probe = qubits["q0"].probe
    channels[probe] = channels[probe].model_copy(update={"lo": "qrc/out0_in0"})

    controller = Cluster(
        name="cluster0",
        address="192.168.0.2",  # replace with your cluster IP
        channels=channels,
    )
    return Hardware(instruments={"cluster0": controller}, qubits=qubits)
