"""Qblox QRM-RF readout-only platform for one qubit."""

from qibolab import Hardware, Qubit
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.platform import map_ports


def create() -> Hardware:
    """Create the platform using QRM-RF slot 7, IO port 1."""
    qubits = {"q0": Qubit.default("q0", channels=["probe", "acquisition"])}
    channels = map_ports({"qrm_rf": (7, {"io1": ["q0"]})}, qubits)

    probe = qubits["q0"].probe
    channels[probe] = channels[probe].model_copy(update={"lo": "qrm_rf/o1/lo"})

    controller = Cluster(
        name="cluster0",
        address="192.168.0.2",
        channels=channels,
    )
    return Hardware(instruments={"cluster0": controller}, qubits=qubits)
