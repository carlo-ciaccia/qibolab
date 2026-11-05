"""Qblox QRC drive test platform: one QRC in slot 2.

For testing only, not for normal operation. Cabling:

- output O3 (drive) is connected to input I2
- port ``io2`` (O2 / I2) is the readout port: nothing is connected to O2, but the
  probe is needed to define the acquisition (it is played at zero amplitude)

All used ports are configured with 20 dB of attenuation, through the LO power
(``-20``): since the QRC input and output attenuators follow the power of the LO of
their port, input I2 gets 20 dB as well. The LO of O3 must match the one of I2 (both
5 GHz) for the drive and the demodulation to share the same intermediate frequency.

The cluster address is read from the ``QRC_CLUSTER_IP`` environment variable.
"""

import os

from qibolab import Hardware, Qubit
from qibolab._core.instruments.qblox.cluster import Cluster
from qibolab._core.instruments.qblox.platform import map_ports

QRC_SLOT = 2
DEFAULT_ADDRESS = "10.10.200.126"


def create() -> Hardware:
    """Create the platform: readout on paired port 2, drive on output 3."""
    qubits = {"q0": Qubit.default("q0", channels=["probe", "acquisition", "drive"])}
    channels = map_ports({"qrc": (QRC_SLOT, {"io2": ["q0"], 3: ["q0"]})}, qubits)

    # qibolab ports are 1-based, while the QRC hardware parameters are 0-based:
    # "io2" is driven by LO "qrc/out1_in1", while "o3" has its own LO "qrc/out2"
    probe, drive = qubits["q0"].probe, qubits["q0"].drive
    channels[probe] = channels[probe].model_copy(update={"lo": "qrc/out1_in1"})
    channels[drive] = channels[drive].model_copy(update={"lo": "qrc/out2"})

    controller = Cluster(
        name="cluster0",
        address=os.environ.get("QRC_CLUSTER_IP", DEFAULT_ADDRESS),
        channels=channels,
    )
    return Hardware(instruments={"cluster0": controller}, qubits=qubits)
