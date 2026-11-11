"""Find out when the QRC ``DDC_ADC_OVERRANGE`` flag is raised during a qibolab run.

Usage (from the repository root)::

    python scripts/qrc_validation/qrc_overrange_diagnosis.py          # dry run, no hardware
    python scripts/qrc_validation/qrc_overrange_diagnosis.py --run    # on hardware

The cluster resets itself at the start of every ``execute`` (``Cluster.play``), then
compiles the sequences, configures the modules (LO, attenuation) and plays. The
overrange of the inputs can be raised at any of these stages, so this script logs the
system status, sampled every 50 ms from a *separate* connection, together with the
moment of each stage, and the attenuators found on the modules at the end of the
configuration.

Two measurements are run, one after the other, each one starting with the reset done by
``execute``:

1. ``single``: the readout at the configured probe frequency only, as it would be done
   in the Qblox Scheduler
2. ``sweep``: the resonator spectroscopy, probe frequency swept around the same one

Reading the log:

- the flag shows up between ``reset`` and ``configure end``: the inputs are overranged
  while the modules are at their defaults, before the attenuation is set
- it shows up during the play of ``single`` too: the level at the ADC is too high
  already at the configured frequency (check the amplitude and the attenuations)
- it only shows up during ``sweep``: the level is the problem only away from the
  resonator, where the transmission is higher

The flags are latched, they survive until the next reset. A cluster that is in error at
the beginning is not an issue, because the first stage is the reset.

Needs the cluster to be free: the script stops if a sequencer is armed or running.
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper, create_platform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PLATFORMS

PLATFORM = "CQT_QPU"
QUBIT = "q2"
SLOTS = (2, 5)
ATTENUATIONS = ("out0_att", "in0_att", "out1_att", "in1_att")
T0 = time.monotonic()
LOG: list[tuple[float, str]] = []
LOCK = threading.Lock()


def log(message: str) -> None:
    with LOCK:
        LOG.append((time.monotonic() - T0, message))
    print(f"{time.monotonic() - T0:8.3f} s  {message}", flush=True)


class Poller(threading.Thread):
    """Log every change of the system status, from its own connection."""

    def __init__(self, address: str):
        super().__init__(daemon=True)
        self.address = address
        self.stop = threading.Event()

    def run(self) -> None:
        from qblox_instruments import Cluster

        cluster = Cluster("overrange_poller", self.address)
        last = None
        while not self.stop.is_set():
            status = str(cluster.get_system_status())
            if status != last:
                log(f"STATUS  {status}")
                last = status
            time.sleep(0.05)
        cluster.close()


def instrument_stages(controller) -> None:
    """Log the start and the end of the stages of ``Cluster.play``."""
    cls = type(controller)

    def wrap(name: str, label: str, after=None):
        original = getattr(cls, name)

        def wrapper(self, *args, **kwargs):
            log(f"{label} ...")
            result = original(self, *args, **kwargs)
            log(f"{label} done")
            if after is not None:
                after(self)
            return result

        setattr(cls, name, wrapper)

    def attenuations(self) -> None:
        for slot in SLOTS:
            module = self._modules[slot]
            values = {n: module.parameters[n]() for n in ATTENUATIONS}
            log(f"        slot {slot} attenuators: {values}")

    wrap("reset", "reset (cluster.reset)")
    wrap("_configure_modules", "configure modules", after=attenuations)
    wrap("_configure_sequencers", "configure sequencers")


def check_idle(controller) -> None:
    for slot in SLOTS:
        module = controller._modules[slot]
        for i, _ in enumerate(module.sequencers):
            state = module.get_sequencer_status(i).state.name
            if state not in ("IDLE", "STOPPED"):
                raise SystemExit(f"slot {slot} sequencer {i} is {state}: cluster busy.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--span", type=float, default=40e6, help="sweep span, Hz")
    parser.add_argument("--step", type=float, default=0.5e6, help="sweep step, Hz")
    parser.add_argument("--nshots", type=int, default=256)
    args = parser.parse_args()

    os.environ.setdefault("QIBOLAB_PLATFORMS", str(PLATFORMS))
    platform = create_platform(PLATFORM)
    controller = platform.instruments["cluster0"]
    qubit = platform.qubits[QUBIT]
    center = platform.parameters.configs[qubit.probe].frequency
    freqs = center + np.arange(-args.span / 2, args.span / 2 + args.step / 2, args.step)
    mz = platform.natives.single_qubit[QUBIT].MZ
    print(
        f"cluster {controller.address} | {QUBIT} readout {center / 1e9:.5f} GHz | "
        f"MZ amplitude {next(e for _, e in mz.create_sequence() if hasattr(e, 'probe')).probe.amplitude}"
    )
    print(f"measurements: single (1 frequency), sweep ({len(freqs)} frequencies)")
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    platform.connect()
    check_idle(controller)
    poller = Poller(controller.address)
    poller.start()
    instrument_stages(controller)
    options = {
        "nshots": args.nshots,
        "acquisition_type": AcquisitionType.INTEGRATION,
        "averaging_mode": AveragingMode.CYCLIC,
    }
    try:
        time.sleep(0.5)
        log("=== single frequency")
        platform.execute([mz.create_sequence()], [], **options)
        log("=== single frequency finished")
        time.sleep(1.0)
        log("=== sweep")
        sweeper = Sweeper(
            parameter=Parameter.frequency, values=freqs, channels=[qubit.probe]
        )
        platform.execute([mz.create_sequence()], [[sweeper]], **options)
        log("=== sweep finished")
        time.sleep(1.0)
    finally:
        poller.stop.set()
        poller.join(timeout=5)
        platform.disconnect()


if __name__ == "__main__":
    main()
