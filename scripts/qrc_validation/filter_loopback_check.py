"""Real-time predistortion filters of a QCM output, measured with a QRM input.

Usage (from the repository root)::

    python scripts/qrc_validation/filter_loopback_check.py            # dry run
    python scripts/qrc_validation/filter_loopback_check.py --run      # on hardware

Cabling: output O1 of the QCM (``QCM_SLOT``, default 2) to an input of the QRM
(``QRM_SLOT``, default 6), a baseband module, which has a DC-coupled input. The address is
read from ``QRC_CLUSTER_IP``. Keep ``--amplitude`` low (default 0.1) or use an attenuator.

The QCM plays a flat step (4 us) and the scope of the QRM records it, averaged over
shots, in the same time window of a synchronised sequence. The filters are set the way the
driver sets them: a ``DcConfig`` with the filters is turned into the parameters of the
output by ``PortConfig.delay_compensation`` and ``PortConfig.filters``, so the conversions
of the driver are the ones tested. Then:

* the step response without filters is the reference,
* a FIR with known taps is compared with the reference convolved with the taps,
* an exponential filter (several amplitudes and time constants) is compared with the
  reference: the ratio of the two responses is fitted with ``1 + b exp(-t / tau)`` and
  ``b``, ``tau`` are printed with the values given to the driver (sign and units),
* the delay (all the filters in ``delay_comp``, or enabled) must be the same.

The IHP filter and the net-zero pulse of the QCM are not used. The digital offset of the
output is 0 (since firmware 2.2.0 it is applied after the filters).
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qblox_instruments import Cluster
from scipy.optimize import curve_fit

from qibolab._core.components import DcConfig
from qibolab._core.components.filters import (
    ExponentialFilter,
    FiniteImpulseResponseFilter,
)
from qibolab._core.instruments.qblox.config.module import _current_parameter
from qibolab._core.instruments.qblox.config.port import PortConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUT_DIR

OUTPUT = OUTPUT_DIR / "filter_loopback_check.png"
STEP_NS = 4000
SCOPE_NS = 16384
START_NS = 500  # wait after the synchronisation, before the step
SHOTS = 256
FIR_TAPS = [0.6, 0.3, 0.1]
EXPONENTIALS = [(0.3, 100.0), (-0.3, 100.0), (0.2, 30.0), (0.2, 300.0)]
"""(amplitude, tau) given to the driver; tau in the units of the driver (samples = ns)."""


def qcm_program(shots: int) -> str:
    return f"""
        move {shots},R0
        nop
    loop:
        wait_sync 4
        wait {START_NS}
        play 0,0,{STEP_NS}
        wait {SCOPE_NS - START_NS - STEP_NS + 4000}
        loop R0,@loop
        stop
    """


def qrm_program(shots: int) -> str:
    return f"""
        move {shots},R0
        nop
    loop:
        wait_sync 4
        acquire 0,0,{SCOPE_NS}
        wait 4000
        loop R0,@loop
        stop
    """


def apply_filters(qcm, filters: list) -> None:
    """Set the filters of output 0 as the driver does, from a ``DcConfig``."""
    port = PortConfig(path="")
    port.delay_compensation()
    port.filters(DcConfig(offset=0.0, filters=filters))
    for name, value in port.model_dump(exclude_none=True).items():
        name, value = _current_parameter(qcm, f"out0_{name}", value)
        qcm.parameters[name].set(value)


def measure(cluster, qcm, qrm, amplitude: float, shots: int) -> np.ndarray:
    """Averaged scope trace, ``(2, SCOPE_NS)``: the two paths (inputs) of the QRM."""
    qcm.sequencer0.sequence(
        {
            "waveforms": {
                "step": {"data": [amplitude] * STEP_NS, "index": 0},
                "zero": {"data": [0.0] * 4, "index": 1},
            },
            "weights": {},
            "acquisitions": {},
            "program": qcm_program(shots),
        }
    )
    qrm.sequencer0.sequence(
        {
            "waveforms": {},
            "weights": {},
            "acquisitions": {"scope": {"num_bins": 1, "index": 0}},
            "program": qrm_program(shots),
        }
    )
    for module in (qcm, qrm):
        module.arm_sequencer(0)
    cluster.start_sequencer()
    qcm.get_sequencer_status(0, timeout=60)
    qrm.get_sequencer_status(0, timeout=60)
    qrm.get_acquisition_status(0, timeout=60)
    qrm.store_scope_acquisition(0, "scope")
    scope = qrm.get_acquisitions(0)["scope"]["acquisition"]["scope"]
    return np.array([scope["path0"]["data"], scope["path1"]["data"]])


def setup(qcm, qrm) -> None:
    qcm.sequencer0.sync_en(True)
    qcm.sequencer0.connect_out0("I")
    qcm.sequencer0.mod_en_awg(False)
    qcm.out0_offset(0.0)
    qrm.sequencer0.sync_en(True)
    qrm.sequencer0.connect_acq_I("in0")
    qrm.sequencer0.connect_acq_Q("in1")
    qrm.scope_acq_sequencer_select(0)
    qrm.scope_acq_trigger_mode_path0("sequencer")
    qrm.scope_acq_trigger_mode_path1("sequencer")
    qrm.scope_acq_avg_mode_en_path0(True)
    qrm.scope_acq_avg_mode_en_path1(True)


def edge(trace: np.ndarray) -> int:
    """Sample where the step rises (half of the plateau)."""
    plateau = np.mean(trace[START_NS + 2000 : START_NS + STEP_NS])
    return int(np.argmax(trace > plateau / 2))


def ratio_fit(
    t: np.ndarray, ratio: np.ndarray, tau0: float = 100.0
) -> tuple[float, float]:
    def model(t, b, tau):
        return 1 + b * np.exp(-t / tau)

    (b, tau), _ = curve_fit(model, t, ratio, p0=(0.1, tau0), maxfev=20000)
    return float(b), float(abs(tau))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute on hardware")
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--shots", type=int, default=SHOTS)
    args = parser.parse_args()

    address = os.environ.get("QRC_CLUSTER_IP", "10.10.200.132")
    qcm_slot = int(os.environ.get("QCM_SLOT", "2"))
    qrm_slot = int(os.environ.get("QRM_SLOT", "6"))
    print(
        f"cluster {address} | QCM {qcm_slot} O1 -> QRM {qrm_slot} input | step "
        f"{args.amplitude} x {STEP_NS} ns | FIR {FIR_TAPS} | exponentials {EXPONENTIALS}"
    )
    if not args.run:
        print("Dry run only. Use --run to execute on the cluster.")
        return

    cluster = Cluster("filters", address)
    qcm = getattr(cluster, f"module{qcm_slot}")
    qrm = getattr(cluster, f"module{qrm_slot}")
    traces: dict[str, np.ndarray] = {}
    try:
        setup(qcm, qrm)
        cases = {"reference": []}
        cases["FIR"] = [FiniteImpulseResponseFilter(coefficients=FIR_TAPS)]
        for amp, tau in EXPONENTIALS:
            cases[f"exp A={amp:+g} tau={tau:g}"] = [
                ExponentialFilter(amplitude=amp, tau=tau)
            ]
        amp, tau = EXPONENTIALS[0]
        cases["both"] = [
            ExponentialFilter(amplitude=amp, tau=tau),
            FiniteImpulseResponseFilter(coefficients=FIR_TAPS),
        ]
        for name, filters in cases.items():
            apply_filters(qcm, filters)
            traces[name] = measure(cluster, qcm, qrm, args.amplitude, args.shots)
    finally:
        apply_filters(qcm, [])  # back to delay compensation only
        qcm.stop_sequencer()
        qrm.stop_sequencer()
        cluster.close()

    reference = traces["reference"]
    path = int(np.argmax([np.ptp(p) for p in reference]))
    noise = np.std(reference[path][: START_NS // 2])
    height = np.mean(reference[path][START_NS + 2000 : START_NS + STEP_NS])
    print(
        f"\ninput used: path{path} ({'in0' if path == 0 else 'in1'}), step height "
        f"{height:.4f}, noise {noise:.5f}"
    )
    if abs(height) < 8 * noise:
        print(
            "No signal: check the cabling (QCM O1 to the QRM input) and the amplitude."
        )
        return

    ref = reference[path] / height
    t0 = edge(ref)
    n = np.arange(len(ref))
    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(n, ref, label="reference")

    fir = traces["FIR"][path] / height
    predicted = np.convolve(ref, FIR_TAPS)[: len(ref)]
    window = slice(t0 - 20, t0 + 400)
    errors = {
        lag: np.sqrt(np.mean((np.roll(predicted, lag)[window] - fir[window]) ** 2))
        for lag in range(-6, 7)
    }
    lag = min(errors, key=errors.get)
    print(
        f"\nFIR {FIR_TAPS}: rms difference with the reference convolved with the taps "
        f"{errors[lag]:.4f} (of the step height) at a lag of {lag} ns; "
        f"edge {edge(fir) - t0:+d} ns from the reference"
    )
    axes[0].plot(n, fir, label="FIR measured")
    axes[0].plot(n, np.roll(predicted, lag), "--", label="reference * taps")

    both = traces["both"][path] / height
    first = traces[f"exp A={EXPONENTIALS[0][0]:+g} tau={EXPONENTIALS[0][1]:g}"][path]
    combined = np.convolve(first / height, FIR_TAPS)[: len(ref)]
    rms = np.sqrt(np.mean((combined[window] - both[window]) ** 2))
    print(
        f"\nexponential {EXPONENTIALS[0]} and FIR together: rms difference with the "
        f"exponential alone convolved with the taps {rms:.4f}"
    )

    print(
        "\nexponentials: ratio of the response to the reference, "
        "fitted with 1 + b exp(-t/tau)"
    )
    print(f"{'A given':>9} {'tau given':>10} {'b':>9} {'tau':>8} {'edge shift':>11}")
    for amp, tau in EXPONENTIALS:
        y = traces[f"exp A={amp:+g} tau={tau:g}"][path] / height
        rising = slice(t0 + 4, t0 + int(6 * max(tau, 30)))
        b, tau_fit = ratio_fit(n[rising] - t0, y[rising] / ref[rising], tau)
        print(f"{amp:>+9g} {tau:>10g} {b:>+9.3f} {tau_fit:>8.1f} {edge(y) - t0:>+10d}")
        axes[1].plot(n[window], y[window], label=f"A={amp:+g}, tau={tau:g}")
    axes[1].plot(n[window], ref[window], "k", label="reference")
    for ax in axes:
        ax.set_xlabel("time in the scope (ns)")
        ax.set_ylabel("response / step height")
        ax.legend()
    axes[0].set_xlim(t0 - 50, t0 + 300)
    axes[1].set_xlim(t0 - 50, t0 + 1000)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print("Saved", OUTPUT)


if __name__ == "__main__":
    main()
