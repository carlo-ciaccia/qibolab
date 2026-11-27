"""XY-Z delay, XY-readout delay and cryoscope (cal20) of q2 on the fieldlab cluster, with Qblox Scheduler.

This is the flow of ``flux_line_calibration_notebook.py`` of the repository
``flux-tunable-transmons-with-flux-tunable-couplers`` (Parts A to D), cut in phases that each fit in one
10-15 min booking of the cluster. The state that has to survive from one phase to the next (the latency
corrections and the exponential filter stages of the loaded hardware config) is kept in a JSON file.

Run it from ``docs/applications/superconducting`` of that repository, with its ``.venv``::

    python <this file> --phase 1     # bare line: XY-Z, XY-readout delay, each scanned again after the correction
    python <this file> --phase 2     # cryoscope of the bare line, fit and apply the filters, XY-Z with the filters on
    python <this file> --phase 3     # cryoscope with the filters on, XY-Z check
    python <this file> --phase 4     # second amplitude: cryoscope without and with the filters
    python <this file> --phase 5     # XY-readout delay again, at the largest XY amplitude and two tones

Nothing is written to the hardware config files. Figures and the measured responses go to ``--out``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import custom_elements  # noqa: E402,F401  (registers FluxTunableTransmonElement for the YAML)
from cal33_xy_z_delay import XYZDelay  # noqa: E402
from cal36_xy_readout_delay import XYReadoutDelay  # noqa: E402
from qblox_scheduler import HardwareAgent  # noqa: E402
from single_qubit_experiment_helpers.experiment import Experiment  # noqa: E402
from single_qubit_experiment_helpers.flux_distortion import (  # noqa: E402
    clear_flux_distortion,
    set_flux_distortion,
)
from single_qubit_experiment_helpers.flux_line_tuning import FluxLineTuning  # noqa: E402

HW_CONFIG = "./dependencies/configs/hw_config_garnet.json"
DUT_CONFIG = "./dependencies/configs/dut_config_garnet.yaml"
QUBIT = "q2"

XYZ = dict(flux_amp=0.4, repetitions=500, delay_start=-300e-9, delay_stop=300e-9, delay_step=2e-9)
XYRO = dict(xy_amp=None, ro_amp=0.7, ro_duration=400e-9, xy_duration=400e-9, repetitions=100_000, max_duty=0.02)
CRYO = dict(tau_stop=200e-9, tau_step=1e-9, t_sep=300e-9, repetitions=2000, t_fit_start=25e-9)
MAX_EXP = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--amp", type=float, default=0.41, help="flux step of the cryoscope (V on top of the sweet spot)")
    parser.add_argument("--amp2", type=float, default=0.205, help="second flux step (phase 4)")
    parser.add_argument("--state", default=str(Path.home() / "qbs_q2_state.json"))
    parser.add_argument("--out", default=str(Path.home() / "qbs_q2_out"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    hw_agent = HardwareAgent(
        hardware_configuration=HW_CONFIG,
        quantum_device_configuration=DUT_CONFIG,
        output_dir=f"./data/{Path(DUT_CONFIG).stem}",
        create_dummy_connections=False,
    )
    Experiment.hw_agent = hw_agent
    Experiment.quantum_device = hw_agent.quantum_device
    qubit = hw_agent.quantum_device.get_element(QUBIT)
    tuning = FluxLineTuning(hw_agent, [qubit], hw_config=HW_CONFIG)

    def snap(tag):
        for number in plt.get_fignums():
            plt.figure(number).savefig(out / f"phase{args.phase}_{tag}_{number}.png", dpi=110)
        plt.close("all")

    def save_state():
        options = hw_agent.hardware_configuration.hardware_options
        state = {"latency": dict(options.latency_corrections or {}), "stages": tuning.stages_on(QUBIT)}
        Path(args.state).write_text(json.dumps(state, indent=2))
        print("state saved:", state)

    def restore_state(with_stages=True):
        if not Path(args.state).exists():
            return
        state = json.loads(Path(args.state).read_text())
        options = hw_agent.hardware_configuration.hardware_options
        options.latency_corrections = {**(options.latency_corrections or {}), **state["latency"]}
        if with_stages and state["stages"]:
            set_flux_distortion(hw_agent, QUBIT, exp=[tuple(stage) for stage in state["stages"]])
        print("state restored:", state, "(stages applied)" if with_stages else "(stages not applied)")

    def xyz_scan(label, post_run, compare_expected=False):
        xyz = XYZDelay([qubit])
        xyz.execute(**XYZ)
        xyz.analyze()
        xyz.plot_analysis()
        tuning.record_skews(label, xyz, compare_expected=compare_expected)
        if post_run:
            xyz.post_run()
        snap("xyz_" + label.replace(" ", "_").replace(":", ""))

    def cryoscope(label, amp):
        run = tuning.measure_cryoscope(amp=amp, **CRYO)
        result = run["experiments"][QUBIT].analyses.get(QUBIT)
        if result is not None:
            np.savez(
                out / f"cryoscope_{label}.npz",
                time=result["time"],
                voltage_response=result["voltage_response"],
                stages=np.array(run["stages"][QUBIT]),
                amp=amp,
            )
        snap("cryoscope_" + label)

    for cluster_name, cluster in hw_agent.get_clusters().items():
        print(f"Resetting {cluster_name}...")
        cluster.reset()
        print(cluster.get_system_status())

    try:
        if args.phase == 1:
            tuning.clear_filters()
            xyz_scan("A first scan", post_run=True)
            xyz_scan("A after the correction", post_run=False)
            xyro_qubits = [qubit]
            xyro = XYReadoutDelay(xyro_qubits)
            xyro.execute(frequency_detuning=50e6, **XYRO)
            xyro.analyze()
            xyro.plot_analysis()
            tuning.record_delays("D first run", xyro)
            xyro.post_run()
            snap("xyro_first")
            xyro = XYReadoutDelay(xyro_qubits)
            xyro.execute(frequency_detuning=50e6, **XYRO)
            xyro.analyze()
            xyro.plot_analysis()
            tuning.record_delays("D after the correction", xyro)
            snap("xyro_check")
            save_state()
        elif args.phase == 2:
            restore_state(with_stages=False)
            tuning.clear_filters()
            cryoscope(f"bare_{args.amp}", args.amp)
            tuning.fit_stages(max_exp=MAX_EXP, t_fit_start=CRYO["t_fit_start"])
            snap("fit")
            tuning.apply_stages()
            xyz_scan("B filters on", post_run=True, compare_expected=True)
            save_state()
        elif args.phase == 3:
            restore_state()
            cryoscope(f"filters_{args.amp}", args.amp)
            tuning.plot_responses(t_fit_start=CRYO["t_fit_start"])
            snap("responses")
            xyz_scan("C final check", post_run=False)
        elif args.phase == 4:
            restore_state(with_stages=False)
            clear_flux_distortion(hw_agent)
            cryoscope(f"bare_{args.amp2}", args.amp2)
            restore_state()
            cryoscope(f"filters_{args.amp2}", args.amp2)
        elif args.phase == 5:
            restore_state(with_stages=False)
            for detuning in (50e6, 100e6):
                xyro = XYReadoutDelay([qubit])
                xyro.execute(frequency_detuning=detuning, **{**XYRO, "xy_amp": 1.0, "repetitions": 400_000})
                xyro.analyze()
                xyro.plot_analysis()
                tuning.record_delays(f"D amplitude 1.0, tone {detuning / 1e6:.0f} MHz", xyro)
                snap(f"xyro_{detuning / 1e6:.0f}")
        tuning.print_summary()
    finally:
        tuning.release_flux_lines()


if __name__ == "__main__":
    main()
