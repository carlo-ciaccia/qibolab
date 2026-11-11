# Hardware scripts

Scripts that run on a Qblox cluster through qibolab. The platform definitions they use
are in `platforms/`. Run them from the repository root, e.g.

```bash
python scripts/cqt_qpu/calibration/resonator_spectroscopy.py            # dry run
python scripts/cqt_qpu/calibration/resonator_spectroscopy.py --run      # on hardware
```

Without `--run` a script only loads the platform, prints what it would do and, where it
can, compiles the experiment offline. Figures and data are written to `scripts/output/`
(not tracked). The cluster address of the CQT_QPU platform can be set with the
`QRC_CLUSTER_IP` environment variable.

| Folder | What it is |
|---|---|
| `qrc_validation/` | Checks of the driver and of the QRC itself (loopback tests, scope trace, overrange flag diagnosis). They do not measure a device. The loopback ones use the `qblox_qrc_*` platforms and a cabled output to input. |
| `cqt_qpu/calibration/` | Experiments that measure the CQT_QPU device: resonator spectroscopy, resonator or qubit spectroscopy against the flux of the couplers. |
| `cqt_qpu/validation/` | Tests of the driver on the CQT_QPU device, using the qubit or its resonator as a detector, e.g. the comparison of the four ways of setting a DC flux. |
| `common.py` | Paths shared by the scripts. |
