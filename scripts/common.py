"""Paths shared by the hardware scripts.

The scripts are run from the repository root, e.g.
``python scripts/cqt_qpu/calibration/resonator_spectroscopy.py --run``, and find the
platforms in the ``platforms`` folder next to this one. Everything they produce
(figures, data) is written to ``scripts/output``, which is not tracked.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ROOT / "platforms"
"""Folder with the platform definitions, to be set in ``QIBOLAB_PLATFORMS``."""
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
"""Folder where figures and data are saved."""
OUTPUT_DIR.mkdir(exist_ok=True)
