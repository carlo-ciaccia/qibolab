import qblox_instruments

WAVEFORM_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#waveform-memory
"""
WAVEFORM_NUMBER = 2**10
"""Maximum number of waveforms available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#waveform-memory
"""

QCM_INSTRUCTION_MEMORY = 2**14
QRM_INSTRUCTION_MEMORY = 12288
"""Maximum number of instructions per program for the QCM and QRM modules.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#instruction-memory
"""

WEIGHT_MEMORY = 2**14
"""Maximum waveform memory available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#integrator
"""
WEIGHT_NUMBER = 2**5
"""Maximum number of weights available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#weight-memory
"""

if qblox_instruments.__version__ >= "1.0.0":
    ACQUISITION_MEMORY = 3e6
else:
    ACQUISITION_MEMORY = 2**17
"""Maximum acquisition memory available.

..note::
    Here the qblox instruments version is used as a proxy for the check if the cluster
    firmware >= 0.13.0, which due to the dependency requirements is an equivalent
    condition.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html
https://docs.qblox.com/en/main/releases.html#new-features
"""

ACQUISITION_NUMBER = 2**5
"""Maximum number of acquisitions available.

https://docs.qblox.com/en/v0.16.0/cluster/q1_sequence_processor.html#:~:text=Square%20Weight%20Acquisitions
"""

SCOPE_MEMORY = 2**14
"""Number of samples in a scope acquisition trace.

https://docs.qblox.com/en/v2026.08.0/products/qblox_instruments/tutorials/QRM/030_scope_acquisition.html
"""

QRC_SEQUENCERS = 12
"""Number of sequencers available on a QRC module.

The first :data:`QRC_READOUT_SEQUENCERS` also have an acquisition path, the others
can only drive outputs.
"""
QRC_READOUT_SEQUENCERS = 8
"""Number of QRC sequencers with an acquisition path."""
QRC_OUTPUTS = 6
"""Number of RF outputs of a QRC module."""
QRC_READOUT_PORTS = frozenset({1, 2})
"""Paired output/input (``io``) ports usable for readout on a QRC module.

Port numbers are 1-based in the qibolab channel path (``io1``), while the hardware
parameters are 0-based (``out0_in0``).
"""
QRC_ATTENUATION_STEP = 0.5
"""Resolution of the QRC input and output attenuators, in dB."""
QRC_MAX_OUTPUT_ATTENUATION = 31.5
"""Upper bound of the QRC output attenuators, in dB.

The output line has one step attenuator of 31.5 dB, part of which is already used by the
frequency plan. The actual maximum depends on the frequency (it is at least 15 dB), and
is checked by the hardware driver.

https://docs.qblox.com/en/main/products/architecture/modules/qrc.html
"""
QRC_MAX_INPUT_ATTENUATION = 63.0
"""Upper bound of the QRC input attenuators, in dB.

The input line has two step attenuators of 31.5 dB each, part of which is already used by
the frequency plan. The actual maximum depends on the frequency (it is at least 40 dB),
and is checked by the hardware driver.

https://docs.qblox.com/en/main/products/architecture/modules/qrc.html
"""

SEQUENCER_REGISTERS = 64
"""Number of registers available in the Q1 sequencer.

https://docs.qblox.com/en/v2026.08.0/products/architecture/sequencers/sequencer.html#registers
"""
