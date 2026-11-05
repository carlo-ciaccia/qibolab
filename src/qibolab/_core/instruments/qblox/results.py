from functools import reduce
from operator import or_
from typing import TypedDict
from uuid import UUID

import numpy as np

from qibolab._core.execution_parameters import AcquisitionType
from qibolab._core.identifier import ChannelId, Result
from qibolab._core.pulses.modulation import demodulate
from qibolab._core.pulses.pulse import PulseId

from .identifiers import SequencerId, SequencerMap, SlotId
from .sequence import Q1Sequence, acquisition

__all__ = []


def _fill_empty_lenghts(
    weighted: dict[acquisition.MeasureId, int | None],
    defaults: dict[tuple[SlotId, SequencerId], int],
    locations: dict[acquisition.MeasureId, tuple[SlotId, SequencerId]],
) -> dict[acquisition.MeasureId, int]:
    return {
        k: v if v is not None else defaults[locations[k]] for k, v in weighted.items()
    }


def integration_lengths(
    sequences: dict[ChannelId, Q1Sequence],
    sequencers: SequencerMap,
    configured_lengths: dict[SlotId, dict[SequencerId, int | None]],
) -> dict[acquisition.MeasureId, int]:
    channels_to_sequencer = {
        ch: (mod, seq) for mod, chs in sequencers.items() for ch, seq in chs.items()
    }
    lengths = reduce(or_, (seq.integration_lengths for seq in sequences.values()))
    locations = {
        acq: channels_to_sequencer[ch]
        for ch, seq in sequences.items()
        for acq in seq.acquisitions
    }
    required_defaults = {
        locations[acq] for acq, length in lengths.items() if length is None
    }
    defaults = {}
    for mod_id, seq_id in required_defaults:
        length = configured_lengths.get(mod_id, {}).get(seq_id)
        if length is None:
            raise ValueError(
                f"No configured integration length for module {mod_id}, "
                f"sequencer {seq_id}."
            )
        defaults[mod_id, seq_id] = length

    return _fill_empty_lenghts(
        lengths,
        defaults,
        locations,
    )


IndividualScope = TypedDict(
    "IndividualScope",
    {"data": list[float], "out-of-range": bool, "avg_cnt": int},
)


class ScopeData(TypedDict):
    path0: IndividualScope
    path1: IndividualScope
    path2: IndividualScope
    path3: IndividualScope


class Integration(TypedDict):
    path0: list[float]
    path1: list[float]
    path2: list[float]
    path3: list[float]


Thresholded = list[int]


class BinData(TypedDict):
    integration: Integration
    threshold: Thresholded
    valid: list
    avg_cnt: list[int]


class Data(TypedDict):
    scope: ScopeData
    bins: BinData


class IndexedData(TypedDict):
    index: int
    acquisition: Data


AcquiredData = dict[str, IndexedData]


def _integration(data: Integration, length: int, input_index: int = 0) -> Result:
    path = 2 * input_index
    res = np.array([data[f"path{path}"], data[f"path{path + 1}"]])
    return np.moveaxis(res, 0, -1) / length


def _scope(
    data: ScopeData,
    frequency: float | None,
    sampling_rate: float,
    input_index: int = 0,
) -> Result:
    path = 2 * input_index
    res = np.array([data[f"path{path}"]["data"], data[f"path{path + 1}"]["data"]])
    # Demodulate scope data to baseband when a frequency is provided.
    if frequency is not None:
        res = demodulate(res, frequency, sampling_rate)
    return np.moveaxis(res, 0, -1)


def _classification(data: Thresholded) -> Result:
    return np.array(data)


def _extract(
    data: Data,
    length: int | None,
    acquisition: AcquisitionType,
    input_index: int,
    frequency: float | None,
    sampling_rate: float,
) -> Result:
    if acquisition is AcquisitionType.INTEGRATION:
        if length is None:
            raise ValueError("Integration result extraction requires a length.")
        return _integration(data["bins"]["integration"], length, input_index)
    if acquisition is AcquisitionType.DISCRIMINATION:
        return _classification(data["bins"]["threshold"])
    if acquisition is AcquisitionType.RAW:
        return _scope(data["scope"], frequency, sampling_rate, input_index)
    return np.array([])


def extract(
    acquired_data: dict[ChannelId, AcquiredData],
    lengths: dict[acquisition.MeasureId, int],
    acquisition: AcquisitionType,
    shape: tuple[int, ...],
    sampling_rate: float,
    frequencies: dict[ChannelId, float],
    input_indices: dict[ChannelId, int] | None = None,
) -> dict[PulseId, Result]:
    # TODO: check if the `lengths` info coincide with the
    # idata["acquisition"]["bins"]["avg_cnt"]
    return {
        UUID(acq): _extract(
            idata["acquisition"],
            lengths.get(UUID(acq)),
            acquisition,
            (input_indices or {}).get(channel, 0),
            frequencies.get(channel, None),
            sampling_rate,
        ).reshape(shape)
        for channel, data in acquired_data.items()
        for acq, idata in data.items()
    }
