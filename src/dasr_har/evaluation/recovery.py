from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import macro_f1, recovery_deficit_area


@dataclass
class RecoveryCurve:
    offsets: np.ndarray
    scores: np.ndarray
    event_counts: np.ndarray
    rda: float


def event_aligned_curve(
    labels: list[np.ndarray],
    predictions: list[np.ndarray],
    reentries: list[np.ndarray],
    maximum_offset: int,
    clean_score: float,
) -> RecoveryCurve:
    offset_labels = [[] for _ in range(maximum_offset + 1)]
    offset_predictions = [[] for _ in range(maximum_offset + 1)]
    for sequence_labels, sequence_predictions, sequence_reentries in zip(
        labels, predictions, reentries
    ):
        for event in np.flatnonzero(sequence_reentries):
            for offset in range(maximum_offset + 1):
                index = event + offset
                if index >= len(sequence_labels):
                    break
                offset_labels[offset].append(int(sequence_labels[index]))
                offset_predictions[offset].append(int(sequence_predictions[index]))
    scores = np.asarray(
        [
            macro_f1(np.asarray(target), np.asarray(prediction)) if target else np.nan
            for target, prediction in zip(offset_labels, offset_predictions)
        ]
    )
    counts = np.asarray([len(values) for values in offset_labels], dtype=np.int64)
    valid = np.isfinite(scores)
    offsets = np.arange(maximum_offset + 1, dtype=np.float64)
    rda = recovery_deficit_area(offsets[valid], scores[valid], clean_score)
    return RecoveryCurve(offsets, scores, counts, rda)


def fixed_window_rda(
    horizon_scores: dict[float, float], clean_score: float
) -> float:
    horizons = np.asarray(sorted(horizon_scores), dtype=np.float64)
    scores = np.asarray([horizon_scores[value] for value in horizons], dtype=np.float64)
    return recovery_deficit_area(horizons, scores, clean_score)

