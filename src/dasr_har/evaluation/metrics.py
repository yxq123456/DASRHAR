from __future__ import annotations

import numpy as np


def confusion_matrix(
    target: np.ndarray, prediction: np.ndarray, num_classes: int | None = None
) -> np.ndarray:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    classes = num_classes or int(max(target.max(initial=0), prediction.max(initial=0)) + 1)
    result = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(result, (target, prediction), 1)
    return result


def macro_f1(
    target: np.ndarray, prediction: np.ndarray, num_classes: int | None = None
) -> float:
    matrix = confusion_matrix(target, prediction, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = matrix.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    scores = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    supported = matrix.sum(axis=1) > 0
    return float(scores[supported].mean()) if supported.any() else 0.0


def accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    return float((target == prediction).mean()) if target.size else 0.0


def mean_robust(scores: dict[str, float]) -> tuple[float, float]:
    conditions = ["point_30", "burst_30", "multi_30", "periodic_30", "burst_50"]
    values = np.asarray([scores[name] for name in conditions], dtype=np.float64)
    return float(values.mean()), float(values.min())


def recovery_deficit_area(
    horizons: np.ndarray,
    scores: np.ndarray,
    clean_score: float,
) -> float:
    horizons = np.asarray(horizons, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(horizons)
    horizons = horizons[order]
    deficits = np.maximum(clean_score - scores[order], 0.0)
    span = horizons[-1] - horizons[0]
    if span <= 0:
        return float(deficits.mean())
    return float(np.trapz(deficits, horizons) / span)


def state_gap(clean_states: np.ndarray, outage_states: np.ndarray) -> np.ndarray:
    clean = np.asarray(clean_states, dtype=np.float64)
    outage = np.asarray(outage_states, dtype=np.float64)
    numerator = (clean * outage).sum(axis=-1)
    denominator = np.linalg.norm(clean, axis=-1) * np.linalg.norm(outage, axis=-1)
    similarity = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    return 1.0 - similarity

