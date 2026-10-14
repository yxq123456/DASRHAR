from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def paired_subject_bootstrap(
    first: np.ndarray,
    second: np.ndarray,
    samples: int = 10000,
    seed: int = 3407,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    difference = first - second
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(samples, len(difference)))
    estimates = difference[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(difference.mean()), float(low), float(high)


def exact_wilcoxon(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    difference = difference[difference != 0]
    if difference.size == 0:
        return 1.0
    ranks = stats.rankdata(np.abs(difference), method="average")
    scaled = np.rint(ranks * 2).astype(np.int64)
    total = int(scaled.sum())
    observed_positive = int(scaled[difference > 0].sum())
    observed = min(observed_positive, total - observed_positive)
    counts = [0] * (total + 1)
    counts[0] = 1
    reachable = 0
    for rank in scaled:
        value = int(rank)
        for subtotal in range(reachable, -1, -1):
            if counts[subtotal]:
                counts[subtotal + value] += counts[subtotal]
        reachable += value
    extreme = sum(
        count
        for subtotal, count in enumerate(counts)
        if min(subtotal, total - subtotal) <= observed
    )
    return float(extreme / (2 ** len(scaled)))


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    observed = abs(values.mean())
    if len(values) <= 20:
        estimates = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            estimates.append(abs((values * np.asarray(signs)).mean()))
        return float(np.mean(np.asarray(estimates) >= observed))
    rng = np.random.default_rng(3407)
    signs = rng.choice((-1.0, 1.0), size=(100000, len(values)))
    estimates = np.abs((signs * values).mean(axis=1))
    return float((np.sum(estimates >= observed) + 1) / (len(estimates) + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def subject_spearman(duration: np.ndarray, gate: np.ndarray) -> float:
    coefficient = stats.spearmanr(duration, gate, nan_policy="omit").statistic
    return float(coefficient)

