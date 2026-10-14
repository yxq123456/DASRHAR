from __future__ import annotations

import numpy as np


def point_mask(length: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones(length, dtype=bool)
    if budget:
        mask[rng.choice(length, budget, replace=False)] = False
    return mask


def single_burst_mask(
    length: int,
    budget: int,
    rng: np.random.Generator,
    require_return: bool = False,
    horizon: int | None = None,
) -> np.ndarray:
    mask = np.ones(length, dtype=bool)
    if budget == 0:
        return mask
    if horizon is not None:
        start = length - horizon - budget
        if start < 0 or start + budget > length:
            raise ValueError("Requested duration and horizon do not fit the window")
    else:
        maximum = length - budget - int(require_return)
        if maximum < 0:
            raise ValueError("Missing budget leaves no requested return sample")
        start = int(rng.integers(0, maximum + 1))
    mask[start : start + budget] = False
    return mask


def _positive_composition(total: int, parts: int, rng: np.random.Generator) -> np.ndarray:
    if parts == 1:
        return np.asarray([total], dtype=np.int64)
    cuts = np.sort(rng.choice(np.arange(1, total), parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(np.int64)


def multi_burst_mask(
    length: int,
    budget: int,
    rng: np.random.Generator,
    bursts: int | None = None,
) -> np.ndarray:
    mask = np.ones(length, dtype=bool)
    if budget == 0:
        return mask
    maximum_bursts = min(4, budget, length - budget + 1)
    minimum_bursts = min(2, maximum_bursts)
    count = bursts or int(rng.integers(minimum_bursts, maximum_bursts + 1))
    count = max(1, min(count, maximum_bursts))
    lengths = _positive_composition(budget, count, rng)
    free = length - budget - (count - 1)
    gaps = rng.multinomial(free, np.full(count + 1, 1.0 / (count + 1)))
    cursor = int(gaps[0])
    for index, segment_length in enumerate(lengths):
        mask[cursor : cursor + segment_length] = False
        cursor += int(segment_length)
        if index < count - 1:
            cursor += 1 + int(gaps[index + 1])
    return mask


def periodic_mask(length: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones(length, dtype=bool)
    if budget == 0:
        return mask
    block = max(1, min(int(round(length * 0.025)), budget))
    period = max(block + 1, int(round(length * block / budget)))
    offset = int(rng.integers(0, min(period, length)))
    order = []
    for cycle in range(-1, length // period + 2):
        start = offset + cycle * period
        order.extend(range(max(0, start), min(length, start + block)))
    order = list(dict.fromkeys(order))
    if len(order) < budget:
        remaining = np.setdiff1d(np.arange(length), np.asarray(order, dtype=np.int64))
        order.extend(rng.choice(remaining, budget - len(order), replace=False).tolist())
    mask[np.asarray(order[:budget], dtype=np.int64)] = False
    return mask


def mask_to_segments(mask: np.ndarray) -> list[list[int]]:
    missing = ~np.asarray(mask, dtype=bool)
    padded = np.pad(missing.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [[int(start), int(stop - 1)] for start, stop in zip(starts, stops)]

