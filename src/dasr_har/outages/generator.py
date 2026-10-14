from __future__ import annotations

import hashlib

import numpy as np

from .patterns import multi_burst_mask, periodic_mask, point_mask, single_burst_mask


def generate_mask(
    length: int,
    missing_rate: float,
    pattern: str,
    rng: np.random.Generator,
    require_return: bool = False,
    horizon: int | None = None,
) -> np.ndarray:
    budget = int(np.floor(missing_rate * length))
    if budget < 0 or budget > length:
        raise ValueError("Missing rate produces an invalid budget")
    if pattern == "point":
        return point_mask(length, budget, rng)
    if pattern == "single_burst":
        return single_burst_mask(length, budget, rng, require_return, horizon)
    if pattern == "multi_burst":
        return multi_burst_mask(length, budget, rng)
    if pattern == "periodic":
        return periodic_mask(length, budget, rng)
    raise ValueError(f"Unknown outage pattern: {pattern}")


def derive_events(
    masks: dict[str, np.ndarray], delta_t: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    ages = {}
    reentries = {}
    durations = {}
    for stream, mask in masks.items():
        available = np.asarray(mask, dtype=bool)
        age = np.zeros(len(available), dtype=np.float32)
        for index in range(len(available)):
            if not available[index]:
                age[index] = (age[index - 1] if index else 0.0) + delta_t
        reentry = np.zeros(len(available), dtype=bool)
        if len(available) > 1:
            reentry[1:] = available[1:] & ~available[:-1]
        duration = np.zeros(len(available), dtype=np.float32)
        if len(available) > 1:
            duration[1:] = reentry[1:] * age[:-1]
        ages[stream] = age
        reentries[stream] = reentry
        durations[stream] = duration
    return ages, reentries, durations


class OnlineOutageGenerator:
    def __init__(
        self,
        streams: list[str],
        seed: int,
        clean_probability: float = 0.4,
        patterns: tuple[str, ...] = ("point", "single_burst", "multi_burst", "periodic"),
        missing_rate_range: tuple[float, float] = (0.1, 0.5),
    ):
        self.streams = streams
        self.seed = seed
        self.clean_probability = clean_probability
        self.patterns = patterns
        self.missing_rate_range = missing_rate_range
        self.counter = 0

    def __call__(self, sample_id: str, length: int) -> dict[str, np.ndarray]:
        digest = hashlib.sha256(f"{self.seed}:{self.counter}:{sample_id}".encode()).digest()
        self.counter += 1
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        result = {stream: np.ones(length, dtype=bool) for stream in self.streams}
        if rng.random() < self.clean_probability:
            return result
        stream = self.streams[int(rng.integers(0, len(self.streams)))]
        pattern = self.patterns[int(rng.integers(0, len(self.patterns)))]
        rate = float(rng.uniform(*self.missing_rate_range))
        result[stream] = generate_mask(
            length,
            rate,
            pattern,
            rng,
            require_return=pattern == "single_burst" and rng.random() < 0.5,
        )
        return result


def overlapping_burst_masks(
    length: int,
    missing_rate: float,
    streams: list[str],
    rng: np.random.Generator,
    overlap_ratio: float,
) -> dict[str, np.ndarray]:
    if not streams:
        raise ValueError("At least one interrupted stream is required")
    budget = int(np.floor(missing_rate * length))
    shift = int(round(budget * (1.0 - overlap_ratio)))
    span = budget + shift * (len(streams) - 1)
    if span > length:
        raise ValueError("Requested overlap condition does not fit the window")
    start = int(rng.integers(0, length - span + 1))
    result = {}
    for index, stream in enumerate(streams):
        mask = np.ones(length, dtype=bool)
        segment_start = start + index * shift
        mask[segment_start : segment_start + budget] = False
        result[stream] = mask
    return result

