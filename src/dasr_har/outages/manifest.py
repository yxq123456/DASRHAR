from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ManifestEntry:
    sample_id: str
    pattern: str
    missing_rate: float
    segments: dict[str, list[list[int]]]
    duration_seconds: float
    recovery_horizon_seconds: float
    mask_seed: int
    condition: str = ""


class Manifest:
    def __init__(self, entries: list[ManifestEntry]):
        self.entries = entries
        self.by_sample = {entry.sample_id: entry for entry in entries}

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        entries = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entries.append(ManifestEntry(**json.loads(line)))
        return cls(entries)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")

    def checksum(self) -> str:
        payload = "\n".join(json.dumps(asdict(entry), sort_keys=True) for entry in self.entries)
        return hashlib.sha256(payload.encode()).hexdigest()

    def transform(self, streams: list[str]) -> "ManifestTransform":
        return ManifestTransform(self.by_sample, streams)


class ManifestTransform:
    def __init__(self, entries: dict[str, ManifestEntry], streams: list[str]):
        self.entries = entries
        self.streams = streams

    def __call__(self, sample_id: str, length: int) -> dict[str, np.ndarray]:
        masks = {stream: np.ones(length, dtype=bool) for stream in self.streams}
        entry = self.entries.get(sample_id)
        if entry is None:
            return masks
        for stream, segments in entry.segments.items():
            if stream not in masks:
                continue
            for start, stop in segments:
                if start < 0 or stop >= length or stop < start:
                    raise ValueError(f"Invalid segment for {sample_id}: {[start, stop]}")
                masks[stream][start : stop + 1] = False
        return masks


def entry_from_dict(value: dict[str, Any]) -> ManifestEntry:
    return ManifestEntry(**value)
