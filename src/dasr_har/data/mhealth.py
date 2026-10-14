from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .preprocessing import contiguous_recordings
from .schema import Recording


class MHealthAdapter:
    stream_columns = {
        "chest_motion": [0, 1, 2],
        "right_wrist_motion": list(range(14, 23)),
        "left_ankle_motion": list(range(5, 14)),
    }

    def __init__(self, raw_dir: str | Path, retained_labels: list[int], sample_rate: int = 50):
        self.raw_dir = Path(raw_dir)
        self.retained_labels = np.asarray(retained_labels)
        self.sample_rate = sample_rate

    def load(self) -> list[Recording]:
        paths = sorted(self.raw_dir.rglob("mHealth_subject*.log"))
        if not paths:
            raise FileNotFoundError(f"No MHEALTH subject logs found under {self.raw_dir}")
        recordings: list[Recording] = []
        for path in paths:
            match = re.search(r"subject(\d+)", path.stem, re.IGNORECASE)
            if match is None:
                continue
            subject = str(int(match.group(1)))
            values = np.loadtxt(path, dtype=np.float32)
            labels = values[:, 23].astype(np.int64)
            streams = {
                name: values[:, columns].astype(np.float32)
                for name, columns in self.stream_columns.items()
            }
            masks = {name: np.isfinite(value).all(axis=1) for name, value in streams.items()}
            timestamps = np.arange(len(labels), dtype=np.float64) / self.sample_rate
            valid = np.isin(labels, self.retained_labels)
            recordings.extend(
                contiguous_recordings(
                    streams,
                    masks,
                    labels,
                    timestamps,
                    subject,
                    path.stem,
                    valid,
                    1.5 / self.sample_rate,
                )
            )
        return recordings

