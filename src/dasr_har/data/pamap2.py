from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .preprocessing import causal_forward_fill, contiguous_recordings
from .schema import Recording


class Pamap2Adapter:
    stream_columns = {
        "heart_rate": [2],
        "hand_imu": list(range(3, 16)),
        "chest_imu": list(range(20, 33)),
        "ankle_imu": list(range(37, 50)),
    }

    def __init__(self, raw_dir: str | Path, retained_labels: list[int], sample_rate: int = 100):
        self.raw_dir = Path(raw_dir)
        self.retained_labels = np.asarray(retained_labels)
        self.sample_rate = sample_rate

    def load(self) -> list[Recording]:
        paths = sorted(self.raw_dir.rglob("subject*.dat"))
        if not paths:
            raise FileNotFoundError(f"No PAMAP2 protocol files found under {self.raw_dir}")
        recordings: list[Recording] = []
        for path in paths:
            match = re.search(r"subject10(\d)", path.stem, re.IGNORECASE)
            if match is None:
                continue
            subject = str(int(match.group(1)))
            values = np.loadtxt(path, dtype=np.float64)
            timestamps = values[:, 0]
            labels = values[:, 1].astype(np.int64)
            raw_streams = {
                name: values[:, columns].astype(np.float32)
                for name, columns in self.stream_columns.items()
            }
            masks = {
                name: np.isfinite(stream).all(axis=1) for name, stream in raw_streams.items()
            }
            streams = {name: causal_forward_fill(stream) for name, stream in raw_streams.items()}
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

