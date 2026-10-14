from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .schema import Recording


class WisdmAdapter:
    stream_paths = {
        "phone_acc": ("phone", "accel"),
        "phone_gyro": ("phone", "gyro"),
        "watch_acc": ("watch", "accel"),
        "watch_gyro": ("watch", "gyro"),
    }

    def __init__(self, raw_dir: str | Path, sample_rate: int = 20):
        self.raw_dir = Path(raw_dir)
        self.sample_rate = sample_rate

    def _paths(self, parts: tuple[str, str]) -> list[Path]:
        result = []
        for path in self.raw_dir.rglob("*.txt"):
            lowered = [part.lower() for part in path.parts]
            if parts[0] in lowered and parts[1] in lowered:
                result.append(path)
        return sorted(result)

    def _read(self, path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rows: dict[str, list[tuple[int, str, float, float, float]]] = defaultdict(list)
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                fields = line.strip().rstrip(";").split(",")
                if len(fields) < 6:
                    continue
                try:
                    subject = str(int(fields[0]))
                    timestamp = int(fields[2])
                    xyz = tuple(float(value) for value in fields[3:6])
                except ValueError:
                    continue
                rows[subject].append((timestamp, fields[1].strip(), *xyz))
        result = {}
        for subject, subject_rows in rows.items():
            subject_rows.sort(key=lambda row: row[0])
            times = np.asarray([row[0] for row in subject_rows], dtype=np.float64)
            labels = np.asarray([row[1] for row in subject_rows])
            values = np.asarray([row[2:] for row in subject_rows], dtype=np.float32)
            result[subject] = (times, labels, values)
        return result

    def load(self) -> list[Recording]:
        data: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for stream, parts in self.stream_paths.items():
            paths = self._paths(parts)
            if not paths:
                raise FileNotFoundError(f"No WISDM files found for {stream} under {self.raw_dir}")
            for path in paths:
                for subject, values in self._read(path).items():
                    data[subject][stream].append(values)
        recordings = []
        for mapped_subject, subject in enumerate(sorted(data, key=lambda value: int(value)), start=1):
            streams = {}
            for stream in self.stream_paths:
                parts = data[subject].get(stream, [])
                if not parts:
                    break
                times = np.concatenate([part[0] for part in parts])
                labels = np.concatenate([part[1] for part in parts])
                values = np.concatenate([part[2] for part in parts])
                order = np.argsort(times, kind="stable")
                streams[stream] = (times[order], labels[order], values[order])
            if len(streams) != len(self.stream_paths):
                continue
            recordings.extend(self._align_subject(str(mapped_subject), subject, streams))
        if not recordings:
            raise ValueError("No WISDM subjects contained all four streams")
        return recordings

    def _align_subject(
        self,
        subject: str,
        original_subject: str,
        streams: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> list[Recording]:
        scale = max(np.median(np.diff(value[0])) for value in streams.values())
        unit = 1e9 if scale > 1e6 else 1.0
        start = max(value[0][0] for value in streams.values())
        stop = min(value[0][-1] for value in streams.values())
        step = unit / self.sample_rate
        grid = np.arange(start, stop + step * 0.25, step, dtype=np.float64)
        aligned = {}
        masks = {}
        label_votes = []
        for name, (times, labels, values) in streams.items():
            indices = np.searchsorted(times, grid, side="right") - 1
            valid = indices >= 0
            clipped = np.clip(indices, 0, len(times) - 1)
            age = grid - times[clipped]
            valid &= (age >= 0) & (age <= step * 1.5)
            aligned[name] = values[clipped]
            masks[name] = valid
            label_votes.append(labels[clipped])
        label_stack = np.stack(label_votes)
        labels = np.asarray(
            [
                np.unique(label_stack[:, index], return_counts=True)[0][
                    np.argmax(np.unique(label_stack[:, index], return_counts=True)[1])
                ]
                for index in range(label_stack.shape[1])
            ]
        )
        valid = np.logical_or.reduce(list(masks.values())) & (labels != "")
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return []
        gaps = np.flatnonzero(np.diff(indices) != 1) + 1
        groups = np.split(indices, gaps)
        result = []
        for group_index, group in enumerate(groups):
            if group.size < self.sample_rate:
                continue
            result.append(
                Recording(
                    streams={name: value[group] for name, value in aligned.items()},
                    native_mask={name: value[group] for name, value in masks.items()},
                    labels=labels[group],
                    timestamps=(grid[group] - grid[group][0]) / unit,
                    subject_id=subject,
                    session_id=f"subject{original_subject}_{group_index:03d}",
                )
            )
        return result

