from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import Recording, Window


def causal_forward_fill(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for column in range(result.shape[1]):
        valid = np.flatnonzero(np.isfinite(result[:, column]))
        if valid.size == 0:
            continue
        first = int(valid[0])
        for index in range(first + 1, result.shape[0]):
            if not np.isfinite(result[index, column]):
                result[index, column] = result[index - 1, column]
    return result


def contiguous_recordings(
    streams: dict[str, np.ndarray],
    native_mask: dict[str, np.ndarray],
    labels: np.ndarray,
    timestamps: np.ndarray,
    subject_id: str,
    session_id: str,
    valid: np.ndarray,
    max_gap: float,
) -> list[Recording]:
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(
        (np.diff(indices) != 1) | (np.diff(timestamps[indices]) > max_gap)
    ) + 1
    groups = np.split(indices, breaks)
    result = []
    for group_index, group in enumerate(groups):
        if group.size == 0:
            continue
        result.append(
            Recording(
                streams={key: value[group] for key, value in streams.items()},
                native_mask={key: value[group] for key, value in native_mask.items()},
                labels=labels[group],
                timestamps=timestamps[group],
                subject_id=str(subject_id),
                session_id=f"{session_id}_{group_index:03d}",
            )
        )
    return result


def save_windows(windows: Iterable[Window], path: str | Path) -> None:
    items = list(windows)
    if not items:
        raise ValueError("No eligible windows were produced")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream_names = list(items[0].streams)
    raw_labels = sorted({item.label for item in items}, key=str)
    label_to_index = {str(label): index for index, label in enumerate(raw_labels)}
    arrays: dict[str, np.ndarray] = {}
    for name in stream_names:
        arrays[f"x__{name}"] = np.stack([item.streams[name] for item in items]).astype(np.float32)
        arrays[f"native__{name}"] = np.stack(
            [item.native_mask[name] for item in items]
        ).astype(bool)
    arrays["label"] = np.asarray([label_to_index[str(item.label)] for item in items], dtype=np.int64)
    arrays["label_sequence"] = np.stack(
        [np.asarray([label_to_index[str(value)] for value in item.labels]) for item in items]
    ).astype(np.int64)
    arrays["subject_id"] = np.asarray([item.subject_id for item in items])
    arrays["sample_id"] = np.asarray([item.sample_id for item in items])
    arrays["session_id"] = np.asarray([item.session_id for item in items])
    arrays["start_index"] = np.asarray([item.start_index for item in items], dtype=np.int64)
    arrays["metadata"] = np.asarray(
        json.dumps(
            {
                "streams": stream_names,
                "raw_labels": [str(value) for value in raw_labels],
                "num_windows": len(items),
            }
        )
    )
    np.savez_compressed(output, **arrays)


def load_metadata(path: str | Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"]))

