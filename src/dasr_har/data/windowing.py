from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import Recording, Window


def window_recordings(
    recordings: list[Recording],
    length: int,
    step: int,
    purity: float,
    max_per_subject_activity: int | None = None,
    seed: int = 3407,
) -> list[Window]:
    windows: list[Window] = []
    for recording in recordings:
        for start in range(0, len(recording.labels) - length + 1, step):
            stop = start + length
            labels = recording.labels[start:stop]
            values, counts = np.unique(labels, return_counts=True)
            choice = int(np.argmax(counts))
            if counts[choice] / length < purity:
                continue
            sample_id = f"{recording.subject_id}_{recording.session_id}_{start:08d}"
            windows.append(
                Window(
                    streams={key: value[start:stop] for key, value in recording.streams.items()},
                    native_mask={
                        key: value[start:stop] for key, value in recording.native_mask.items()
                    },
                    labels=labels,
                    label=values[choice].item(),
                    subject_id=recording.subject_id,
                    sample_id=sample_id,
                    session_id=recording.session_id,
                    start_index=start,
                )
            )
    if max_per_subject_activity is None:
        return windows
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[Window]] = defaultdict(list)
    for window in windows:
        groups[(window.subject_id, str(window.label))].append(window)
    selected: list[Window] = []
    for key in sorted(groups):
        group = groups[key]
        if len(group) > max_per_subject_activity:
            indices = np.sort(rng.choice(len(group), max_per_subject_activity, replace=False))
            selected.extend(group[int(index)] for index in indices)
        else:
            selected.extend(group)
    return sorted(selected, key=lambda item: item.sample_id)

