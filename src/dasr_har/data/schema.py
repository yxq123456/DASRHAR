from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Recording:
    streams: dict[str, np.ndarray]
    native_mask: dict[str, np.ndarray]
    labels: np.ndarray
    timestamps: np.ndarray
    subject_id: str
    session_id: str


@dataclass
class Window:
    streams: dict[str, np.ndarray]
    native_mask: dict[str, np.ndarray]
    labels: np.ndarray
    label: int | str
    subject_id: str
    sample_id: str
    session_id: str
    start_index: int

