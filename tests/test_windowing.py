import numpy as np

from dasr_har.data.schema import Recording
from dasr_har.data.windowing import window_recordings


def test_window_length_step_and_purity():
    recording = Recording(
        streams={"a": np.ones((20, 2), dtype=np.float32)},
        native_mask={"a": np.ones(20, dtype=bool)},
        labels=np.asarray([1] * 10 + [2] * 10),
        timestamps=np.arange(20),
        subject_id="1",
        session_id="s",
    )
    windows = window_recordings([recording], 10, 5, 0.8)
    assert len(windows) == 2
    assert all(window.streams["a"].shape == (10, 2) for window in windows)

