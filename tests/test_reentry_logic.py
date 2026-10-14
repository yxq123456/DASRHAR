import numpy as np

from dasr_har.outages.generator import derive_events


def test_reentry_and_gap_age():
    mask = np.asarray([1, 1, 0, 0, 0, 1, 1], dtype=bool)
    age, reentry, duration = derive_events({"sensor": mask}, 0.02)
    assert reentry["sensor"].tolist() == [False, False, False, False, False, True, False]
    assert np.isclose(duration["sensor"][5], 0.06)
    assert np.all(age["sensor"][mask] == 0)

