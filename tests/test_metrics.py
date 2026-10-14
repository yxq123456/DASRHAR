import numpy as np

from dasr_har.evaluation.metrics import macro_f1, recovery_deficit_area, state_gap


def test_macro_f1_perfect():
    target = np.asarray([0, 1, 2, 2])
    assert macro_f1(target, target, 3) == 1.0


def test_recovery_deficit_area():
    value = recovery_deficit_area(
        np.asarray([0.0, 0.5, 1.0]), np.asarray([0.5, 0.75, 1.0]), 1.0
    )
    assert np.isclose(value, 0.25)


def test_state_gap_identity():
    values = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
    assert np.allclose(state_gap(values, values), 0.0)
