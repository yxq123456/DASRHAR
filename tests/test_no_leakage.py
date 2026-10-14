import numpy as np

from dasr_har.data.dataset import fit_normalizer


def test_normalizer_uses_only_train_subjects(tmp_path):
    path = tmp_path / "windows.npz"
    np.savez_compressed(
        path,
        x__a=np.asarray([[[1.0], [1.0]], [[100.0], [100.0]]]),
        native__a=np.ones((2, 2), dtype=bool),
        subject_id=np.asarray(["1", "2"]),
        metadata=np.asarray('{"streams": ["a"]}'),
    )
    normalizer = fit_normalizer(path, ["1"])
    assert normalizer["a"]["mean"] == [1.0]

