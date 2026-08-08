import numpy as np
import pytest

from synthefy_nori.evaluation.protocol import MaterializedSplit, compose, decompose


@pytest.mark.parametrize(
    ("fold", "folds_per_repeat", "repeat", "fold_in_repeat"),
    [(0, 1, 0, 0), (7, 3, 2, 1), (29, 3, 9, 2)],
)
def test_fold_identity_round_trips(fold, folds_per_repeat, repeat, fold_in_repeat):
    assert decompose(fold, folds_per_repeat) == (repeat, fold_in_repeat)
    assert compose(repeat, fold_in_repeat, folds_per_repeat) == fold


def test_materialized_split_rejects_misaligned_rows():
    with pytest.raises(ValueError, match="row mismatch"):
        MaterializedSplit(
            X_train=np.zeros((3, 2)),
            y_train=np.zeros(2),
            X_test=np.zeros((1, 2)),
            y_test=np.zeros(1),
            n_features=2,
        )
