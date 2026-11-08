import numpy as np
import pytest

pytest.importorskip("numba")
from omnit2i.analysis.paired_bootstrap import comparable_pairs, pair_scores, weighted_c


def test_tied_event_censor_pairs_and_patient_multiplicity():
    time = np.array([1., 1., 2., 3.])
    event = np.array([True, False, True, True])
    left, right = comparable_pairs(time, event, np.zeros(4))
    assert set(zip(left, right)) == {(0, 1), (0, 2), (0, 3), (2, 3)}
    risks = np.array([[[4., 4., 2., 1.]], [[1., 2., 3., 4.]]])
    scores = pair_scores(risks, left, right)
    np.testing.assert_allclose(weighted_c(np.ones((1, 4)), left, right, scores), [[.875, 0.]])
    # Duplicating patient 0 doubles each of its pair contributions.
    np.testing.assert_allclose(weighted_c(np.array([[2, 1, 1, 1]]), left, right, scores), [[6/7, 0.]])
    assert np.isnan(weighted_c(np.zeros((1, 4)), left, right, scores)).all()
