import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv")
from omnit2i.analysis.ipcw import evaluate_fold, validate_test_rows, weighted_seed_summary


def test_perfect_risk_and_outer_training_censoring_reference():
    train = pd.DataFrame({"time": [100., 500., 1200., 2000., 2500., 3000.], "event": [True] * 6})
    test = pd.DataFrame({"time": [200., 700., 1400., 2100.], "event": [True] * 4,
                         "risk": [4., 3., 2., 1.]})
    result = evaluate_fold(train, test, "risk")
    for metric in ["uno_c_tau_5y", "auc_1y", "auc_3y", "auc_5y"]:
        assert result[metric] == pytest.approx(1.)
    assert result["censoring_survival_5y"] == 1.
    censored_train = train.copy()
    censored_train.loc[1, "event"] = False
    assert evaluate_fold(censored_train, test, "risk")["censoring_survival_5y"] < 1.


def test_oof_membership_and_test_count_weighting():
    expected = pd.DataFrame({"patient": ["a", "b"], "event": [True, False], "time": [1., 2.]})
    validate_test_rows(expected.iloc[::-1], expected, 101, 0, "natural")
    with pytest.raises(ValueError, match="membership"):
        validate_test_rows(expected.iloc[:1], expected, 101, 0, "natural")
    metrics = ["uno_c_tau_5y", "auc_1y", "auc_3y", "auc_5y", "mean_auc_1_3_5y"]
    frame = pd.DataFrame({"scenario": "natural", "seed": 101, "n_test": [1, 3], "test_events": [1, 2],
                          **{key: [.5, 1.] for key in metrics}})
    result = weighted_seed_summary(frame).iloc[0]
    assert result.n_patients == 4 and result.auc_1y == .875
