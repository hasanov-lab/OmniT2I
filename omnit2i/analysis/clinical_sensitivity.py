"""Two fixed sensitivity specifications plus one age-PH diagnostic follow-up.

Uses the primary clinical cohort without refitting OmniT2I. All P values/CIs remain post-hoc, conditional.
"""

import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test

from .clinical import FOLDS


SPECS = {
    "age_cubic_spline_df3": ("risk + bs(age10, df=3) + sex + C(stage)", ["cancer_type"]),
    "cancer_by_stage_strata": ("risk + age10 + sex", ["cancer_type", "stage"]),
    "cancer_by_stage_age_time": ("risk + age10 + age10_after_year + sex", ["cancer_type", "stage"]),
}


def split_at_one_year(data):
    """Start-stop records for age * I(t > 365.25 days), not age * observed time."""
    early = data.rename(columns={"time": "stop"}).copy()
    late = early.loc[early.stop.gt(365.25)].copy()
    early["start"] = 0.0
    early["event"] = early.event * early.stop.le(365.25)
    early["stop"] = early.stop.clip(upper=365.25)
    early["age10_after_year"] = 0.0
    late["start"] = 365.25
    late["age10_after_year"] = late.age10
    long = pd.concat([early, late], ignore_index=True)
    assert long.stop.gt(long.start).all()
    original = data.set_index("patient").sort_index()
    assert long.groupby("patient").event.sum().astype(int).sort_index().equals(original.event.astype(int))
    duration = long.assign(duration=long.stop - long.start).groupby("patient").duration.sum().sort_index()
    assert np.allclose(duration, original.time)
    return long


def analyze(data, output_dir):
    primary = pd.read_csv(output_dir / "fold_adjusted_cox.csv").set_index("fold")
    rows, diagnostic_rows, warning_rows = [], [], []
    for name, (formula, strata) in SPECS.items():
        for fold in FOLDS:
            d = data.loc[data.fold.eq(fold), ["patient", "time", "event", "risk", "age10", "sex", "stage", "cancer_type"]].copy()
            assert len(d) == primary.loc[fold, "patients"]
            assert d.event.sum() == primary.loc[fold, "events"]
            counts = d.groupby(strata, observed=True).agg(patients=("event", "size"), events=("event", "sum"))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                if name == "cancer_by_stage_age_time":
                    fit = CoxTimeVaryingFitter(penalizer=0.0).fit(
                        split_at_one_year(d), event_col="event", id_col="patient",
                        start_col="start", stop_col="stop", strata=strata, formula=formula)
                    # The static-Cox residual test is not supported for this fit.
                    risk_ph_p = np.nan
                else:
                    fit = CoxPHFitter(penalizer=0.0).fit(d, "time", "event", strata=strata, formula=formula)
                    ph = proportional_hazard_test(fit, d, time_transform="rank").summary
                    diagnostic_rows.append(ph.reset_index().rename(columns={"index": "covariate"}).assign(specification=name, fold=fold))
                    risk_ph_p = float(ph.loc["risk", "p"])
            warning_rows.extend({"specification": name, "fold": fold, "category": w.category.__name__, "message": str(w.message)} for w in captured)
            risk = fit.summary.loc["risk"]
            row = {"specification": name, "fold": fold, "patients": len(d), "events": int(d.event.sum()),
                   "strata": len(counts), "strata_with_events": int(counts.events.gt(0).sum()),
                   "patients_in_strata_without_events": int(counts.loc[counts.events.eq(0), "patients"].sum()),
                   "risk_hr": float(risk["exp(coef)"]),
                   "ci95_lower": float(risk["exp(coef) lower 95%"]),
                   "ci95_upper": float(risk["exp(coef) upper 95%"]),
                   "nominal_wald_p": float(risk["p"]), "risk_ph_rank_p": risk_ph_p,
                   "fit_warnings": len(captured)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = pd.DataFrame(rows)
    # Runnable checks: all specifications must retain every fold and patient;
    # no filtering on effect direction, P value or diagnostic result.
    assert len(result) == len(SPECS) * len(FOLDS)
    assert not result.duplicated(["specification", "fold"]).any()
    assert result.groupby("specification").patients.sum().eq(len(data)).all()
    numeric = result[["risk_hr", "ci95_lower", "ci95_upper", "nominal_wald_p"]]
    assert np.isfinite(numeric).all().all()
    assert result.ci95_lower.gt(0).all() and result.ci95_upper.gt(result.ci95_lower).all()
    assert result.nominal_wald_p.between(0, 1).all()
    static = result.specification.ne("cancer_by_stage_age_time")
    assert result.loc[static, "risk_ph_rank_p"].between(0, 1).all()
    assert result.loc[~static, "risk_ph_rank_p"].isna().all()
    result.to_csv(output_dir / "sensitivity_fold_results.csv", index=False)
    pd.concat(diagnostic_rows, ignore_index=True).to_csv(output_dir / "sensitivity_ph_checks.csv", index=False)
    pd.DataFrame(warning_rows, columns=["specification", "fold", "category", "message"]).to_csv(output_dir / "sensitivity_fit_warnings.csv", index=False)
    print("PASS: all three specifications completed all five folds with unchanged patient and event counts.", flush=True)
