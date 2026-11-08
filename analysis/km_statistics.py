"""Reproduce the final KM table statistics from the native-R patient groups."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from lifelines.statistics import proportional_hazard_test
from lifelines import CoxPHFitter

GROUPS = ["Lower risk", "Higher risk"]

def curves_and_logrank(patients):
    rows = []
    for group in GROUPS:
        frame = patients.loc[patients.risk_group.eq(group)]
        t, e = frame.time.to_numpy(), frame.event.to_numpy(dtype=bool)
        probability, greenwood = 1.0, 0.0
        rows.append((0.0, 1.0, 1.0, 1.0, group))
        for when in np.unique(t[e]):
            n, d = np.count_nonzero(t >= when), np.count_nonzero((t == when) & e)
            probability *= 1 - d / n
            if n > d:
                greenwood += d / (n * (n - d))
            lower = upper = probability
            if 0 < probability < 1:
                center = np.log(-np.log(probability))
                se = np.sqrt(greenwood) / abs(np.log(probability))
                lower, upper = np.exp(-np.exp(center + 1.96 * se)), np.exp(-np.exp(center - 1.96 * se))
            rows.append((when / 365.25, probability, lower, upper, group))
    curve = pd.DataFrame(rows, columns=["time_years", "survival", "lower", "upper", "risk_group"])
    oe, variance = 0.0, 0.0
    for _, frame in patients.groupby("cancer_type"):
        t, e, high = frame.time.to_numpy(), frame.event.to_numpy(dtype=bool), frame.risk_group.eq(GROUPS[1]).to_numpy()
        for when in np.unique(t[e]):
            at, died = t >= when, (t == when) & e
            nh, nl = np.count_nonzero(at & high), np.count_nonzero(at & ~high)
            dh, d = np.count_nonzero(died & high), np.count_nonzero(died)
            n = nh + nl
            if n > 1 and d > 0:
                oe += dh - d * nh / n
                variance += nh * nl * d * (n - d) / (n**2 * (n - 1))
    statistic = oe**2 / variance
    return curve, statistic, chi2.sf(statistic, 1)

def cox_fit(patients):
    frame = patients[["time", "event", "cancer_type"]].copy()
    frame["high_risk"] = patients.risk_group.eq(GROUPS[1]).astype(int)
    model = CoxPHFitter().fit(frame, "time", "event", strata=["cancer_type"], fit_options={"precision": 1e-9})
    return model, frame

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    out = args.workdir / "Source_Data/Figure3"
    for panel in ("c", "d"):
        patients = pd.read_csv(out / f"Fig3{panel}_patient_groups.tsv", sep="\t").sort_values("patient").reset_index(drop=True)
        assert patients.patient.is_unique and patients.event.dtype == bool
        _, lr, lr_p = curves_and_logrank(patients)
        model, frame = cox_fit(patients)
        ph = proportional_hazard_test(model, frame, time_transform="km").summary
        row = model.summary.loc["high_risk"]
        stats = dict(contrast="Higher risk vs lower risk", hazard_ratio=row["exp(coef)"],
            hazard_ratio_lower_95=row["exp(coef) lower 95%"], hazard_ratio_upper_95=row["exp(coef) upper 95%"],
            cox_wald_p=row["p"], cox_ph_km_approx_p=ph.loc["high_risk", "p"],
            stratified_logrank_chi_square=lr, stratified_logrank_p=lr_p)
        pd.DataFrame([stats]).to_csv(out / f"Fig3{panel}_statistics.tsv", sep="\t", index=False)
        print(panel, stats)


if __name__ == "__main__":
    main()
