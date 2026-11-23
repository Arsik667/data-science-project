"""
Stage 5 - Fairness checks (Role 5).

Religion (and every other protected characteristic) is NOT a model input and
is never used to change the estimates. It is used here only to AUDIT them.

What can be checked without religion-by-OA data:
  A. School level: is the Method 2 error (predicted - survey car share)
     different for faith schools? Denomination is read from the school NAME,
     because the ReligiousEthos field of school locations.csv is almost empty
     (204 of 225 schools say "Does not apply", including all Catholic ones).
     Bootstrap 95% CIs (seed 42) and a regression that controls for the
     catchment car-access rate (so the gap is not just "richer areas").
  B. Socio-economic check: error by quintile of catchment car-access rate
     (car access is a rough proxy for household income).
  C. Postcode level: where would the school-level errors land? For each
     postcode, propagated_error = attendance-weighted mean error of the
     surveyed schools its children are predicted to attend. Diagnostic only:
     it is NOT used to correct the estimates (that would be tuning on the
     validation survey).

What needs religion-by-OA data (NOT RUN - file not provided):
  D. religion_check(): correlate each OA's religion shares with the estimated
     car rate and with the propagated error. Written and tested on synthetic
     data; runs automatically if "religion by oa.csv" is put in data/raw/.

Run:  .venv/bin/python src/stage5_fairness.py | tee outputs/stage5_log.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import stage3_method2 as s3

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"  # original input files (names kept as supplied)
OUT, FIG = ROOT / "outputs", ROOT / "figures"
SEED = 42
N_BOOT = 2000
RELIGION_FILE = RAW / "religion by oa.csv"
MIN_COVERAGE = 0.5  # postcode needs >= 50% of its attendance at surveyed schools


# ----------------------------------------------------------------- helpers
def denomination(name: str) -> str:
    """Faith designation from the school name (the ethos field is unreliable)."""
    if pd.Series([name]).str.contains(r"Catholic|\bRC\b", regex=True)[0]:
        return "Catholic"
    if pd.Series([name]).str.contains(r"Church of England|\bCofE\b|\bC of E\b", regex=True)[0]:
        return "Church of England"
    return "No religious designation"


def boot_mean_ci(x, w=None, n=N_BOOT, seed=SEED):
    """Mean and percentile 95% CI; resamples schools with replacement."""
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    means = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return np.average(x, weights=w), *np.percentile(means, [2.5, 97.5])


def ols(y, X):
    """Least squares with an intercept; returns coefficients."""
    X1 = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def boot_ols(y, X, n=N_BOOT, seed=SEED):
    y, X = np.asarray(y, float), np.asarray(X, float)
    rng = np.random.default_rng(seed)
    b = np.array([ols(y[i], X[i]) for i in rng.integers(0, len(y), (n, len(y)))])
    return ols(y, X), np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


def catchment_car_rate(P, kids, car_rate):
    """Children-weighted car-access rate of each school's predicted pupils."""
    W = P * kids[:, None]
    return (W * car_rate[:, None]).sum(0) / W.sum(0)


def propagated_error(P, err_by_school, min_coverage=MIN_COVERAGE):
    """Attendance-weighted mean school error per postcode (NaN at unsurveyed schools)."""
    has = ~np.isnan(err_by_school)
    cover = P[:, has].sum(1)
    e = P[:, has] @ err_by_school[has] / np.where(cover > 0, cover, np.nan)
    e[cover < min_coverage] = np.nan
    return e, cover


# ----------------------------------------------------------------- D. religion (needs data)
def religion_check(rel: pd.DataFrame, pcf: pd.DataFrame) -> pd.DataFrame:
    """
    rel : religion counts by OA (e.g. Census 2021 TS030). The OA column is
          detected as the one whose values match pcf["OA"]; every other
          numeric column is treated as a religion category (persons).
    pcf : postcode table with OA, est_kids, est_car_kids, propagated_error.
    For each category returns:
      - children-weighted mean estimated car rate, assuming the OA's children
        share the OA's religious mix (ecological assumption);
      - ratio to the all-children mean (disparity ratio, 1 = parity);
      - children-weighted Spearman, across OAs, between the category's share
        and (a) estimated car rate, (b) propagated error.
    """
    oa_col = max(rel.columns, key=lambda c: rel[c].astype(str).isin(pcf["OA"]).mean())
    assert rel[oa_col].astype(str).isin(pcf["OA"]).mean() > 0.5, "no OA code column found"
    cats = [c for c in rel.columns if c != oa_col and pd.api.types.is_numeric_dtype(rel[c])]
    cats = [c for c in cats if not str(c).lower().startswith(("total", "all usual"))]
    r = rel.rename(columns={oa_col: "OA"}).assign(OA=lambda d: d["OA"].astype(str))
    r = r.groupby("OA")[cats].sum()
    shares = r.div(r.sum(axis=1), axis=0)

    oa = pcf.groupby("OA").agg(kids=("est_kids", "sum"), car_kids=("est_car_kids", "sum"),
                               perr=("propagated_error",
                                     lambda e: np.average(e.dropna(), weights=pcf.loc[e.dropna().index, "est_kids"])
                                     if e.notna().any() and pcf.loc[e.dropna().index, "est_kids"].sum() > 0
                                     else np.nan))
    oa["car_rate_est"] = oa["car_kids"] / oa["kids"]
    oa = oa.join(shares, how="inner")
    overall = oa["car_kids"].sum() / oa["kids"].sum()
    rows = []
    for c in cats:
        w = oa["kids"] * oa[c]                      # children of this group (ecological)
        ok = oa["perr"].notna()
        rows.append({
            "group": c,
            "children_est": w.sum(),
            "mean_est_car_rate": np.average(oa["car_rate_est"], weights=w) if w.sum() > 0 else np.nan,
            "disparity_ratio": np.average(oa["car_rate_est"], weights=w) / overall if w.sum() > 0 else np.nan,
            "spearman_share_vs_car_rate": stats.spearmanr(oa[c], oa["car_rate_est"])[0],
            "spearman_share_vs_propagated_error": stats.spearmanr(oa.loc[ok, c], oa.loc[ok, "perr"])[0],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main
def main():
    pc, sc0, D = s3.load_inputs()
    sc = s3.prepare_schools(sc0)
    P = np.load(OUT / "attend_prob.npy")
    est = pd.read_csv(OUT / "estimates_postcode.csv", dtype={"POSTCODE": str})
    v = pd.read_csv(OUT / "school_validation.csv")
    assert (est["POSTCODE"] == pc["POSTCODE"]).all()
    assert (v["EstablishmentNumber"].to_numpy() == sc["EstablishmentNumber"].to_numpy()).all()

    kids, car_rate = pc["est_kids"].to_numpy(), pc["car_rate"].to_numpy()
    v["denomination"] = v["name_school_list"].map(denomination)
    v["ethos_field"] = sc["ReligiousEthos"].fillna("missing").to_numpy()
    v["catchment_car_rate"] = catchment_car_rate(P, kids, car_rate)

    print("=== Data check: ReligiousEthos field vs school name")
    print(pd.crosstab(v["denomination"], v["ethos_field"]).to_string())
    print("-> the field marks 0 of the Catholic schools; denomination is taken from the name.\n")

    # ---------------------------------------------------------- A. faith schools
    s = v[v["in_survey_comparison"]].copy()
    print(f"=== A. Error by denomination (survey-compared primary schools, n = {len(s)})")
    print("error = predicted - observed car/van share; negative = model UNDER-estimates car use")
    rows = []
    for g, d in s.groupby("denomination"):
        m, lo, hi = boot_mean_ci(d["error_M2"])
        mw, lw, hw = boot_mean_ci(d["error_M2"], d["survey_total"])
        rows.append({"group": g, "n_schools": len(d), "survey_pupils": int(d["survey_total"].sum()),
                     "obs_share": d["obs_share"].mean(), "pred_share": d["pred_share_M2"].mean(),
                     "mean_error": m, "ci_low": lo, "ci_high": hi,
                     "mean_error_pupil_weighted": mw, "ci_low_w": lw, "ci_high_w": hw,
                     "share_underpredicted_by_10pp": (d["error_M2"] < -0.10).mean()})
    grp = pd.DataFrame(rows)
    print(grp.round(3).to_string(index=False))
    kw = stats.kruskal(*[d["error_M2"] for _, d in s.groupby("denomination")])
    mw_cath = stats.mannwhitneyu(s.loc[s.denomination == "Catholic", "error_M2"],
                                 s.loc[s.denomination == "No religious designation", "error_M2"])
    mw_ce = stats.mannwhitneyu(s.loc[s.denomination == "Church of England", "error_M2"],
                               s.loc[s.denomination == "No religious designation", "error_M2"])
    print(f"Kruskal-Wallis across 3 groups p = {kw.pvalue:.4f}; Mann-Whitney vs no designation: "
          f"Catholic p = {mw_cath.pvalue:.4f}, CofE p = {mw_ce.pvalue:.4f}")

    # does the Catholic gap survive controlling for catchment car access?
    X = np.column_stack([(s.denomination == "Catholic").astype(float),
                         (s.denomination == "Church of England").astype(float),
                         s["catchment_car_rate"]])
    b, lo, hi = boot_ols(s["error_M2"].to_numpy(), X)
    names = ["intercept", "Catholic", "Church of England", "catchment car rate"]
    reg = pd.DataFrame({"term": names, "coef": b, "ci_low": lo, "ci_high": hi})
    print("\nOLS error ~ denomination + catchment car rate (bootstrap 95% CI, seed 42):")
    print(reg.round(3).to_string(index=False))

    # ---------------------------------------------------------- B. socio-economic
    s["car_rate_quintile"] = pd.qcut(s["catchment_car_rate"], 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"])
    print("\n=== B. Error by quintile of catchment car-access rate (proxy for income)")
    q = s.groupby("car_rate_quintile", observed=True).agg(
        n=("error_M2", "size"), catchment_car_rate=("catchment_car_rate", "mean"),
        obs_share=("obs_share", "mean"), pred_share=("pred_share_M2", "mean"),
        mean_error=("error_M2", "mean"), mean_abs_error=("error_M2", lambda e: e.abs().mean()))
    print(q.round(3).to_string())
    rho = stats.spearmanr(s["catchment_car_rate"], s["error_M2"])
    print(f"Spearman(catchment car rate, error) = {rho[0]:.3f}, p = {rho[1]:.4f}")

    # ---------------------------------------------------------- C. postcode level
    err = v["error_M2"].where(v["in_survey_comparison"]).to_numpy(float)
    perr, cover = propagated_error(P, err)
    cath = v["denomination"].eq("Catholic").to_numpy()
    ce = v["denomination"].eq("Church of England").to_numpy()
    pcf = pd.DataFrame({"POSTCODE": pc["POSTCODE"], "OA": pc["OA"], "X": pc["X"], "Y": pc["Y"],
                        "est_kids": kids, "est_car_kids": est["est_car_kids"],
                        "est_car_rate": est["est_car_rate"], "car_rate": car_rate,
                        "catholic_exposure": P[:, cath].sum(1), "cofe_exposure": P[:, ce].sum(1),
                        "survey_coverage": cover, "propagated_error": perr})
    ok = pcf["propagated_error"].notna()
    wk = pcf.loc[ok, "est_kids"]
    print("\n=== C. School errors propagated to postcodes (diagnostic only, NOT a correction)")
    print(f"postcodes with >= {MIN_COVERAGE:.0%} of attendance at surveyed schools: {ok.sum()} of {len(pcf)} "
          f"({wk.sum() / kids.sum():.1%} of children)")
    print("children-weighted propagated error: mean %.3f, p10 %.3f, p50 %.3f, p90 %.3f"
          % (np.average(pcf.loc[ok, "propagated_error"], weights=wk),
             *[s3.weighted_quantile(pcf.loc[ok, "propagated_error"].to_numpy(), wk.to_numpy(), qq) for qq in (.1, .5, .9)]))
    for t in (0.05, 0.10):
        print(f"children in postcodes with |propagated error| > {t:.2f}: "
              f"{wk[pcf.loc[ok, 'propagated_error'].abs() > t].sum():,.0f} ({wk[pcf.loc[ok, 'propagated_error'].abs() > t].sum() / wk.sum():.1%})")
    r1 = stats.spearmanr(pcf.loc[ok, "catholic_exposure"], pcf.loc[ok, "propagated_error"])[0]
    r2 = stats.spearmanr(pcf.loc[ok, "car_rate"], pcf.loc[ok, "propagated_error"])[0]
    print(f"Spearman across postcodes: Catholic-school exposure vs propagated error {r1:.3f}; "
          f"car_rate vs propagated error {r2:.3f}")
    print("caveat: Catholic schools draw pupils from far away (the survey shows high car use), but the model\n"
          "sends their places to NEARBY postcodes, so the error lands near the school, not where the real\n"
          "pupils live. The map shows where estimates are least reliable, not who is mis-priced.")

    # ---------------------------------------------------------- D. religion
    print("\n=== D. Religion by OA")
    if RELIGION_FILE.exists():
        rc = religion_check(pd.read_csv(RELIGION_FILE), pcf)
        rc.to_csv(OUT / "fairness_religion.csv", index=False)
        print(rc.round(3).to_string(index=False))
    else:
        print(f"NOT RUN: '{RELIGION_FILE.name}' not provided. Method in outputs/fairness_method.md; "
              "religion_check() is tested on synthetic data (tests/test_stage5.py).")

    # ---------------------------------------------------------- save
    cols = ["EstablishmentNumber", "name_school_list", "denomination", "ethos_field", "in_survey_comparison",
            "survey_total", "obs_share", "pred_share_M2", "error_M2", "catchment_car_rate"]
    v[cols].to_csv(OUT / "fairness_school.csv", index=False)
    grp.to_csv(OUT / "fairness_groups.csv", index=False)
    reg.to_csv(OUT / "fairness_regression.csv", index=False)
    q.reset_index().to_csv(OUT / "fairness_car_rate_quintiles.csv", index=False)
    pcf.drop(columns=["X", "Y"]).to_csv(OUT / "fairness_postcode.csv", index=False)
    figures(s, pcf, grp)
    print("\nwrote outputs/fairness_*.csv, figures/stage5_*.png")


def figures(s, pcf, grp):
    order = ["No religious designation", "Church of England", "Catholic"]
    colour = {order[0]: "#4C72B0", order[1]: "#55A868", order[2]: "#C44E52"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    rng = np.random.default_rng(SEED)
    for k, g in enumerate(order):
        d = s[s.denomination == g]
        ax[0].scatter(k + rng.uniform(-.15, .15, len(d)), d["error_M2"], s=14, alpha=.5, c=colour[g])
        r = grp.set_index("group").loc[g]
        ax[0].errorbar(k + .3, r.mean_error, yerr=[[r.mean_error - r.ci_low], [r.ci_high - r.mean_error]],
                       fmt="o", color="k", capsize=4)
    ax[0].axhline(0, color="grey", lw=1, ls="--")
    ax[0].set_xticks(range(3), [f"{g}\n(n={(s.denomination == g).sum()})" for g in order])
    ax[0].set(ylabel="predicted - survey car/van share", title="Method 2 error by school designation\n(black: mean, bootstrap 95% CI)")
    ax[1].scatter(s["catchment_car_rate"], s["error_M2"], s=14, alpha=.5,
                  c=s["denomination"].map(colour))
    ax[1].axhline(0, color="grey", lw=1, ls="--")
    ax[1].set(xlabel="catchment car-access rate", ylabel="error", title="Error vs catchment car access\n(red = Catholic, green = CofE)")
    fig.tight_layout()
    fig.savefig(FIG / "stage5_error_by_group.png", dpi=150)
    plt.close(fig)

    d = pcf[pcf.propagated_error.notna()]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pcf.loc[pcf.propagated_error.isna(), "X"], pcf.loc[pcf.propagated_error.isna(), "Y"],
               s=1, c="lightgrey", label="< 50% attendance at surveyed schools")
    sc_ = ax.scatter(d["X"], d["Y"], s=1.5, c=d["propagated_error"], cmap="RdBu", vmin=-.3, vmax=.3)
    fig.colorbar(sc_, label="propagated error (red = model under-estimates car use, blue = over-estimates)")
    ax.set(aspect="equal", xticks=[], yticks=[], title="Where school-level errors would land (diagnostic)")
    ax.legend(loc="lower left", markerscale=6)
    fig.tight_layout()
    fig.savefig(FIG / "stage5_propagated_error_map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
