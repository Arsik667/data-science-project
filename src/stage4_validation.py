"""
Stage 4 - Validation (Role 4).

Compares the finished Method 2 estimates (and Method 1 / baseline for
reference) with two independent sources that were NOT used to build or tune
the model:
  A. 2025 travel to school survey  -> Car/Van share per school
  B. 2024 cut-off distances (PDF)  -> furthest admitted distance per school

Nothing here feeds back into the model. The sensitivity table at the end only
REPORTS how validation metrics move with d50 and lambda; the main settings
stay as fixed in Stage 3.

Run after school_matching.py and stage3_method2.py:
    .venv/bin/python src/stage4_validation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import stage3_method2 as s3
from school_matching import load_survey

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
MIN_RESPONSES = 30  # schools with fewer survey responses are reported but flagged


def metrics(pred, obs) -> dict:
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    e = pred - obs
    varies = np.ptp(pred) > 1e-12  # correlations are undefined for a constant prediction
    slope = stats.linregress(pred, obs).slope if varies else np.nan
    return {"n": len(obs), "pearson": stats.pearsonr(pred, obs)[0] if varies else np.nan,
            "spearman": stats.spearmanr(pred, obs)[0] if varies else np.nan,
            "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()), "bias": e.mean(),
            "sd_pred": pred.std(), "sd_obs": obs.std(), "slope_obs_on_pred": slope}


def school_level_models(pc, D, P, sc):
    """Predicted car share per school for each model variant."""
    kids, car = pc["est_kids"].to_numpy(), pc["car_rate"].to_numpy()
    pupils = kids @ P
    out = pd.DataFrame({"EstablishmentNumber": sc["EstablishmentNumber"]})
    _, ij = s3.car_kids(P, D, car, kids)
    out["pred_share_M2"] = ij.sum(0) / pupils
    _, ij = s3.car_kids(P, D, car, kids, baseline=True)
    out["pred_share_M2_baseline"] = ij.sum(0) / pupils
    # Method 1: nearest school, same g(d) as the main model, for a like-for-like view
    P1 = np.zeros_like(D)
    P1[np.arange(len(D)), D.argmin(1)] = 1
    _, ij = s3.car_kids(P1, D, car, kids)
    out["pred_share_M1"] = ij.sum(0) / np.maximum(kids @ P1, 1e-9)
    return out


def main():
    pc, sc0, D = s3.load_inputs()
    sc = s3.prepare_schools(sc0)
    P = np.load(OUT / "attend_prob.npy")
    m2 = pd.read_csv(OUT / "method2_school.csv")
    key = pd.read_csv(OUT / "school_key.csv")
    cut = pd.read_csv(OUT / "cutoff_parsed.csv")
    sv = load_survey()

    # ---------------------------------------------------------- assemble
    v = key.merge(m2.drop(columns=["EstablishmentName"]), on="EstablishmentNumber")
    v = v.merge(school_level_models(pc, D, P, sc), on="EstablishmentNumber")
    svc = sv.rename(columns={"School Name": "name_survey", "Grand Total": "survey_total",
                             "Car/Van": "survey_car",
                             "Car/Van Share (with child/Children from a different household)": "survey_car_share_lift"})
    svc[["survey_car", "survey_car_share_lift", "Taxi"]] = svc[["survey_car", "survey_car_share_lift", "Taxi"]].fillna(0)
    v = v.merge(svc[["name_survey", "survey_total", "survey_car", "survey_car_share_lift", "Taxi"]],
                on="name_survey", how="left")
    v["obs_share"] = v["survey_car"] / v["survey_total"]
    v["obs_share_incl_lift"] = (v["survey_car"] + v["survey_car_share_lift"]) / v["survey_total"]
    v["response_rate"] = v["survey_total"] / v["roll_primary"]
    v = v.merge(cut[["dfe", "cutoff_flag", "cutoff_m", "last_criteria", "criteria_desc"]],
                left_on="pdf_dfe", right_on="dfe", how="left").drop(columns="dfe")
    v["in_survey_comparison"] = v["primary"] & v["survey_total"].notna()
    v["low_responses"] = v["survey_total"] < MIN_RESPONSES
    v["error_M2"] = v["pred_share_M2"] - v["obs_share"]

    # ---------------------------------------------------------- A. survey
    s = v[v["in_survey_comparison"]]
    print("=== A. Survey comparison")
    print(f"schools: listed {len(v)} | linked to survey {v.survey_total.notna().sum()} | "
          f"primary & linked (compared) {len(s)} | all-through excluded "
          f"{(v.survey_total.notna() & ~v.primary).sum()} | <{MIN_RESPONSES} responses {s.low_responses.sum()}")
    print("response rate (survey pupils / primary roll): median %.2f, p10 %.2f, p90 %.2f"
          % tuple(s["response_rate"].quantile([.5, .1, .9])))
    print(f"pooled observed car/van share over compared schools: "
          f"{s.survey_car.sum() / s.survey_total.sum():.3f} (incl. car-share lifts "
          f"{(s.survey_car + s.survey_car_share_lift).sum() / s.survey_total.sum():.3f})")
    w = s["survey_total"]
    print(f"pooled predicted share on the same schools, weighted by survey pupils: "
          f"M2 {np.average(s.pred_share_M2, weights=w):.3f} | baseline {np.average(s.pred_share_M2_baseline, weights=w):.3f}")

    rows = []
    for subset, d in [("all compared", s), (f">= {MIN_RESPONSES} responses", s[~s.low_responses])]:
        for model in ["pred_share_M2", "pred_share_M2_baseline", "pred_share_M1"]:
            rows.append({"subset": subset, "model": model, **metrics(d[model], d["obs_share"])})
        const = np.full(len(d), d["obs_share"].mean())  # reference only: knows the true mean
        rows.append({"subset": subset, "model": "constant = observed mean (reference)",
                     **metrics(const, d["obs_share"])})
    mt = pd.DataFrame(rows)
    mt.to_csv(OUT / "validation_metrics_survey.csv", index=False)
    print(mt.round(3).to_string(index=False))

    # counts: raw and scaled to the number of survey respondents
    cnt = metrics(s["pred_car_pupils"], s["survey_car"])
    scaled = metrics(s["pred_share_M2"] * s["survey_total"], s["survey_car"])
    print("\ncounts, M2 predicted car pupils vs survey Car/Van count: pearson %.3f, spearman %.3f, MAE %.1f, bias %.1f"
          % (cnt["pearson"], cnt["spearman"], cnt["MAE"], cnt["bias"]))
    print("counts, M2 share x survey respondents vs Car/Van count:  pearson %.3f, spearman %.3f, MAE %.1f, bias %.1f"
          % (scaled["pearson"], scaled["spearman"], scaled["MAE"], scaled["bias"]))
    print("alt. definition incl. car-share lifts: M2 pearson %.3f, spearman %.3f, bias %.3f"
          % tuple(metrics(s.pred_share_M2, s.obs_share_incl_lift)[k] for k in ("pearson", "spearman", "bias")))

    print("\nlargest under-predictions (obs >> pred):")
    print(s.nsmallest(5, "error_M2")[["name_school_list", "survey_total", "obs_share", "pred_share_M2"]]
          .round(3).to_string(index=False))
    print("largest over-predictions (pred >> obs):")
    print(s.nlargest(5, "error_M2")[["name_school_list", "survey_total", "obs_share", "pred_share_M2"]]
          .round(3).to_string(index=False))

    # does the error relate to school features we did NOT model?
    s = s.assign(faith=s["name_school_list"].str.contains(r"Catholic|CofE|Church of England|St\.? |Saint|RC|Holy",
                                                          regex=True))
    g = s.groupby("faith")[["obs_share", "pred_share_M2", "error_M2"]].mean()
    print("\nmean by faith-named school (name contains Catholic/CofE/St/Holy...):")
    print(g.round(3).to_string())
    print("faith vs non-faith error difference, Mann-Whitney p = %.4f"
          % stats.mannwhitneyu(s.loc[s.faith, "error_M2"], s.loc[~s.faith, "error_M2"]).pvalue)

    # ---------------------------------------------------------- B. cut-offs
    c = v[v["primary"] & v["cutoff_m"].notna()]
    print("\n=== B. Cut-off distance comparison")
    print(f"primary schools with a numeric cut-off: {len(c)} (non-standard rows: "
          f"{(c.cutoff_flag == 'nonstandard').sum()})")
    rows = []
    for subset, d in [("all numeric", c), ("standard rows only", c[c.cutoff_flag == "numeric"])]:
        for col in ["catchment_p80_m", "catchment_p95_m"]:
            ratio = d[col] / d["cutoff_m"]
            rows.append({"subset": subset, "implied": col, "n": len(d),
                         "spearman": stats.spearmanr(d[col], d["cutoff_m"])[0],
                         "pearson_log": stats.pearsonr(np.log(d[col]), np.log(d["cutoff_m"]))[0],
                         "median_cutoff_m": d["cutoff_m"].median(), "median_implied_m": d[col].median(),
                         "median_ratio": ratio.median(), "share_ratio_0.5_to_2": ratio.between(.5, 2).mean()})
    ct = pd.DataFrame(rows)
    ct.to_csv(OUT / "validation_metrics_cutoff.csv", index=False)
    print(ct.round(3).to_string(index=False))

    # oversubscription: schools with a numeric cut-off turned applicants away;
    # does the model see them as fuller (B < 1 or intake near capacity)?
    pr = v[v["primary"] & v["cutoff_flag"].notna()].copy()
    pr["oversubscribed"] = pr["cutoff_flag"] != "all_admitted"
    pr["model_capped"] = pr["balancing_B"] < 0.999
    pr["model_fullness"] = pr["pred_pupils"] / pr["cap_effective"]
    tab = pd.crosstab(pr["oversubscribed"], pr["model_capped"], margins=True)
    print("\noversubscribed in 2024 (rows) vs capped by the model (cols):")
    print(tab.to_string())
    auc = stats.mannwhitneyu(pr.loc[pr.oversubscribed, "model_fullness"],
                             pr.loc[~pr.oversubscribed, "model_fullness"]).statistic \
        / (pr.oversubscribed.sum() * (~pr.oversubscribed).sum())
    print("model fullness median: oversubscribed %.2f vs all-admitted %.2f | AUC %.3f"
          % (pr.loc[pr.oversubscribed, "model_fullness"].median(),
             pr.loc[~pr.oversubscribed, "model_fullness"].median(), auc))

    # ---------------------------------------------------------- sensitivity (report only)
    sens = []
    kids, car = pc["est_kids"].to_numpy(), pc["car_rate"].to_numpy()
    idx = sc.reset_index().set_index("EstablishmentNumber").loc[s["EstablishmentNumber"], "index"].to_numpy()
    for lam_k in (0.5, 1.0, 2.0):
        Pk, _, _ = s3.attend_prob(s3.decay(D, "exp", 650 * lam_k), sc["cap_primary"].to_numpy(), kids,
                                  cap=sc["cap_effective"].to_numpy())
        pupils = kids @ Pk
        for d50 in (500, 1000, 2000):
            _, ij = s3.car_kids(Pk, D, car, kids, d50=d50)
            share = (ij.sum(0) / pupils)[idx]
            m = metrics(share, s["obs_share"])
            sens.append({"lambda_m": 650 * lam_k, "d50_m": d50, **{k: m[k] for k in
                         ("pearson", "spearman", "MAE", "bias", "sd_pred")}})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "validation_sensitivity.csv", index=False)
    print("\n=== Sensitivity of survey metrics (REPORT ONLY - main settings are not changed)")
    print(sens.round(3).to_string(index=False))

    # ---------------------------------------------------------- outputs
    cols = ["EstablishmentNumber", "name_school_list", "name_survey", "name_pdf", "primary",
            "survey_match_method", "survey_total", "survey_car", "obs_share", "obs_share_incl_lift",
            "response_rate", "low_responses", "pred_pupils", "pred_car_pupils", "pred_share_M2",
            "pred_share_M2_baseline", "pred_share_M1", "error_M2", "cutoff_flag", "cutoff_m",
            "catchment_p80_m", "catchment_p95_m", "balancing_B", "in_survey_comparison"]
    v[cols].to_csv(OUT / "school_validation.csv", index=False)
    figures(s, c)
    print("\nwrote school_validation.csv, validation_metrics_*.csv, validation_sensitivity.csv, figures")


def figures(s, c):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, col, title in [(ax[0], "pred_share_M2", "Method 2 (main)"),
                          (ax[1], "pred_share_M2_baseline", "Method 2 baseline (g = 1)")]:
        a.scatter(s[col], s["obs_share"], s=np.clip(s["survey_total"] / 4, 5, 80), alpha=.6,
                  c=np.where(s["low_responses"], "grey", "#4C72B0"))
        lim = [0, max(s[col].max(), s["obs_share"].max()) * 1.05]
        a.plot(lim, lim, "k--", lw=1, label="y = x")
        r = stats.spearmanr(s[col], s["obs_share"])[0]
        a.set(title=f"{title}\nSpearman = {r:.2f}, n = {len(s)}", xlabel="predicted car share",
              ylabel="survey Car/Van share", xlim=lim, ylim=lim)
        a.legend(loc="upper left")
    ax[2].scatter(s["pred_car_pupils"], s["survey_car"], s=12, alpha=.6)
    lim = [0, max(s["pred_car_pupils"].max(), s["survey_car"].max()) * 1.05]
    ax[2].plot(lim, lim, "k--", lw=1)
    ax[2].set(title="Counts: predicted car pupils vs survey Car/Van", xlabel="predicted (M2)",
              ylabel="survey Car/Van count", xlim=lim, ylim=lim)
    fig.tight_layout()
    fig.savefig(FIG / "stage4_survey_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for flag, mk in [("numeric", "o"), ("nonstandard", "x")]:
        d = c[c.cutoff_flag == flag]
        ax.scatter(d["cutoff_m"], d["catchment_p80_m"], marker=mk, label=f"{flag} (n={len(d)})")
    lim = [100, max(c["cutoff_m"].max(), c["catchment_p80_m"].max()) * 1.2]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="2024 cut-off distance (m)",
           ylabel="Method 2 implied catchment, 80% of pupils (m)",
           title="Implied catchment vs admission cut-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage4_cutoff_scatter.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
