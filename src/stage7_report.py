"""
Stage 7 - Report draft built from code output.

Every number in the report comes from this script. The script:
  1. recomputes each number from the files in outputs/ (and, where needed,
     the raw inputs);
  2. writes outputs/report_numbers.csv (key, value, source) so that any
     number in the report can be traced back to code;
  3. fills report/report_template.md and report/presentation_template.md
     and writes report/report_draft.md, report/presentation_outline.md and README.md.

The render fails if the template uses a key that was not computed, or if a
computed key is never used.

Run (after stages 1-6):  .venv/bin/python src/stage7_report.py
"""
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from scipy import stats

import stage1_data_prep as s1

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"  # original input files (names kept as supplied)
OUT, REP = ROOT / "outputs", ROOT / "report"

NUM, SRC = {}, {}


# ----------------------------------------------------------------- formatting
def put(key, value, source, kind="n0"):
    """Store a formatted number and where it came from."""
    f = {"n0": lambda x: f"{x:,.0f}", "n1": lambda x: f"{x:,.1f}", "f2": lambda x: f"{x:.2f}",
         "f3": lambda x: f"{x:.3f}", "sf3": lambda x: f"{x:+.3f}", "pct": lambda x: f"{100 * x:.1f}%", "pp": lambda x: f"{100 * x:+.1f} pp",
         "p": lambda x: "< 0.0001" if x < 1e-4 else f"{x:.4f}", "raw": str}[kind]
    NUM[key] = f(value)
    SRC[key] = (value, source)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(m, lo, hi):
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def rd(path, **kw):
    return pd.read_csv(OUT / path, **kw)


# ----------------------------------------------------------------- collect
def collect():
    pc = rd("postcodes_master.csv", dtype={"POSTCODE": str, "OA": str})
    est = rd("estimates_postcode.csv", dtype={"POSTCODE": str})
    m1p = rd("method1_postcode.csv", dtype={"POSTCODE": str})
    m1s = rd("method1_school.csv")
    m2s = rd("method2_school.csv")
    grid = rd("method2_calibration_grid.csv")
    ladder = rd("method2_ladder.csv")
    sens3 = rd("method2_sensitivity.csv")
    key = rd("school_key.csv")
    cut = rd("cutoff_parsed.csv")
    v = rd("school_validation.csv")
    vm = rd("validation_metrics_survey.csv")
    vc = rd("validation_metrics_cutoff.csv")
    vs = rd("validation_sensitivity.csv")
    fs = rd("fairness_school.csv")
    fg = rd("fairness_groups.csv").set_index("group")
    freg = rd("fairness_regression.csv").set_index("term")
    fq = rd("fairness_car_rate_quintiles.csv").set_index("car_rate_quintile")
    fp = rd("fairness_postcode.csv", dtype={"POSTCODE": str})
    oat = rd("sensitivity_one_at_a_time.csv")
    mc = rd("sensitivity_mc_draws.csv")
    unc = rd("estimates_postcode_uncertainty.csv", dtype={"POSTCODE": str})

    # ---------------------------------------------------------- data / stage 1
    S = "postcodes_master.csv"
    kids = pc["est_kids"]
    n_pc = pc.groupby("OA")["POSTCODE"].transform("size")
    put("n_postcodes", len(pc), S)
    put("n_oa", pc["OA"].nunique(), S)
    put("pc_per_oa_mean", pc.groupby("OA").size().mean(), S, "n1")
    put("pc_per_oa_max", pc.groupby("OA").size().max(), S)
    put("total_kids", kids.sum(), S)
    put("inflation_factor", (kids * n_pc).sum() / kids.sum(), S + " (full OA count per postcode)", "n1")
    ages = s1.load_ages(set(pc["OA"]))
    put("total_kids_flat", ages["kids_oa_flat"].sum(), "agebands by oa.csv, flat 0.2/1/0.2")
    put("car_rate_hh", np.average(pc["car_rate"], weights=pc["households"]), S, "f3")
    put("kids_car_access", (kids * pc["car_rate"]).sum(), S)
    put("kids_car_access_share", (kids * pc["car_rate"]).sum() / kids.sum(), S, "pct")
    roll = m2s["roll_primary"].sum()
    put("primary_roll", roll, "method2_school.csv roll_primary")
    put("est_over_roll", kids.sum() / roll, "postcodes_master / method2_school", "f3")
    put("n_all_through", int(m2s["all_through"].sum()), "method2_school.csv")
    put("n_roll_imputed", int(m2s["roll_imputed"].sum()), "method2_school.csv")
    area = pc["POSTCODE"].str.extract(r"^([A-Z]+)")[0]
    for a in ("LS", "WF", "BD"):
        put(f"kids_{a}", kids[area == a].sum(), S)

    # ---------------------------------------------------------- stage 2 / method 1
    S = "method1_postcode.csv"
    d = m1p["dist_nearest_m"]
    put("n_schools", len(m1s), "method1_school.csv")
    put("near_median", d.median(), S)
    put("near_p90", d.quantile(.9), S)
    put("near_max", d.max(), S)
    put("near_gt3km", int((d > 3000).sum()), S)
    put("m1_corr_roll", m1s["assigned_kids"].corr(m1s["NumberOfPupils"]), "method1_school.csv", "f3")
    put("m1_over15", int((m1s["assigned_over_capacity"] > 1.5).sum()), "method1_school.csv")
    put("m1_under05", int((m1s["assigned_over_capacity"] < 0.5).sum()), "method1_school.csv")
    assert np.isclose(m1p["est_car_kids_m1"].sum(), (kids * pc["car_rate"]).sum())  # upper bound = car access

    # ---------------------------------------------------------- stage 3 / method 2
    S = "method2_calibration_grid.csv"
    oos = {}
    for form, g in grid.groupby("form"):
        r = []
        for fit_k, test_k in [(0, 1), (1, 0)]:
            b = g.loc[g[f"rmse_fold{fit_k}"].idxmin()]
            r.append((b["param"], b[f"rmse_fold{test_k}"]))
        oos[form] = r
    put("oos_exp", np.mean([x[1] for x in oos["exp"]]), S + " (2-fold)", "n1")
    put("oos_power", np.mean([x[1] for x in oos["power"]]), S + " (2-fold)", "n1")
    put("fold_lambda_a", oos["exp"][0][0], S)
    put("fold_lambda_b", oos["exp"][1][0], S)
    ge = grid[grid.form == "exp"]
    put("lambda_main", ge.loc[ge["rmse_all"].idxmin(), "param"], S)
    gp = grid[grid.form == "power"]
    put("beta_best", gp.loc[gp["rmse_all"].idxmin(), "param"], S, "f2")
    put("n_calib", int((~m2s["all_through"] & ~m2s["roll_imputed"]).sum()), "method2_school.csv")
    S = "method2_school.csv"
    capped = m2s["balancing_B"] < 0.999
    put("n_capped", int(capped.sum()), S)
    put("min_B", m2s["balancing_B"].min(), S, "f2")
    put("min_B_school", m2s.loc[m2s["balancing_B"].idxmin(), "EstablishmentName"], S, "raw")
    put("catch_p80_median", m2s["catchment_p80_m"].median(), S)
    S = "method2_ladder.csv"
    main = ladder.iloc[-1]
    put("ladder_corr_main", main["corr_with_roll"], S, "f2")
    put("ladder_corr_m1", ladder.iloc[0]["corr_with_roll"], S, "f2")
    import stage3_method2 as s3
    put("d50_main", s3.G_D50, "stage3_method2.G_D50 (set a priori)")
    put("g_scale", s3.G_SCALE, "stage3_method2.G_SCALE (set a priori)")
    put("median_trip", main["median_trip_m"], S)
    put("car_kids", est["est_car_kids"].sum(), "estimates_postcode.csv")
    put("car_share", est["est_car_kids"].sum() / kids.sum(), "estimates_postcode.csv", "pct")
    put("baseline_share", est["est_car_kids_baseline"].sum() / kids.sum(), "estimates_postcode.csv", "pct")
    rate = est.loc[est.est_kids > 0, "est_car_rate"]
    put("rate_p10", rate.quantile(.1), "estimates_postcode.csv", "pct")
    put("rate_p50", rate.quantile(.5), "estimates_postcode.csv", "pct")
    put("rate_p90", rate.quantile(.9), "estimates_postcode.csv", "pct")
    k = est.est_kids > 0
    put("rate_dist_rho", stats.spearmanr(est.loc[k, "est_car_rate"], m1p.loc[k, "dist_nearest_m"])[0],
        "estimates_postcode.csv + method1_postcode.csv", "f2")
    lt = ladder[["version", "corr_with_roll", "rmse_vs_roll", "schools_over_eff_cap", "median_trip_m",
                 "car_share_main"]].copy()
    lt = lt[~lt.version.str.contains("power")]
    lt.columns = ["Version", "Corr. with roll", "RMSE vs roll", "Schools over capacity",
                  "Median trip (m)", "Car share"]
    lt["Corr. with roll"] = lt["Corr. with roll"].map("{:.3f}".format)
    lt["RMSE vs roll"] = lt["RMSE vs roll"].map("{:.1f}".format)
    lt["Median trip (m)"] = lt["Median trip (m)"].map("{:,.0f}".format)
    lt["Car share"] = lt["Car share"].map("{:.1%}".format)
    NUM["tbl_ladder"] = md_table(lt)
    SRC["tbl_ladder"] = ("table", S)
    s3d = sens3.set_index(["assumption", "value"])["car_share"]
    put("d50_500_share", s3d[("g(d) midpoint d50 (m)", "500")], "method2_sensitivity.csv", "pct")
    put("d50_2000_share", s3d[("g(d) midpoint d50 (m)", "2000")], "method2_sensitivity.csv", "pct")

    # ---------------------------------------------------------- stage 4 matching
    S = "school_key.csv"
    mc_ = key["survey_match_method"].value_counts()
    put("match_exact", mc_.get("exact", 0), S)
    put("match_norm", mc_.get("normalised", 0), S)
    put("match_fuzzy", mc_.get("fuzzy", 0), S)
    put("match_manual_linked", int(((key.survey_match_method == "manual") & key.name_survey.notna()).sum()), S)
    put("match_linked", int(key["name_survey"].notna().sum()), S)
    put("pdf_by_number", int((key.pdf_match_method == "dfe_number").sum()), S)
    put("pdf_manual", int((key.pdf_match_method == "manual").sum()), S)
    put("pdf_rows", len(cut), "cutoff_parsed.csv")
    put("pdf_all_admitted", int((cut.cutoff_flag == "all_admitted").sum()), "cutoff_parsed.csv")
    put("pdf_numeric", int((cut.cutoff_flag == "numeric").sum()), "cutoff_parsed.csv")
    put("pdf_nonstandard", int((cut.cutoff_flag == "nonstandard").sum()), "cutoff_parsed.csv")

    # ---------------------------------------------------------- stage 4 survey
    S = "school_validation.csv"
    s = v[v["in_survey_comparison"]]
    put("n_compared", len(s), S)
    put("response_median", s["response_rate"].median(), S, "f2")
    put("obs_pooled", s.survey_car.sum() / s.survey_total.sum(), S, "f3")
    put("pred_pooled", np.average(s.pred_share_M2, weights=s.survey_total), S, "f3")
    m = vm[vm.subset == "all compared"].set_index("model")
    M2, BL, M1, C = (m.loc["pred_share_M2"], m.loc["pred_share_M2_baseline"], m.loc["pred_share_M1"],
                     m.loc["constant = observed mean (reference)"])
    S = "validation_metrics_survey.csv"
    put("sv_spearman", M2.spearman, S, "f2")
    put("sv_pearson", M2.pearson, S, "f2")
    put("sv_mae", M2.MAE, S, "f3")
    put("sv_mae_const", C.MAE, S, "f3")
    put("sv_sd_pred", M2.sd_pred, S, "f3")
    put("sv_sd_obs", M2.sd_obs, S, "f3")
    put("sv_slope", M2.slope_obs_on_pred, S, "f2")
    put("sv_bl_spearman", BL.spearman, S, "f2")
    put("sv_bl_bias", BL.bias, S, "f3")
    put("sv_m1_spearman", M1.spearman, S, "f2")
    put("sv_m1_bias", M1.bias, S, "f3")
    tv = pd.DataFrame({
        "Model": ["Method 2 (main)", "Method 2 baseline (g = 1)", "Method 1 (nearest school)",
                  "Constant = observed mean (reference)"],
        "Spearman": [f"{x.spearman:.2f}" if pd.notna(x.spearman) else "–" for x in (M2, BL, M1, C)],
        "MAE": [f"{x.MAE:.3f}" for x in (M2, BL, M1, C)],
        "Bias": [f"{x.bias:+.3f}" for x in (M2, BL, M1, C)],
        "SD predicted / observed": [f"{x.sd_pred:.3f} / {x.sd_obs:.3f}" for x in (M2, BL, M1, C)]})
    NUM["tbl_survey"] = md_table(tv)
    SRC["tbl_survey"] = ("table", S)
    sc_ = metrics_scaled(s)
    put("cnt_pearson_scaled", sc_, "school_validation.csv (pred share x respondents vs Car/Van)", "f2")
    lo = s.nsmallest(1, "error_M2").iloc[0]
    put("worst_under_school", lo["name_school_list"], S, "raw")
    put("worst_under_obs", lo["obs_share"], S, "pct")
    put("worst_under_pred", lo["pred_share_M2"], S, "pct")
    hi = s.nlargest(1, "error_M2").iloc[0]
    put("worst_over_school", hi["name_school_list"], S, "raw")
    put("worst_over_obs", hi["obs_share"], S, "pct")
    put("worst_over_pred", hi["pred_share_M2"], S, "pct")
    best = vs.loc[vs["spearman"].idxmax()]
    put("vs_best_spearman", best["spearman"], "validation_sensitivity.csv", "f2")
    put("vs_best_lambda", best["lambda_m"], "validation_sensitivity.csv")
    put("vs_best_d50", best["d50_m"], "validation_sensitivity.csv")

    # ---------------------------------------------------------- stage 4 cut-offs
    S = "validation_metrics_cutoff.csv"
    c80 = vc[(vc.subset == "all numeric") & (vc.implied == "catchment_p80_m")].iloc[0]
    put("cut_n", c80["n"], S)
    put("cut_spearman", c80["spearman"], S, "f2")
    put("cut_median", c80["median_cutoff_m"], S)
    put("cut_implied_median", c80["median_implied_m"], S)
    pr = v[v["primary"] & v["cutoff_flag"].notna()].merge(m2s[["EstablishmentNumber", "cap_effective"]])
    over = pr["cutoff_flag"] != "all_admitted"
    full = pr["pred_pupils"] / pr["cap_effective"]
    auc = stats.mannwhitneyu(full[over], full[~over]).statistic / (over.sum() * (~over).sum())
    put("auc_oversub", auc, "school_validation.csv + method2_school.csv", "f3")

    # ---------------------------------------------------------- stage 5 fairness
    S = "fairness_groups.csv"
    for g, k in [("Catholic", "cath"), ("Church of England", "cofe"), ("No religious designation", "none")]:
        r = fg.loc[g]
        put(f"f_{k}_n", r.n_schools, S)
        put(f"f_{k}_obs", r.obs_share, S, "pct")
        put(f"f_{k}_pred", r.pred_share, S, "pct")
        NUM[f"f_{k}_err"] = ci(r.mean_error, r.ci_low, r.ci_high)
        SRC[f"f_{k}_err"] = ((r.mean_error, r.ci_low, r.ci_high), S + " (bootstrap, seed 42)")
    sf = fs[fs.in_survey_comparison]
    put("f_kw_p", stats.kruskal(*[x["error_M2"] for _, x in sf.groupby("denomination")]).pvalue,
        "fairness_school.csv", "p")
    put("f_n_cath_named", int((fs.denomination == "Catholic").sum()), "fairness_school.csv")
    put("f_n_cath_field", int(((fs.denomination == "Catholic") & (fs.ethos_field != "Does not apply")
                               & (fs.ethos_field != "missing")).sum()), "fairness_school.csv")
    r = freg.loc["Catholic"]
    NUM["f_reg_cath"] = ci(r.coef, r.ci_low, r.ci_high)
    SRC["f_reg_cath"] = ((r.coef, r.ci_low, r.ci_high), "fairness_regression.csv")
    put("f_q1_err", fq.loc["Q1 lowest", "mean_error"], "fairness_car_rate_quintiles.csv", "sf3")
    put("f_q5_err", fq.loc["Q5 highest", "mean_error"], "fairness_car_rate_quintiles.csv", "sf3")
    put("f_rho_car_err", stats.spearmanr(sf["catchment_car_rate"], sf["error_M2"])[0], "fairness_school.csv", "f2")
    ok = fp["propagated_error"].notna()
    w = fp.loc[ok, "est_kids"]
    put("f_prop_gt10", w[fp.loc[ok, "propagated_error"].abs() > 0.10].sum() / w.sum(), "fairness_postcode.csv", "pct")

    # ---------------------------------------------------------- stage 6 sensitivity
    S = "sensitivity_mc_draws.csv"
    put("mc_n", len(mc), S)
    q = mc["car_share"].quantile([.05, .5, .95])
    put("mc_p5", q[.05], S, "pct")
    put("mc_p50", q[.5], S, "pct")
    put("mc_p95", q[.95], S, "pct")
    put("mc_rho_median", mc["spearman_vs_main"].median(), S, "f2")
    put("mc_rho_min", mc["spearman_vs_main"].min(), S, "f2")
    put("mc_rho_d50", stats.spearmanr(mc["d50"], mc["car_share"])[0], S, "f2")
    put("mc_rho_lambda", stats.spearmanr(mc["lambda"], mc["car_share"])[0], S, "f2")
    S = "estimates_postcode_uncertainty.csv"
    u = unc[unc.est_kids > 0]
    top = u["est_car_rate"] >= u["est_car_rate"].quantile(.9)
    put("mc_top_stable", (u.loc[top, "share_of_draws_in_top10pct"] >= 0.9).mean(), S, "pct")
    o = oat.copy()
    o["Change"] = o["assumption"] + " = " + o["value"].astype(str)
    o = o.iloc[1:]
    to = pd.DataFrame({"Change": o["Change"],
                       "Car share": o["car_share"].map("{:.1%}".format),
                       "Spearman vs main (postcodes)": o["spearman_vs_main"].map("{:.2f}".format),
                       "Top-10% overlap": o["top10_overlap"].map("{:.0%}".format)})
    NUM["tbl_oat"] = md_table(to)
    SRC["tbl_oat"] = ("table", "sensitivity_one_at_a_time.csv")

    # ---------------------------------------------------------- summary PDF sanity check
    text = " ".join(p.extract_text() for p in PdfReader(RAW / "2024-25 Annual Travel To School Survey Summary.pdf").pages)
    flat = re.sub(r"\s+", " ", text)
    m_ = re.search(r"Car/van travel is proportionally higher for primary pupils at (\d+\.\d)%", flat)
    put("pdf_primary_car", float(m_.group(1)) / 100, "Summary PDF text (pypdf): primary car/van share", "pct")
    put("pdf_mentions", len(re.findall(re.escape(m_.group(1)) + "%", flat)), "Summary PDF text (pypdf)")
    m_ = re.search(r"lower than the national average of (\d+\.\d)%", flat)
    put("pdf_national_claim", float(m_.group(1)) / 100, "Summary PDF text (pypdf): 'national average of ...'", "pct")


def metrics_scaled(s):
    return stats.pearsonr(s["pred_share_M2"] * s["survey_total"], s["survey_car"])[0]


# ----------------------------------------------------------------- render
def render(template: Path, out: Path):
    tpl = template.read_text()
    used = {f for _, f, _, _ in string.Formatter().parse(tpl) if f}
    missing = used - NUM.keys()
    assert not missing, f"template uses keys that were not computed: {missing}"
    out.write_text(tpl.format(**NUM))
    return used


def main():
    collect()
    used = render(REP / "report_template.md", REP / "report_draft.md")
    used |= render(REP / "presentation_template.md", REP / "presentation_outline.md")
    used |= render(REP / "readme_template.md", ROOT / "README.md")
    unused = NUM.keys() - used
    assert not unused, f"computed but never used: {unused}"
    rows = [{"key": k, "value_in_report": NUM[k] if not k.startswith("tbl_") else "(table)",
             "raw_value": SRC[k][0] if not k.startswith("tbl_") else "", "source": SRC[k][1]} for k in NUM]
    pd.DataFrame(rows).to_csv(OUT / "report_numbers.csv", index=False)
    words = len(re.sub(r"\|.*\|", " ", (REP / "report_draft.md").read_text()).split())
    print(f"{len(NUM)} numbers/tables -> outputs/report_numbers.csv")
    print(f"wrote report/report_draft.md (about {words:,} words outside tables) and report/presentation_outline.md")


if __name__ == "__main__":
    main()
