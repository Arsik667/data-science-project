"""
Stage 6 - Sensitivity and uncertainty.

Stage 3 already showed how the CITYWIDE total moves when one assumption
changes. For pricing, the question that matters more is whether the
POSTCODE estimates and their ranking stay stable. This script answers it in
two ways.

A. One-at-a-time scenarios: each assumption is changed on its own. For each
   scenario it reports:
     - total car-driven children and their share;
     - Spearman rank correlation of postcode est_car_rate with the main model;
     - top-decile overlap: the share of the main model's 10% highest-risk
       postcodes that stay in the top 10%;
     - median absolute change in postcode est_car_rate;
     - Spearman with the survey at school level. This is REPORT ONLY and is
       never used to choose a scenario (survey = validation data).

B. Monte Carlo: the three unidentified parameters are drawn jointly. This
   gives a 90% band for the total and for every postcode. Seed 42.
     lambda  log-uniform on [325, 1300] m  (main 650 x 0.5 ... x 2)
     d50     uniform on [500, 2000] m      (main 1000)
     scale   uniform on [150, 400] m       (main 250)
   The ranges are judgement calls, recorded in the assumptions log. They are
   not derived from data.

Run:  .venv/bin/python src/stage6_sensitivity.py | tee outputs/stage6_log.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import stage1_data_prep as s1
import stage3_method2 as s3

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
SEED = 42
N_MC = 200
MAIN = dict(form="exp", p=650.0, d50=s3.G_D50, scale=s3.G_SCALE, detour=1.0,
            cap_weight=True, cap=True, all_through=s3.ALL_THROUGH_SHARE, ages="main")
MC_RANGES = {"lambda": (325.0, 1300.0), "d50": (500.0, 2000.0), "scale": (150.0, 400.0)}


# ----------------------------------------------------------------- model run
def run_model(D, pc, sc0, kids_flat, **kw):
    """Method 2 under settings kw (missing keys = main). Returns a dict of arrays."""
    c = {**MAIN, **kw}
    sc = s3.prepare_schools(sc0, c["all_through"])
    kids = pc["est_kids"].to_numpy() if c["ages"] == "main" else kids_flat
    Dd = D * c["detour"]                                   # route length = straight line x detour
    A = sc["cap_primary"].to_numpy() if c["cap_weight"] else np.ones(len(sc))
    P, _, _ = s3.attend_prob(s3.decay(Dd, c["form"], c["p"]), A, kids,
                             cap=sc["cap_effective"].to_numpy() if c["cap"] else None)
    pcar_ij = pc["car_rate"].to_numpy()[:, None] * s3.g_drive(Dd, c["d50"], c["scale"])
    rate = (P * pcar_ij).sum(1)                            # est_car_rate per postcode
    W = P * kids[:, None]
    school_share = (W * pcar_ij).sum(0) / W.sum(0)
    return {"rate": rate, "car_kids": kids * rate, "kids": kids, "school_share": school_share}


def compare(base, r, has_kids, top, obs, surveyed):
    rate_b, rate = base["rate"][has_kids], r["rate"][has_kids]
    top_r = rate >= np.quantile(rate, 0.9)
    return {"car_kids": r["car_kids"].sum(), "car_share": r["car_kids"].sum() / r["kids"].sum(),
            "spearman_vs_main": stats.spearmanr(rate_b, rate)[0],
            "top10_overlap": (top & top_r).sum() / top.sum(),
            "median_abs_change_pp": 100 * np.median(np.abs(rate - rate_b)),
            "survey_spearman_REPORT_ONLY": stats.spearmanr(r["school_share"][surveyed], obs[surveyed])[0]}


def main():
    pc, sc0, D = s3.load_inputs()
    v = pd.read_csv(OUT / "school_validation.csv")
    assert (v["EstablishmentNumber"].to_numpy() == sc0["EstablishmentNumber"].to_numpy()).all()
    surveyed = v["in_survey_comparison"].to_numpy()
    obs = v["obs_share"].to_numpy()

    # flat age fractions (0.2 / 1 / 0.2): same equal split, different OA totals
    ages = s1.load_ages(set(pc["OA"]))
    ratio = pc["OA"].map(ages["kids_oa_flat"] / ages["kids_oa"]).to_numpy()
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)        # OAs with 0 children: 0/0
    kids_flat = pc["est_kids"].to_numpy() * ratio
    assert np.isfinite(kids_flat).all()
    print(f"flat age fractions: {kids_flat.sum():,.1f} children (main {pc['est_kids'].sum():,.1f})")

    grid = pd.read_csv(OUT / "method2_calibration_grid.csv")
    pw = grid[grid.form == "power"]
    beta = float(pw.loc[pw["rmse_all"].idxmin(), "param"])

    base = run_model(D, pc, sc0, kids_flat)
    est = pd.read_csv(OUT / "estimates_postcode.csv", dtype={"POSTCODE": str})
    assert np.allclose(base["car_kids"], est["est_car_kids"]), "main run must reproduce Stage 3"
    has_kids = base["kids"] > 0
    top = base["rate"][has_kids] >= np.quantile(base["rate"][has_kids], 0.9)

    # ---------------------------------------------------------- A. one at a time
    scen = [("main", "-", {}),
            ("decay lambda (m)", 325, {"p": 325.0}), ("decay lambda (m)", 1300, {"p": 1300.0}),
            ("decay form", f"power beta={beta:g}", {"form": "power", "p": beta}),
            ("g(d) midpoint d50 (m)", 500, {"d50": 500.0}), ("g(d) midpoint d50 (m)", 2000, {"d50": 2000.0}),
            ("g(d) steepness scale (m)", 125, {"scale": 125.0}), ("g(d) steepness scale (m)", 500, {"scale": 500.0}),
            ("route detour factor", 1.3, {"detour": 1.3}),
            ("capacity weight A_j", "off (A=1)", {"cap_weight": False}),
            ("capacity cap", "off", {"cap": False}),
            ("all-through primary share", 0.3, {"all_through": 0.3}),
            ("age fractions", "flat 0.2/1/0.2", {"ages": "flat"})]
    rows = []
    for name, val, kw in scen:
        r = run_model(D, pc, sc0, kids_flat, **kw)
        rows.append({"assumption": name, "value": val, **compare(base, r, has_kids, top, obs, surveyed)})
    oat = pd.DataFrame(rows)
    oat.to_csv(OUT / "sensitivity_one_at_a_time.csv", index=False)
    pd.set_option("display.width", 220)
    print("=== A. One assumption at a time (postcode stability vs the main model)")
    print(oat.round(3).to_string(index=False))

    # ---------------------------------------------------------- B. Monte Carlo
    rng = np.random.default_rng(SEED)
    lo, hi = np.log(MC_RANGES["lambda"])
    draws = pd.DataFrame({"lambda": np.exp(rng.uniform(lo, hi, N_MC)),
                          "d50": rng.uniform(*MC_RANGES["d50"], N_MC),
                          "scale": rng.uniform(*MC_RANGES["scale"], N_MC)})
    rates = np.empty((N_MC, len(pc)))
    for k, d in draws.iterrows():
        r = run_model(D, pc, sc0, kids_flat, p=d["lambda"], d50=d["d50"], scale=d["scale"])
        rates[k] = r["rate"]
        draws.loc[k, "car_share"] = r["car_kids"].sum() / r["kids"].sum()
        draws.loc[k, "spearman_vs_main"] = stats.spearmanr(base["rate"][has_kids], r["rate"][has_kids])[0]
    draws.to_csv(OUT / "sensitivity_mc_draws.csv", index=False)

    kids = base["kids"]
    tot = draws["car_share"] * kids.sum()
    print(f"\n=== B. Monte Carlo, {N_MC} joint draws (seed {SEED}); ranges {MC_RANGES}")
    print("citywide car-driven children: p5 %.0f, p50 %.0f, p95 %.0f  (share p5 %.1f%%, p50 %.1f%%, p95 %.1f%%)"
          % (*tot.quantile([.05, .5, .95]), *(100 * draws["car_share"].quantile([.05, .5, .95]))))
    print("postcode-rank Spearman vs main over draws: min %.3f, p5 %.3f, median %.3f"
          % (draws.spearman_vs_main.min(), *draws.spearman_vs_main.quantile([.05, .5])))
    # which parameter drives the total? rank correlation of each input with the share
    for c in MC_RANGES:
        print(f"  Spearman({c}, car_share) = {stats.spearmanr(draws[c], draws['car_share'])[0]:+.3f}")

    q5, q50, q95 = np.quantile(rates, [.05, .5, .95], axis=0)
    in_top = (rates[:, has_kids] >= np.quantile(rates[:, has_kids], 0.9, axis=1, keepdims=True))
    top_freq = np.full(len(pc), np.nan)
    top_freq[has_kids] = in_top.mean(0)
    unc = pd.DataFrame({"POSTCODE": pc["POSTCODE"], "est_kids": kids, "est_car_rate": base["rate"],
                        "car_rate_p5": q5, "car_rate_p50": q50, "car_rate_p95": q95,
                        "car_kids_p5": kids * q5, "car_kids_p95": kids * q95,
                        "band_width_pp": 100 * (q95 - q5), "share_of_draws_in_top10pct": top_freq})
    unc.to_csv(OUT / "estimates_postcode_uncertainty.csv", index=False)
    w = unc.loc[has_kids]
    print("postcode 90%% band width (p95 - p5), children-weighted: median %.1f pp, p90 %.1f pp"
          % (s3.weighted_quantile(w.band_width_pp.to_numpy(), w.est_kids.to_numpy(), .5),
             s3.weighted_quantile(w.band_width_pp.to_numpy(), w.est_kids.to_numpy(), .9)))
    stable = w["share_of_draws_in_top10pct"]
    main_top = w["est_car_rate"] >= w["est_car_rate"].quantile(0.9)
    print("main-model top-10%% postcodes that stay top-10%% in >= 90%% of draws: %.1f%%"
          % (100 * (stable[main_top] >= 0.9).mean()))
    print("postcodes never in the top 10%% of any draw: %.1f%%" % (100 * (stable == 0).mean()))
    ratio_band = (w.car_rate_p95 / w.car_rate_p5.where(w.car_rate_p5 > 0))
    print("ratio p95 / p5 of postcode car rate: median %.2f (relative uncertainty is similar everywhere)"
          % ratio_band.median())

    # ---------------------------------------------------------- C. coverage facts for the limitations table
    area = pc["POSTCODE"].str.extract(r"^([A-Z]+)")[0]
    print("\n=== C. Coverage (for limitations)")
    print("estimated children by postcode area:", pc.groupby(area)["est_kids"].sum().round(0).to_dict())
    print("children whose nearest listed school is > 2 km away: %.0f" % kids[D.min(1) > 2000].sum())

    figures(oat, draws, unc, pc, has_kids)
    print("\nwrote outputs/sensitivity_*.csv, outputs/estimates_postcode_uncertainty.csv, figures/stage6_*.png")


def figures(oat, draws, unc, pc, has_kids):
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    o = oat.iloc[1:].copy()
    o["label"] = o["assumption"] + " = " + o["value"].astype(str)
    main_share = oat.loc[0, "car_share"]
    o = o.iloc[np.argsort(-np.abs(o["car_share"] - main_share).to_numpy())][::-1]
    ax[0].barh(o["label"], 100 * (o["car_share"] - main_share), color="#4C72B0")
    ax[0].axvline(0, color="k", lw=1)
    ax[0].set(xlabel="change in citywide car share (pp)", title=f"Total: main = {100 * main_share:.1f}%")
    ax[1].barh(o["label"], o["spearman_vs_main"], color="#55A868")
    ax[1].set(xlim=(0.5, 1.0), xlabel="Spearman of postcode est_car_rate vs main",
              title="Postcode ranking stability")
    ax[1].set_yticklabels([])
    ax[2].hist(100 * draws["car_share"], bins=25, color="#8172B2")
    ax[2].axvline(100 * main_share, color="k", ls="--", label="main")
    ax[2].set(xlabel="citywide car share (%)", ylabel="draws", title=f"Monte Carlo, {len(draws)} joint draws")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage6_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    d = unc[has_kids]
    xy = pc.loc[has_kids]
    s_ = ax.scatter(xy["X"], xy["Y"], s=1.5, c=d["share_of_draws_in_top10pct"], cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(s_, label="share of Monte Carlo draws in which the postcode is top-10% risk")
    ax.set(aspect="equal", xticks=[], yticks=[], title="Is a postcode's high-risk label robust?")
    fig.tight_layout()
    fig.savefig(FIG / "stage6_top_decile_stability.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
