"""
Stage 3 - Method 2 (Role 3): probabilistic school attendance + car use.

For postcode i and school j:

    P(j | i)   = A_j * B_j * f(d_ij) / sum_k A_k * B_k * f(d_ik)      (attendance)
    P(car | i, j) = car_rate_i * g(d_ij)                                (car use)
    est_car_kids_i = est_kids_i * sum_j P(j | i) * P(car | i, j)

  f(d)  distance decay, two forms compared:
          exponential  f = exp(-d / lam)
          power        f = (d + 100 m) ** (-beta)    (100 m offset avoids f -> inf at d = 0)
  A_j   attractiveness = primary-phase capacity (refinement 1)
  B_j   capacity balancing factor in (0, 1], found iteratively so that no
        school receives more children than its effective capacity (refinement 2)
  g(d)  share of car-owning families who actually drive, rising with distance:
          baseline   g = 1                      (everyone with a car drives; upper bound)
          main       g = 1 / (1 + exp(-(d - 1000) / 250))
        d50 = 1,000 m is fixed a priori (about a 15-20 minute walk for a
        primary-age child), NOT tuned on any validation data.

Decay parameters are chosen ONLY from input data, using geography alone
(A_j = 1, no cap): the lambda/beta under which population + distance best
reproduce each school's NumberOfPupils (school locations.csv is an input,
not validation data). Capacity is added AFTERWARDS at the fixed parameter.
Why not calibrate with A_j = capacity? Capacity already ~ roll (median fill
0.95), so the fit is best when distance plays no role at all (beta -> 0,
median trip ~7 km): the criterion stops identifying the decay. A 2-fold
split of schools (seed 42) checks that the choice is stable out of sample.
The survey and cut-off PDF are not read here.

Run:  .venv/bin/python src/stage3_method2.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"  # original input files (names kept as supplied)
OUT, FIG = ROOT / "outputs", ROOT / "figures"
SEED = 42

ALL_THROUGH_SHARE = 7 / 14  # primary year groups (R-Y6) out of R-Y13
G_D50, G_SCALE = 1000.0, 250.0
POWER_OFFSET = 100.0
LAMBDAS = np.array([150, 200, 300, 400, 450, 500, 550, 600, 650, 700, 800, 1000, 1500, 2000], float)
BETAS = np.array([1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0], float)


# ----------------------------------------------------------------- inputs
def load_inputs():
    pc = pd.read_csv(OUT / "postcodes_master.csv", dtype={"POSTCODE": str, "OA": str})
    D = np.load(OUT / "dist_matrix.npy")
    cols = pd.read_csv(OUT / "dist_cols_schools.csv")
    sc = pd.read_csv(RAW / "school locations.csv")
    sc.columns = ["EstablishmentNumber", "EstablishmentName", "TypeOfEstablishment",
                  "ReligiousEthos", "SchoolCapacity", "NumberOfPupils", "X", "Y"]
    sc = sc.set_index("EstablishmentNumber").loc[cols["EstablishmentNumber"]].reset_index()
    assert D.shape == (len(pc), len(sc))
    return pc, sc, D


def prepare_schools(sc: pd.DataFrame, all_through_share=ALL_THROUGH_SHARE) -> pd.DataFrame:
    """Primary-phase capacity/roll, imputation of the 3 missing rolls."""
    sc = sc.copy()
    # DfE numbers 4000-4999 are secondary-phase establishments -> all-through here
    sc["all_through"] = sc["EstablishmentNumber"].between(4000, 4999)
    fill_median = (sc["NumberOfPupils"] / sc["SchoolCapacity"])[~sc["all_through"]].median()
    sc["roll_imputed"] = sc["NumberOfPupils"].isna()
    roll = sc["NumberOfPupils"].fillna(sc["SchoolCapacity"] * fill_median)
    share = np.where(sc["all_through"], all_through_share, 1.0)
    sc["cap_primary"] = sc["SchoolCapacity"] * share
    sc["roll_primary"] = roll * share
    # many schools are over nominal capacity, so the cap is the larger of the two
    sc["cap_effective"] = np.maximum(sc["cap_primary"], sc["roll_primary"])
    # schools whose roll is real and fully primary -> used to choose parameters
    sc["calib"] = ~sc["all_through"] & ~sc["roll_imputed"]
    sc.attrs["fill_median"] = fill_median
    return sc


# ----------------------------------------------------------------- model
def decay(D, form, p):
    if form == "exp":
        # subtracting each row's minimum does not change P (row-normalised)
        # but prevents underflow for small lambda
        return np.exp(-(D - D.min(axis=1, keepdims=True)) / p)
    if form == "power":
        return (D + POWER_OFFSET) ** (-p)
    raise ValueError(form)


def attend_prob(F, A, kids, cap=None, tol=1e-3, max_iter=1000):
    """Row-normalised attendance probabilities; optional capacity balancing.
    Returns (P, B, iterations)."""
    W = F * A[None, :]
    B = np.ones(F.shape[1])
    for it in range(1, max_iter + 1):
        Wb = W * B
        P = Wb / Wb.sum(axis=1, keepdims=True)
        if cap is None:
            return P, B, 0
        intake = kids @ P
        if (intake <= cap * (1 + tol)).all():
            return P, B, it
        # shrink over-full schools; let previously shrunk ones recover up to 1
        B = np.minimum(1.0, B * cap / intake)
    raise RuntimeError("capacity balancing did not converge")


def g_drive(D, d50=G_D50, scale=G_SCALE):
    return 1.0 / (1.0 + np.exp(-(D - d50) / scale))


def intake_rmse(intake, sc, mask):
    """RMSE between predicted intake (rescaled to the same total) and real roll."""
    roll = sc["roll_primary"].to_numpy()[mask]
    pred = intake[mask] * roll.sum() / intake[mask].sum()
    return float(np.sqrt(np.mean((pred - roll) ** 2)))


def school_fit(intake, sc):
    m = sc["calib"].to_numpy()
    return {"rmse": intake_rmse(intake, sc, m),
            "corr": float(np.corrcoef(intake[m], sc["roll_primary"].to_numpy()[m])[0, 1])}


# ----------------------------------------------------------------- stage
def calibrate(D, kids, sc, A):
    """Grid search for both decay forms + 2-fold out-of-sample check."""
    rng = np.random.default_rng(SEED)
    calib_idx = np.flatnonzero(sc["calib"])
    fold = np.zeros(len(sc), int)
    fold[rng.permutation(calib_idx)[: len(calib_idx) // 2]] = 1
    rows = []
    for form, grid in [("exp", LAMBDAS), ("power", BETAS)]:
        for p in grid:
            P, _, _ = attend_prob(decay(D, form, p), A, kids)
            intake = kids @ P
            r = {"form": form, "param": p}
            for k in (0, 1):
                r[f"rmse_fold{k}"] = intake_rmse(intake, sc, sc["calib"].to_numpy() & (fold == k))
            r["rmse_all"] = intake_rmse(intake, sc, sc["calib"].to_numpy())
            r["median_trip_m"] = weighted_median(D, P, kids)
            rows.append(r)
    grid = pd.DataFrame(rows)

    # choose on one fold, score on the other (and vice versa)
    oos = []
    for form, g in grid.groupby("form"):
        for fit_k, test_k in [(0, 1), (1, 0)]:
            best = g.loc[g[f"rmse_fold{fit_k}"].idxmin()]
            oos.append({"form": form, "fit_fold": fit_k, "chosen_param": best["param"],
                        "rmse_test_fold": best[f"rmse_fold{test_k}"]})
    return grid, pd.DataFrame(oos)


def weighted_median(D, P, kids):
    """Median home-school distance over all children (weights kids_i * P_ij)."""
    w = (kids[:, None] * P).ravel()
    d = D.ravel()
    o = np.argsort(d)
    c = np.cumsum(w[o])
    return float(d[o][np.searchsorted(c, c[-1] / 2)])


def car_kids(P, D, car_rate, kids, d50=G_D50, baseline=False):
    g = 1.0 if baseline else g_drive(D, d50)
    p_car = (P * (car_rate[:, None] * g)).sum(axis=1)  # sum_j P(j|i) P(car|i,j)
    return kids * p_car, P * car_rate[:, None] * g * kids[:, None]


def main():
    pc, sc0, D = load_inputs()
    sc = prepare_schools(sc0)
    kids = pc["est_kids"].to_numpy()
    car = pc["car_rate"].to_numpy()
    A_cap = sc["cap_primary"].to_numpy()
    cap_eff = sc["cap_effective"].to_numpy()

    print("=== Schools")
    print("all-through (DfE 4xxx):", sc.loc[sc.all_through, "EstablishmentName"].tolist())
    print(f"imputed rolls (capacity x median fill {sc.attrs['fill_median']:.3f}):",
          sc.loc[sc.roll_imputed, ["EstablishmentName", "roll_primary"]].round(0).values.tolist())
    print(f"schools used for calibration: {sc.calib.sum()} | total primary capacity "
          f"{sc.cap_primary.sum():,.0f} | effective capacity {cap_eff.sum():,.0f} | "
          f"primary roll {sc.roll_primary.sum():,.0f} | est_kids {kids.sum():,.1f}")

    # ---- 1. choose decay parameters (input data only)
    grid, oos = calibrate(D, kids, sc, np.ones(len(sc)))  # geography only
    grid.round(3).to_csv(OUT / "method2_calibration_grid.csv", index=False)
    print("\n=== Calibration grid (geography only: A = 1, no cap)")
    print(grid.round(2).to_string(index=False))
    print("\n=== 2-fold out-of-sample check (choose on one half of schools, score on the other)")
    print(oos.round(1).to_string(index=False))
    best = {f: grid[grid.form == f].loc[grid[grid.form == f]["rmse_all"].idxmin()] for f in ("exp", "power")}
    oos_mean = oos.groupby("form")["rmse_test_fold"].mean()
    main_form = oos_mean.idxmin()
    main_p = float(best[main_form]["param"])
    print(f"\nmean out-of-sample RMSE: {oos_mean.round(1).to_dict()} -> main form = {main_form}, "
          f"param = {main_p:g}")

    # ---- 2. refinement ladder, one step at a time
    m1 = pd.read_csv(OUT / "method1_school.csv").set_index("EstablishmentNumber") \
        .loc[sc["EstablishmentNumber"], "assigned_kids"].to_numpy()
    ladder = []

    def record(name, P, extra=""):
        intake = kids @ P
        fit = school_fit(intake, sc)
        ck, _ = car_kids(P, D, car, kids)
        ckb, _ = car_kids(P, D, car, kids, baseline=True)
        ladder.append({"version": name, "corr_with_roll": fit["corr"], "rmse_vs_roll": fit["rmse"],
                       "schools_over_eff_cap": int((intake > cap_eff * 1.001).sum()),
                       "median_trip_m": weighted_median(D, P, kids),
                       "car_kids_main": ck.sum(), "car_share_main": ck.sum() / kids.sum(),
                       "car_kids_baseline": ckb.sum(), "note": extra})
        return intake

    P_nn = np.zeros_like(D)
    P_nn[np.arange(len(D)), D.argmin(1)] = 1
    assert np.allclose(kids @ P_nn, m1)
    record("M1 nearest school", P_nn)
    ones = np.ones(len(sc))
    for f in ("exp", "power"):
        p = float(best[f]["param"])
        record(f"M2a {f} decay only (p={p:g})", attend_prob(decay(D, f, p), ones, kids)[0])
        record(f"M2b {f} + capacity weight", attend_prob(decay(D, f, p), A_cap, kids)[0])
    F_main = decay(D, main_form, main_p)
    P, B, iters = attend_prob(F_main, A_cap, kids, cap=cap_eff)
    record(f"M2c {main_form} + capacity weight + cap (MAIN)", P, f"{iters} balancing iterations")
    ladder = pd.DataFrame(ladder)
    ladder.to_csv(OUT / "method2_ladder.csv", index=False)
    print("\n=== Refinement ladder")
    print(ladder.round(3).to_string(index=False))
    print(f"schools with balancing factor B < 1: {(B < 0.999).sum()} | min B {B.min():.3f}")

    # ---- 3. outputs of the main model
    rs = P.sum(axis=1)
    assert P.shape == D.shape and (P >= 0).all() and (rs <= 1 + 1e-9).all()
    np.save(OUT / "attend_prob.npy", P)
    ck, ck_ij = car_kids(P, D, car, kids)
    ckb, _ = car_kids(P, D, car, kids, baseline=True)
    est = pd.DataFrame({"POSTCODE": pc["POSTCODE"], "est_kids": kids, "est_car_kids": ck,
                        "est_car_rate": np.divide(ck, kids, out=np.zeros_like(ck), where=kids > 0),
                        "est_car_kids_baseline": ckb,
                        "exp_dist_m": (P * D).sum(axis=1)})
    est.to_csv(OUT / "estimates_postcode.csv", index=False)

    intake = kids @ P
    school = sc[["EstablishmentNumber", "EstablishmentName", "TypeOfEstablishment", "all_through",
                 "roll_imputed", "SchoolCapacity", "NumberOfPupils", "cap_primary", "roll_primary",
                 "cap_effective"]].copy()
    school["pred_pupils"] = intake
    school["pred_car_pupils"] = ck_ij.sum(axis=0)
    school["pred_car_share"] = school["pred_car_pupils"] / school["pred_pupils"]
    school["balancing_B"] = B
    # implied catchment: distance within which 80% / 95% of predicted pupils live
    for q in (0.8, 0.95):
        school[f"catchment_p{int(q * 100)}_m"] = [weighted_quantile(D[:, j], kids * P[:, j], q)
                                                  for j in range(len(sc))]
    school.to_csv(OUT / "method2_school.csv", index=False)

    print("\n=== Main model outputs")
    print(f"row sums: min {rs.min():.6f}, max {rs.max():.6f}")
    print(f"est_car_kids (main, g(d) with d50={G_D50:g} m): {ck.sum():,.1f} = {ck.sum() / kids.sum():.1%} of children")
    print(f"est_car_kids (baseline g=1, upper bound):     {ckb.sum():,.1f} = {ckb.sum() / kids.sum():.1%}")
    print("per-postcode est_car_rate: median %.3f, p10 %.3f, p90 %.3f"
          % tuple(est.loc[kids > 0, "est_car_rate"].quantile([.5, .1, .9])))
    print("per-school pred_car_share: median %.3f, p10 %.3f, p90 %.3f"
          % tuple(school["pred_car_share"].quantile([.5, .1, .9])))
    print("implied catchment p80: median %.0f m, max %.0f m"
          % (school["catchment_p80_m"].median(), school["catchment_p80_m"].max()))

    sensitivity(D, kids, car, sc, main_form, main_p)
    figures(grid, D, est, pc, main_form, main_p)


def weighted_quantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, q * c[-1])])


def sensitivity(D, kids, car, sc, form, p):
    """One assumption changed at a time, everything else at the main setting."""
    base_sc = sc
    rows = []

    def run(label, value, form=form, p=p, sc=base_sc, d50=G_D50):
        P, _, _ = attend_prob(decay(D, form, p), sc["cap_primary"].to_numpy(), kids,
                              cap=sc["cap_effective"].to_numpy())
        ck, _ = car_kids(P, D, car, kids, d50=d50)
        rows.append({"assumption": label, "value": value, "car_kids": ck.sum(),
                     "car_share": ck.sum() / kids.sum(), "median_trip_m": weighted_median(D, P, kids),
                     **school_fit(kids @ P, sc)})

    run("main", "-")
    for k in (0.5, 0.75, 1.5, 2.0):
        run("decay parameter x", k, p=p * k)
    for d50 in (500, 750, 1500, 2000):
        run("g(d) midpoint d50 (m)", d50, d50=d50)
    run("all-through primary share", 0.3, sc=prepare_schools(sc, 0.3))
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "method2_sensitivity.csv", index=False)
    print("\n=== Sensitivity (one change at a time)")
    print(s.round(3).to_string(index=False))


def figures(grid, D, est, pc, form, p):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, f, lab in [(ax[0], "exp", "lambda (m)"), (ax[1], "power", "beta")]:
        g = grid[grid.form == f]
        for c, s in [("rmse_fold0", "fold 0"), ("rmse_fold1", "fold 1"), ("rmse_all", "all")]:
            a.plot(g["param"], g[c], marker="o", label=s)
        a.set(title=f"{f} decay: school intake vs roll", xlabel=lab, ylabel="RMSE (pupils)")
        if f == "exp":
            a.set_xscale("log")
            a.set_xticks([150, 300, 500, 1000, 2000], ["150", "300", "500", "1000", "2000"])
            a.minorticks_off()
        a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage3_calibration.png", dpi=150)
    plt.close(fig)

    d = np.linspace(0, 4000, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for f, pp in [("exp", float(grid[grid.form == "exp"].loc[grid[grid.form == "exp"].rmse_all.idxmin(), "param"])),
                  ("power", float(grid[grid.form == "power"].loc[grid[grid.form == "power"].rmse_all.idxmin(), "param"]))]:
        y = np.exp(-d / pp) if f == "exp" else ((d + POWER_OFFSET) / POWER_OFFSET) ** (-pp)
        ax[0].plot(d, y, label=f"{f} (p={pp:g})")
    ax[0].set(title="Distance decay f(d), scaled to 1 at d=0", xlabel="distance (m)", ylabel="relative weight")
    ax[0].legend()
    for d50 in (500, 1000, 2000):
        ax[1].plot(d, g_drive(d, d50), label=f"d50 = {d50} m" + (" (main)" if d50 == G_D50 else ""))
    ax[1].set(title="g(d): share of car-owning families who drive", xlabel="distance (m)", ylabel="g(d)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage3_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    m = est["est_kids"] > 0
    s = ax.scatter(pc.loc[m, "X"], pc.loc[m, "Y"], c=est.loc[m, "est_car_rate"], s=1.5, cmap="viridis")
    fig.colorbar(s, ax=ax, shrink=0.7, label="est_car_rate (share of children driven)")
    ax.set(title=f"Method 2: estimated share of primary children driven to school\n"
                 f"({form} decay p={p:g}, capacity-weighted and capped, d50={G_D50:g} m)",
           xlabel="Easting (m)", ylabel="Northing (m)", aspect="equal")
    fig.tight_layout()
    fig.savefig(FIG / "stage3_map_car_rate.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
