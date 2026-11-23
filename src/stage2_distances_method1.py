"""
Stage 2 - Distances and Method 1 (Role 2).

1. Full postcode x school straight-line distance matrix (metres), built with
   numpy broadcasting (no Python loops):
       d[i, j] = sqrt((X_i - X_j)^2 + (Y_i - Y_j)^2)
2. Method 1 = every child attends the NEAREST school.
   (Provisional: confirm against the brief's definition of Method 1.)
   Car-driven children use the baseline P(car | postcode) = car_rate, i.e.
   every child with car access is assumed to be driven. This is an UPPER
   BOUND and is refined in Method 2.

Inputs : outputs/postcodes_master.csv (Stage 1), school locations.csv
Outputs: outputs/dist_matrix.npy            float64, shape (n_postcodes, n_schools), metres
         outputs/dist_rows_postcodes.csv    row order    (row_index, POSTCODE)
         outputs/dist_cols_schools.csv      column order (col_index, EstablishmentNumber)
         outputs/method1_postcode.csv       POSTCODE, nearest_school, dist_nearest_m, est_kids, est_car_kids_m1
         outputs/method1_school.csv         per-school totals under Method 1
         figures/stage2_map.png, figures/stage2_nearest_distance.png

Run from the project folder:
    .venv/bin/python src/stage2_distances_method1.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"  # original input files (names kept as supplied)
OUT = ROOT / "outputs"
FIG = ROOT / "figures"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = pd.read_csv(OUT / "postcodes_master.csv", dtype={"POSTCODE": str, "OA": str})
    sc = pd.read_csv(RAW / "school locations.csv")
    sc = sc.rename(columns={"TypeOfEstablishment (name)": "TypeOfEstablishment",
                            "ReligiousEthos (name)": "ReligiousEthos"})
    assert sc["EstablishmentNumber"].is_unique
    return pc, sc


def distance_matrix(px, py, sx, sy) -> np.ndarray:
    """Euclidean distance in metres between every point (px, py) and every
    school (sx, sy). Broadcasting: (n, 1) - (1, m) -> (n, m)."""
    dx = np.asarray(px, float)[:, None] - np.asarray(sx, float)[None, :]
    dy = np.asarray(py, float)[:, None] - np.asarray(sy, float)[None, :]
    return np.hypot(dx, dy)


def method1(pc: pd.DataFrame, sc: pd.DataFrame, D: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    j = D.argmin(axis=1)  # column index of the nearest school
    m1 = pd.DataFrame({
        "POSTCODE": pc["POSTCODE"],
        "nearest_school": sc["EstablishmentNumber"].to_numpy()[j],
        "dist_nearest_m": D[np.arange(len(pc)), j],
        "est_kids": pc["est_kids"],
        "est_car_kids_m1": pc["est_kids"] * pc["car_rate"],
    })

    agg = (m1.groupby("nearest_school")
             .agg(n_postcodes=("POSTCODE", "size"),
                  assigned_kids=("est_kids", "sum"),
                  assigned_car_kids=("est_car_kids_m1", "sum"),
                  median_dist_m=("dist_nearest_m", "median"),
                  max_dist_m=("dist_nearest_m", "max")))
    school = (sc[["EstablishmentNumber", "EstablishmentName", "TypeOfEstablishment",
                  "SchoolCapacity", "NumberOfPupils"]]
              .merge(agg, left_on="EstablishmentNumber", right_index=True, how="left")
              .fillna({"n_postcodes": 0, "assigned_kids": 0, "assigned_car_kids": 0}))
    school["assigned_over_capacity"] = school["assigned_kids"] / school["SchoolCapacity"]
    school["assigned_over_roll"] = school["assigned_kids"] / school["NumberOfPupils"]
    return m1, school


def make_figures(pc, sc, m1) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    s = ax.scatter(pc["X"], pc["Y"], c=m1["dist_nearest_m"], s=1.5, cmap="plasma_r", vmax=2000)
    ax.scatter(sc["X"], sc["Y"], marker="^", s=28, facecolor="white", edgecolor="black",
               linewidth=0.8, label=f"schools (n={len(sc)})")
    far = m1["dist_nearest_m"] > 3000
    ax.scatter(pc.loc[far, "X"], pc.loc[far, "Y"], s=30, facecolor="none", edgecolor="red",
               label=f"postcodes > 3 km from any school (n={far.sum()})")
    fig.colorbar(s, ax=ax, shrink=0.7, label="distance to nearest school (m, capped at 2,000)")
    ax.set(title="Leeds postcodes and schools", xlabel="Easting (m)", ylabel="Northing (m)",
           aspect="equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage2_map.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(m1["dist_nearest_m"], bins=80, color="#4C72B0")
    med = m1["dist_nearest_m"].median()
    ax.axvline(med, color="red", ls="--", label=f"median = {med:,.0f} m")
    ax.set(title="Distance from each postcode to its nearest school",
           xlabel="metres", ylabel="number of postcodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "stage2_nearest_distance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    pc, sc = load_inputs()
    D = distance_matrix(pc["X"], pc["Y"], sc["X"], sc["Y"])
    assert D.shape == (len(pc), len(sc)) and np.isfinite(D).all() and (D >= 0).all()

    np.save(OUT / "dist_matrix.npy", D)
    pd.DataFrame({"row_index": range(len(pc)), "POSTCODE": pc["POSTCODE"]}) \
        .to_csv(OUT / "dist_rows_postcodes.csv", index=False)
    pd.DataFrame({"col_index": range(len(sc)), "EstablishmentNumber": sc["EstablishmentNumber"]}) \
        .to_csv(OUT / "dist_cols_schools.csv", index=False)

    m1, school = method1(pc, sc, D)
    m1.to_csv(OUT / "method1_postcode.csv", index=False)
    school.to_csv(OUT / "method1_school.csv", index=False)

    d = m1["dist_nearest_m"]
    print("=== Distance matrix")
    print(f"shape {D.shape}, size {D.nbytes / 1e6:.1f} MB, all distances: min {D.min():,.0f} m, "
          f"max {D.max():,.0f} m")
    print("=== Distance to nearest school (per postcode)")
    print(f"median {d.median():,.0f} m | mean {d.mean():,.0f} m | p90 {d.quantile(.9):,.0f} m | "
          f"max {d.max():,.0f} m")
    print(f"postcodes > 1 km: {(d > 1000).sum()} | > 2 km: {(d > 2000).sum()} | > 3 km: {(d > 3000).sum()}")
    print("child-weighted median distance: %.0f m"
          % np.interp(0.5, (m1.sort_values("dist_nearest_m")["est_kids"].cumsum()
                            / m1["est_kids"].sum()).to_numpy(),
                      m1.sort_values("dist_nearest_m")["dist_nearest_m"].to_numpy()))
    far = m1[d > 3000].merge(pc[["POSTCODE", "OA"]], on="POSTCODE")
    print(f"postcodes > 3 km hold {far['est_kids'].sum():.1f} estimated children")

    print("\n=== Method 1 (nearest school)")
    print(f"total est_kids {m1['est_kids'].sum():,.1f} | est_car_kids_m1 (upper bound) "
          f"{m1['est_car_kids_m1'].sum():,.1f} = {m1['est_car_kids_m1'].sum() / m1['est_kids'].sum():.1%}")
    print(f"schools that are nobody's nearest: {(school['n_postcodes'] == 0).sum()}")
    r = school["assigned_over_capacity"]
    print(f"assigned kids / capacity: median {r.median():.2f} | <0.5: {(r < .5).sum()} schools | "
          f">1.5: {(r > 1.5).sum()} schools | max {r.max():.2f}")
    print("most overloaded under Method 1:")
    print(school.nlargest(5, "assigned_over_capacity")[
        ["EstablishmentNumber", "EstablishmentName", "SchoolCapacity", "NumberOfPupils",
         "assigned_kids", "assigned_over_capacity"]].round(2).to_string(index=False))
    ok = school["NumberOfPupils"].notna()
    print("corr(assigned_kids, NumberOfPupils) over %d schools: %.3f"
          % (ok.sum(), school.loc[ok, ["assigned_kids", "NumberOfPupils"]].corr().iloc[0, 1]))

    make_figures(pc, sc, m1)
    print("\nwrote dist_matrix.npy, row/col order files, method1_postcode.csv, method1_school.csv, 2 figures")


if __name__ == "__main__":
    main()
