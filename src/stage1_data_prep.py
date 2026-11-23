"""
Stage 1 - Data preparation (Role 1).

Builds outputs/postcodes_master.csv with one row per postcode:
    POSTCODE, OA, X, Y, households, car_rate, est_kids

Run from the project folder:
    .venv/bin/python src/stage1_data_prep.py

Key decisions (details in outputs/assumptions_log.md):
  * car access  = household has 1 or more cars/vans
                  car_rate = 1 - "No cars" / (sum of the four real categories)
  * primary age = Reception..Year 6 on Census day (21 March 2021),
                  i.e. children aged 4y7m .. 11y6m (exactly 7 school cohorts).
                  Assuming ages are spread evenly inside each 5-year band:
                      share of "0-4"   = (5/12 of age 4)            / 5 = 1/12
                      share of "5-9"   = all                              = 1
                      share of "10-14" = (age 10 + 7/12 of age 11)  / 5 = 19/60
  * each OA's children and households are split EQUALLY between the
    postcodes in that OA (no postcode-level population is provided).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"  # original input files (names kept as supplied)
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# Fractions of each census age band that are primary-school pupils.
# Main choice is calendar-based (see module docstring); the "flat" choice
# (1/5 of each outer band) is kept only for the sensitivity check below.
AGE_FRACTIONS = {"Aged 4 years and under": 1 / 12,
                 "Aged 5 to 9 years": 1.0,
                 "Aged 10 to 14 years": 19 / 60}
AGE_FRACTIONS_FLAT = {"Aged 4 years and under": 0.2,
                      "Aged 5 to 9 years": 1.0,
                      "Aged 10 to 14 years": 0.2}
NO_CAR = "No cars or vans in household"
CAR_CATS = ["1 car or van in household", "2 cars or vans in household",
            "3 or more cars or vans in household"]


def load_postcodes() -> pd.DataFrame:
    pc = pd.read_csv(RAW / "postcode locations.csv",
                     dtype={"POSTCODE": str, "OA": str, "X": float, "Y": float})
    pc["POSTCODE"] = pc["POSTCODE"].str.strip()
    assert pc["POSTCODE"].is_unique, "duplicate postcodes"
    assert pc.notna().all().all(), "NaN in postcode file"
    return pc


def load_car(oas: set) -> pd.DataFrame:
    car = pd.read_csv(RAW / "access to car by oa.csv")
    car.columns = ["OA", "cat", "n"]
    car = car[car["OA"].isin(oas)]
    wide = car.pivot(index="OA", columns="cat", values="n")
    assert (wide["Does not apply"] == 0).all()  # verified: category is empty
    out = pd.DataFrame(index=wide.index)
    out["hh_no_car"] = wide[NO_CAR]
    out["hh_with_car"] = wide[CAR_CATS].sum(axis=1)
    out["hh_car_table"] = out["hh_no_car"] + out["hh_with_car"]
    # Numerator and denominator come from the SAME table, so the rate is
    # always in [0, 1] even though this total differs slightly from the
    # households file (census disclosure-control noise, max |diff| = 6).
    out["car_rate"] = out["hh_with_car"] / out["hh_car_table"]
    return out


def load_households(oas: set) -> pd.Series:
    hh = pd.read_csv(RAW / "number of households by oa.csv")
    hh.columns = ["OA", "households"]
    return hh[hh["OA"].isin(oas)].set_index("OA")["households"]


def load_ages(oas: set) -> pd.DataFrame:
    age = pd.read_csv(RAW / "agebands by oa.csv")
    age.columns = ["OA", "band", "n"]
    age = age[age["OA"].isin(oas) & age["band"].isin(AGE_FRACTIONS)]
    wide = age.pivot(index="OA", columns="band", values="n")
    out = pd.DataFrame(index=wide.index)
    out["kids_oa"] = sum(wide[b] * f for b, f in AGE_FRACTIONS.items())
    out["kids_oa_flat"] = sum(wide[b] * f for b, f in AGE_FRACTIONS_FLAT.items())
    out["age_5_9"] = wide["Aged 5 to 9 years"]
    return out


def build_master() -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = load_postcodes()
    oas = set(pc["OA"])

    oa = (load_car(oas)
          .join(load_households(oas), how="outer")
          .join(load_ages(oas), how="outer"))
    assert len(oa) == len(oas) and oa.notna().all().all(), "OA join incomplete"

    # number of postcodes in each OA -> equal split
    oa["n_postcodes"] = pc.groupby("OA").size()

    m = pc.merge(oa, left_on="OA", right_index=True, how="left", validate="many_to_one")
    m["households"] = m["households"] / m["n_postcodes"]
    m["est_kids"] = m["kids_oa"] / m["n_postcodes"]
    return m, oa


def sanity_report(m: pd.DataFrame, oa: pd.DataFrame) -> dict:
    sc = pd.read_csv(RAW / "school locations.csv")
    all_through = sc["EstablishmentName"].str.contains("Roundhay School|Carr Manor")
    roll_primary = sc.loc[~all_through, "NumberOfPupils"].sum()

    r = {
        "postcodes": len(m),
        "OAs": len(oa),
        "total households (households file)": int(oa["households"].sum()),
        "total households (car table)": int(oa["hh_car_table"].sum()),
        "car_rate, household-weighted": oa["hh_with_car"].sum() / oa["hh_car_table"].sum(),
        "car_rate, mean over OAs": oa["car_rate"].mean(),
        "car_rate, mean over postcodes": m["car_rate"].mean(),
        "car_rate, min / max over OAs": (oa["car_rate"].min(), oa["car_rate"].max()),
        "children aged 5-9": int(oa["age_5_9"].sum()),
        "est primary children, calendar fractions (MAIN)": oa["kids_oa"].sum(),
        "est primary children, flat 0.2 fractions": oa["kids_oa_flat"].sum(),
        "sum of est_kids over postcodes": m["est_kids"].sum(),
        "school roll excl. Roundhay & Carr Manor (3 schools missing roll)": roll_primary,
        "est primary children / roll": oa["kids_oa"].sum() / roll_primary,
        "est primary children who have car access": (m["est_kids"] * m["car_rate"]).sum(),
        "postcodes per OA: median / max": (oa["n_postcodes"].median(), oa["n_postcodes"].max()),
        "est_kids per postcode: median / p99 / max": (m["est_kids"].median(),
                                                     m["est_kids"].quantile(0.99),
                                                     m["est_kids"].max()),
    }
    return r


def make_figures(m: pd.DataFrame, oa: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(oa["car_rate"], bins=40, color="#4C72B0")
    ax[0].set(title="Share of households with a car, by OA (n=%d)" % len(oa),
              xlabel="car_rate", ylabel="number of OAs")
    ax[1].hist(m["est_kids"], bins=60, color="#55A868")
    ax[1].set(title="Estimated primary-age children per postcode",
              xlabel="est_kids", ylabel="number of postcodes", yscale="log")
    fig.tight_layout()
    fig.savefig(FIG / "stage1_distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for a, col, title, cmap in [(ax[0], "car_rate", "Car access rate", "viridis"),
                                (ax[1], "est_kids", "Est. primary children per postcode", "magma_r")]:
        vmax = m[col].quantile(0.99) if col == "est_kids" else None
        s = a.scatter(m["X"], m["Y"], c=m[col], s=1.5, cmap=cmap, vmax=vmax)
        a.set(title=title, xlabel="Easting (m)", ylabel="Northing (m)", aspect="equal")
        fig.colorbar(s, ax=a, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIG / "stage1_maps.png", dpi=150)
    plt.close(fig)


def main() -> None:
    m, oa = build_master()
    cols = ["POSTCODE", "OA", "X", "Y", "households", "car_rate", "est_kids"]
    m[cols].to_csv(OUT / "postcodes_master.csv", index=False)

    print("=== Stage 1 sanity report")
    for k, v in sanity_report(m, oa).items():
        if isinstance(v, tuple):
            v = " / ".join(f"{x:,.3f}" for x in v)
        elif isinstance(v, float):
            v = f"{v:,.4f}" if v < 10 else f"{v:,.1f}"
        print(f"{k:<70} {v}")

    # OAs with unusually many postcodes: each postcode gets a tiny share
    print("\nOAs with the most postcodes:")
    print(oa.nlargest(5, "n_postcodes")[["n_postcodes", "households", "kids_oa", "car_rate"]]
          .round(3).to_string())

    make_figures(m, oa)
    print("\nwrote outputs/postcodes_master.csv, figures/stage1_distributions.png, figures/stage1_maps.png")


if __name__ == "__main__":
    main()
