"""
Quality checks for Stage 5. Run after src/stage5_fairness.py:
    .venv/bin/python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import stage5_fairness as s5  # noqa: E402


def test_denomination_from_name():
    assert s5.denomination("St Francis Catholic Primary School, Morley") == "Catholic"
    assert s5.denomination("Rothwell St Mary's RC Primary School") == "Catholic"
    assert s5.denomination("Otley All Saints CofE Primary School") == "Church of England"
    assert s5.denomination("Manston St James Primary Academy") == "No religious designation"
    assert s5.denomination("Dixons Trinity Chapeltown") == "No religious designation"


def test_propagated_error_toy():
    # 2 postcodes x 3 schools; school 2 is not surveyed (NaN)
    P = np.array([[0.5, 0.5, 0.0],
                  [0.2, 0.0, 0.8]])
    err = np.array([0.1, -0.3, np.nan])
    e, cover = s5.propagated_error(P, err, min_coverage=0.5)
    assert np.allclose(cover, [1.0, 0.2])
    assert np.isclose(e[0], -0.1)          # (0.5*0.1 + 0.5*-0.3) / 1
    assert np.isnan(e[1])                  # only 20% of attendance at surveyed schools


def test_boot_ci_contains_mean_and_is_reproducible():
    x = np.random.default_rng(0).normal(size=50)
    a, b = s5.boot_mean_ci(x), s5.boot_mean_ci(x)
    assert a == b and a[1] <= a[0] <= a[2]


def test_religion_check_on_synthetic_data():
    # group "A" lives where estimated car use is high -> disparity ratio > 1
    pcf = pd.DataFrame({"OA": ["o1", "o1", "o2", "o3"], "est_kids": [1.0, 1.0, 2.0, 2.0],
                        "est_car_kids": [0.8, 0.8, 0.4, 1.0], "propagated_error": [-0.1, -0.1, 0.1, 0.0]})
    rel = pd.DataFrame({"geography code": ["o1", "o2", "o3"], "Total": [100, 100, 100],
                        "A": [90, 10, 50], "B": [10, 90, 50]})
    r = s5.religion_check(rel, pcf).set_index("group")
    assert set(r.index) == {"A", "B"}       # Total column ignored
    assert r.loc["A", "disparity_ratio"] > 1 > r.loc["B", "disparity_ratio"]
    assert np.isclose(r["children_est"].sum(), pcf["est_kids"].sum())
    assert r.loc["A", "spearman_share_vs_propagated_error"] < 0


def test_outputs_consistent():
    f = pd.read_csv(OUT / "fairness_school.csv")
    assert len(f) == 225 and f["EstablishmentNumber"].is_unique
    assert f["catchment_car_rate"].between(0, 1).all()
    g = pd.read_csv(OUT / "fairness_groups.csv")
    assert g["n_schools"].sum() == f["in_survey_comparison"].sum()
    assert (g["ci_low"] <= g["mean_error"]).all() and (g["mean_error"] <= g["ci_high"]).all()
    p = pd.read_csv(OUT / "fairness_postcode.csv", dtype={"POSTCODE": str})
    assert len(p) == 20799 and p["POSTCODE"].is_unique and p["POSTCODE"].notna().all()
    assert (p["catholic_exposure"] + p["cofe_exposure"] <= 1 + 1e-9).all()
    assert p["survey_coverage"].between(0, 1 + 1e-9).all()
