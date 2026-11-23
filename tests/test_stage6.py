"""
Quality checks for Stage 6. Run after src/stage6_sensitivity.py:
    .venv/bin/python -m pytest tests -q
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def test_one_at_a_time_table():
    s = pd.read_csv(OUT / "sensitivity_one_at_a_time.csv")
    main = s[s.assumption == "main"].iloc[0]
    assert np.isclose(main.spearman_vs_main, 1) and np.isclose(main.top10_overlap, 1)
    assert np.isclose(main.car_share, 0.3535, atol=5e-4)          # Stage 3 headline
    # directions that must hold by construction
    d50 = s[s.assumption.str.startswith("g(d) midpoint")].set_index("value")["car_share"]
    assert d50["500"] > main.car_share > d50["2000"]
    lam = s[s.assumption.str.startswith("decay lambda")].set_index("value")["car_share"]
    assert lam["325"] < main.car_share < lam["1300"]
    assert s["car_share"].between(0, 1).all()


def test_monte_carlo_draws():
    d = pd.read_csv(OUT / "sensitivity_mc_draws.csv")
    assert len(d) == 200
    assert d["lambda"].between(325, 1300).all() and d["d50"].between(500, 2000).all()
    assert d["scale"].between(150, 400).all() and d["car_share"].between(0, 1).all()


def test_uncertainty_bands():
    u = pd.read_csv(OUT / "estimates_postcode_uncertainty.csv", dtype={"POSTCODE": str})
    e = pd.read_csv(OUT / "estimates_postcode.csv", dtype={"POSTCODE": str})
    assert len(u) == 20799 and u["POSTCODE"].is_unique and (u["POSTCODE"] == e["POSTCODE"]).all()
    assert (u.car_rate_p5 <= u.car_rate_p50 + 1e-12).all() and (u.car_rate_p50 <= u.car_rate_p95 + 1e-12).all()
    assert u[["car_rate_p5", "car_rate_p95"]].stack().between(0, 1).all()
    assert (u.car_kids_p95 <= u.est_kids + 1e-9).all()
    k = e["est_kids"] > 0          # Stage 3 writes rate 0 where there are no children
    assert np.allclose(u.loc[k, "est_car_rate"], e.loc[k, "est_car_rate"])
