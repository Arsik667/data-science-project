"""
Quality checks for Stage 3 (Method 2). Run after src/stage3_method2.py:
    .venv/bin/python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import stage3_method2 as s3  # noqa: E402


def test_hand_example_exponential():
    # 1 postcode, 2 schools at 0 m and 650 m, lambda = 650, equal attractiveness:
    # weights 1 and e^-1 -> P = 1/(1+e^-1), e^-1/(1+e^-1)
    D = np.array([[0.0, 650.0]])
    P, _, _ = s3.attend_prob(s3.decay(D, "exp", 650), np.ones(2), np.array([10.0]))
    e = np.exp(-1)
    assert np.allclose(P, [[1 / (1 + e), e / (1 + e)]])


def test_capacity_balancing_redistributes_and_conserves():
    # 2 postcodes x 2 schools; school 0 is near both but has only 5 places
    D = np.array([[100.0, 900.0], [200.0, 800.0]])
    kids = np.array([10.0, 10.0])
    cap = np.array([5.0, 100.0])
    P, B, _ = s3.attend_prob(s3.decay(D, "exp", 300), np.ones(2), kids, cap=cap)
    intake = kids @ P
    assert intake[0] <= 5 * 1.001
    assert np.isclose(intake.sum(), 20)       # nobody disappears
    assert B[0] < 1 and B[1] == 1


def test_g_drive_shape():
    d = np.array([0, 500, 1000, 1500, 5000.0])
    g = s3.g_drive(d)
    assert np.isclose(g[2], 0.5) and np.all(np.diff(g) > 0) and g[0] < 0.05 and g[-1] > 0.99


@pytest.fixture(scope="module")
def out():
    P = np.load(OUT / "attend_prob.npy")
    est = pd.read_csv(OUT / "estimates_postcode.csv", dtype={"POSTCODE": str})
    sch = pd.read_csv(OUT / "method2_school.csv")
    pc = pd.read_csv(OUT / "postcodes_master.csv", dtype={"POSTCODE": str})
    return P, est, sch, pc


def test_attend_prob_valid(out):
    P, est, sch, _ = out
    assert P.shape == (len(est), len(sch)) == (20799, 225)
    assert (P >= 0).all() and (P.sum(axis=1) <= 1 + 1e-9).all()


def test_estimates_columns_and_bounds(out):
    _, est, _, pc = out
    assert list(est.columns[:4]) == ["POSTCODE", "est_kids", "est_car_kids", "est_car_rate"]
    assert (est["POSTCODE"] == pc["POSTCODE"]).all() and est["POSTCODE"].is_unique
    assert est.notna().all().all()
    # g(d) <= 1, so the main estimate never exceeds the baseline kids x car_rate
    assert (est["est_car_kids"] <= pc["est_kids"] * pc["car_rate"] + 1e-9).all()
    assert np.allclose(est["est_car_kids_baseline"], pc["est_kids"] * pc["car_rate"])


def test_school_totals_consistent(out):
    _, est, sch, _ = out
    assert np.isclose(sch["pred_pupils"].sum(), est["est_kids"].sum())
    assert np.isclose(sch["pred_car_pupils"].sum(), est["est_car_kids"].sum())
    assert (sch["pred_pupils"] <= sch["cap_effective"] * 1.001).all()
