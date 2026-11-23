"""
Quality checks for Stage 2 outputs. Run after src/stage2_distances_method1.py:
    .venv/bin/python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
import stage2_distances_method1 as s2  # noqa: E402


@pytest.fixture(scope="module")
def data():
    D = np.load(OUT / "dist_matrix.npy")
    rows = pd.read_csv(OUT / "dist_rows_postcodes.csv", dtype={"POSTCODE": str})
    cols = pd.read_csv(OUT / "dist_cols_schools.csv")
    pc, sc = s2.load_inputs()
    m1 = pd.read_csv(OUT / "method1_postcode.csv", dtype={"POSTCODE": str})
    return D, rows, cols, pc, sc, m1


def test_tiny_hand_example():
    # 3-4-5 triangle and a zero distance, checked by hand
    D = s2.distance_matrix([0, 3], [0, 4], [0, 3], [0, 0])
    assert np.allclose(D, [[0, 3], [5, 4]])


def test_shape_and_order(data):
    D, rows, cols, pc, sc, _ = data
    assert D.shape == (20799, 225)
    assert (rows["POSTCODE"] == pc["POSTCODE"]).all()
    assert (cols["EstablishmentNumber"] == sc["EstablishmentNumber"]).all()


def test_matches_scipy_cdist(data):
    D, _, _, pc, sc, _ = data
    ref = cdist(pc[["X", "Y"]].to_numpy(float), sc[["X", "Y"]].to_numpy(float))
    assert np.allclose(D, ref)


def test_random_entries_by_hand(data):
    D, _, _, pc, sc, _ = data
    rng = np.random.default_rng(42)
    for i, j in zip(rng.integers(0, len(pc), 50), rng.integers(0, len(sc), 50)):
        exp = np.hypot(pc.X[i] - sc.X[j], pc.Y[i] - sc.Y[j])
        assert np.isclose(D[i, j], exp)


def test_expected_nearest_distance_figures(data):
    d = data[5]["dist_nearest_m"]
    assert 400 < d.median() < 450
    assert 3500 < d.max() < 3700
    assert (d > 3000).sum() == 13


def test_method1_nearest_is_true_minimum(data):
    D, _, cols, _, _, m1 = data
    j = D.argmin(axis=1)
    assert (m1["nearest_school"].to_numpy() == cols["EstablishmentNumber"].to_numpy()[j]).all()
    assert np.allclose(m1["dist_nearest_m"], D.min(axis=1))


def test_method1_conserves_children(data):
    _, _, _, pc, _, m1 = data
    school = pd.read_csv(OUT / "method1_school.csv")
    assert np.isclose(school["assigned_kids"].sum(), pc["est_kids"].sum())
    assert (m1["est_car_kids_m1"] <= m1["est_kids"] + 1e-12).all()
