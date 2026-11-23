"""
Quality checks for Stage 1 output. Run from the project folder:
    .venv/bin/python -m pytest tests -q
(run src/stage1_data_prep.py first so outputs/postcodes_master.csv exists)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import stage1_data_prep as s1  # noqa: E402


@pytest.fixture(scope="module")
def master():
    return pd.read_csv(ROOT / "outputs" / "postcodes_master.csv", dtype={"POSTCODE": str, "OA": str})


def test_columns_and_shape(master):
    assert list(master.columns) == ["POSTCODE", "OA", "X", "Y", "households", "car_rate", "est_kids"]
    assert len(master) == 20799


def test_keys_unique_no_nan(master):
    assert master["POSTCODE"].is_unique
    assert master.notna().all().all()


def test_rates_and_counts_in_range(master):
    assert master["car_rate"].between(0, 1).all()
    assert (master["est_kids"] >= 0).all()
    assert (master["households"] > 0).all()


def test_equal_split_preserves_oa_totals(master):
    """Splitting must not create or lose children (no 8x inflation)."""
    _, oa = s1.build_master()
    assert np.isclose(master["est_kids"].sum(), oa["kids_oa"].sum())
    assert np.isclose(master["households"].sum(), oa["households"].sum())
    per_oa = master.groupby("OA")["est_kids"].sum()
    assert np.allclose(per_oa, oa.loc[per_oa.index, "kids_oa"])


def test_age_fractions_cover_seven_year_groups():
    # each band is 5 single years: fraction*5 = single years taken from the band
    years = sum(f * 5 for f in s1.AGE_FRACTIONS.values())
    assert np.isclose(years, 7.0)


def test_total_children_close_to_school_rolls(master):
    # primary rolls without the two all-through schools = 71,682 (check_inputs.py)
    assert 0.85 < master["est_kids"].sum() / 71682 < 1.15
