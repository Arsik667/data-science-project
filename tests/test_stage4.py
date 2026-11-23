"""
Quality checks for Stage 4 (matching + validation). Run after
src/school_matching.py and src/stage4_validation.py:
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
import school_matching as sm  # noqa: E402


@pytest.fixture(scope="module")
def cut():
    return sm.parse_cutoff_pdf()


def test_pdf_row_counts(cut):
    assert len(cut) == 226 and cut["dfe"].is_unique
    assert cut["cutoff_flag"].value_counts().to_dict() == {"all_admitted": 147, "numeric": 76, "nonstandard": 3}
    assert cut.loc[cut.cutoff_flag == "all_admitted", "cutoff_m"].isna().all()


def test_pdf_known_values(cut):
    row = cut.set_index("dfe")
    assert row.loc[2416, "cutoff_miles"] == 0.649                       # Adel Primary
    assert np.isclose(row.loc[2416, "cutoff_m"], 0.649 * 1609.344)
    assert row.loc[3315, "cutoff_miles"] == 0.542                       # St Chad's, non-standard
    assert row.loc[3315, "cutoff_flag"] == "nonstandard"


def test_normalise():
    assert sm.normalise("St. Oswald's C of E Primary") == sm.normalise("St Oswald's CofE Primary")
    assert sm.core("Co-op Academy Brownhill") != sm.core("Co-op Academy Brierley")


def test_school_key():
    key = pd.read_csv(OUT / "school_key.csv")
    sv = sm.load_survey()
    assert len(key) == 225 and key["EstablishmentNumber"].is_unique
    linked = key["name_survey"].dropna()
    assert linked.is_unique and linked.isin(sv["School Name"]).all()
    k = key.set_index("EstablishmentNumber")
    assert k.loc[2358, "name_survey"] == "Yeadon Westfield Infant School"
    assert k.loc[2041, "pdf_dfe"] == 2487
    assert not k.loc[[4041, 4063, 4068, 4075], "primary"].any()


def test_school_validation():
    v = pd.read_csv(OUT / "school_validation.csv")
    assert len(v) == 225
    comp = v[v["in_survey_comparison"]]
    assert comp["primary"].all() and comp["obs_share"].between(0, 1).all()
    assert comp["pred_share_M2"].between(0, 1).all()
