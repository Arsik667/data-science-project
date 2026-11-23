"""
Quality checks for Stage 7. Run after src/stage7_report.py:
    .venv/bin/python -m pytest tests -q
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP, OUT = ROOT / "report", ROOT / "outputs"


def test_no_unfilled_placeholders():
    for f in (REP / "report_draft.md", REP / "presentation_outline.md", ROOT / "README.md"):
        text = f.read_text()
        assert not re.search(r"\{[A-Za-z0-9_]+\}", text), f


def test_every_template_key_is_traced():
    nums = pd.read_csv(OUT / "report_numbers.csv")
    assert nums["key"].is_unique and nums["source"].notna().all()
    keys = set()
    for f in ("report_template.md", "presentation_template.md", "readme_template.md"):
        keys |= set(re.findall(r"\{([A-Za-z0-9_]+)\}", (REP / f).read_text()))
    assert keys == set(nums["key"])


def test_headline_numbers_match_outputs():
    nums = pd.read_csv(OUT / "report_numbers.csv").set_index("key")["value_in_report"]
    est = pd.read_csv(OUT / "estimates_postcode.csv")
    assert nums["car_kids"] == f"{est.est_car_kids.sum():,.0f}"
    assert nums["total_kids"] == f"{est.est_kids.sum():,.0f}"
    assert nums["n_postcodes"] == f"{len(est):,.0f}"
    draft = (REP / "report_draft.md").read_text()
    assert f"{nums['car_kids']} of {nums['total_kids']} children ({nums['car_share']})" in draft
