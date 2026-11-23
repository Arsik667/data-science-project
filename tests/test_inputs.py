"""
Checks that the raw input files are present and look as the pipeline expects.
    .venv/bin/python -m pytest tests -q
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = ["postcode locations.csv", "school locations.csv", "access to car by oa.csv",
         "number of households by oa.csv", "agebands by oa.csv", "2025 travel to school survey.csv",
         "2024 cut off distances.pdf", "2024-25 Annual Travel To School Survey Summary.pdf"]


def test_all_files_present():
    missing = [f for f in FILES if not (RAW / f).exists()]
    assert not missing, f"put these files in data/raw/: {missing}"


def test_postcodes():
    pc = pd.read_csv(RAW / "postcode locations.csv", dtype={"POSTCODE": str, "OA": str})
    assert len(pc) == 20799 and pc["POSTCODE"].is_unique and pc["POSTCODE"].notna().all()
    assert pc["OA"].notna().all() and pc[["X", "Y"]].notna().all().all()


def test_schools():
    sc = pd.read_csv(RAW / "school locations.csv")
    assert len(sc) == 225 and sc["EstablishmentNumber"].is_unique
    assert sc[["X", "Y"]].notna().all().all()


def test_survey_reads_with_cp1252():
    sv = pd.read_csv(RAW / "2025 travel to school survey.csv", encoding="cp1252")
    assert "Grand Total" in sv["School Name"].values and "Car/Van" in sv.columns
