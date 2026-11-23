"""
School matching for validation (Role 4, part 1).

Builds outputs/school_key.csv:
    EstablishmentNumber, name_school_list, name_survey, name_pdf, primary,
    survey_match_method, survey_match_score, pdf_match_method

Order: (1) exact name, (2) exact after normalisation, (3) fuzzy (rapidfuzz),
(4) manual decisions in data/manual/manual_matches.csv, which always win.
Fuzzy matches below AUTO_ACCEPT are NOT accepted automatically; they are
printed for a human to confirm and must be added to the manual file.

The cut-off PDF is keyed by DfE number (= EstablishmentNumber); names are only
used to double-check the number and to find schools whose number changed.

Run:  .venv/bin/python src/school_matching.py
"""
import re
from pathlib import Path

import pandas as pd
import pdfplumber
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"  # original input files (names kept as supplied)
OUT = ROOT / "outputs"
MANUAL = ROOT / "data" / "manual" / "manual_matches.csv"
AUTO_ACCEPT = 95  # fuzzy score needed to accept without review

# words that carry no identifying information
STOP = r"\b(primary|school|academy|a|voluntary|aided|controlled|the|and|of|infant|junior|nursery|" \
       r"church|england|ce|cofe|c|e|va|vc|catholic|rc|community|federation)\b"


def normalise(name: str) -> str:
    s = name.lower().replace("&", " and ").replace("’", "'")
    s = re.sub(r"\(.*?\)", " ", s)              # drop "(previously ...)" notes
    s = re.sub(r"\bst\.?\b", "st", s)
    s = re.sub(r"\bc\s*of\s*e\b", "cofe", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("'", ""))
    return re.sub(r"\s+", " ", s).strip()


def core(name: str) -> str:
    """Normalised name without generic words: used for fuzzy scoring."""
    return re.sub(r"\s+", " ", re.sub(STOP, " ", normalise(name))).strip()


def load_schools() -> pd.DataFrame:
    sc = pd.read_csv(RAW / "school locations.csv")
    sc = sc.rename(columns={"TypeOfEstablishment (name)": "TypeOfEstablishment"})
    sc["primary"] = ~sc["EstablishmentNumber"].between(4000, 4999)  # 4xxx = all-through
    return sc


def load_survey() -> pd.DataFrame:
    sv = pd.read_csv(RAW / "2025 travel to school survey.csv", encoding="cp1252")
    sv = sv[sv["School Name"] != "Grand Total"].copy()
    sv["School Name"] = sv["School Name"].str.strip()
    return sv


def parse_cutoff_pdf() -> pd.DataFrame:
    rows = []
    with pdfplumber.open(RAW / "2024 cut off distances.pdf") as pdf:
        for page in pdf.pages:
            for t in page.extract_tables():
                for r in t:
                    r = [(c or "").replace("\n", " ").strip() for c in r]
                    if re.fullmatch(r"\d{4}", r[0]):       # skips repeated header rows
                        rows.append(r[:5])
    df = pd.DataFrame(rows, columns=["dfe", "name_pdf", "distance_raw", "last_criteria", "criteria_desc"])
    df["dfe"] = df["dfe"].astype(int)
    raw = df["distance_raw"]
    df["all_admitted"] = raw.str.fullmatch(r"All applicants( admitted)?")
    df["cutoff_flag"] = "numeric"
    df.loc[df["all_admitted"], "cutoff_flag"] = "all_admitted"
    num = pd.to_numeric(raw, errors="coerce")
    # non-standard rows ("Criteria 3 = All applicants admitted & Criteria 4 = 0.542"):
    # keep the single number in the cell, flag the row for a sensitivity check
    odd = num.isna() & ~df["all_admitted"]
    num[odd] = raw[odd].str.extract(r"(\d+\.\d+)")[0].astype(float)
    df.loc[odd, "cutoff_flag"] = "nonstandard"
    df["cutoff_miles"] = num
    df["cutoff_m"] = df["cutoff_miles"] * 1609.344
    assert df["dfe"].is_unique
    return df


def best_match(name: str, choices: list[str]):
    """Return (best choice, score, runner-up score) using the 'core' names."""
    cores = [core(c) for c in choices]
    res = process.extract(core(name), cores, scorer=fuzz.token_sort_ratio, limit=2)
    (c1, s1, i1), (_, s2, _) = res[0], res[1]
    return choices[i1], s1, s2


def build_key(verbose=True) -> pd.DataFrame:
    sc, sv, pdf = load_schools(), load_survey(), parse_cutoff_pdf()
    manual = pd.read_csv(MANUAL, dtype={"name_survey": str}) if MANUAL.exists() else pd.DataFrame(
        columns=["EstablishmentNumber", "name_survey", "pdf_dfe", "note"])
    manual = manual.set_index("EstablishmentNumber")

    survey_names = sv["School Name"].tolist()
    norm_map = {}
    for n in survey_names:
        norm_map.setdefault(normalise(n), []).append(n)

    out, review = [], []
    for _, s in sc.iterrows():
        eid, name = s["EstablishmentNumber"], s["EstablishmentName"].strip()
        rec = {"EstablishmentNumber": eid, "name_school_list": name, "primary": s["primary"],
               "TypeOfEstablishment": s["TypeOfEstablishment"]}

        # ---- survey
        cand, score, s2 = best_match(name, survey_names)
        if eid in manual.index and pd.notna(manual.loc[eid, "name_survey"]):
            m = manual.loc[eid, "name_survey"]
            assert m == "NO_MATCH" or m in survey_names, m
            rec.update(name_survey=None if m == "NO_MATCH" else m,
                       survey_match_method="manual", survey_match_score=score)
        elif name in survey_names:
            rec.update(name_survey=name, survey_match_method="exact", survey_match_score=100)
        elif len(norm_map.get(normalise(name), [])) == 1:
            rec.update(name_survey=norm_map[normalise(name)][0], survey_match_method="normalised",
                       survey_match_score=100)
        elif score >= AUTO_ACCEPT and score - s2 >= 5:
            rec.update(name_survey=cand, survey_match_method="fuzzy", survey_match_score=score)
        else:
            rec.update(name_survey=None, survey_match_method="unmatched", survey_match_score=score)
            review.append({"EstablishmentNumber": eid, "school_list": name, "best_survey": cand,
                           "score": round(score), "runner_up": round(s2)})

        # ---- cut-off PDF (by DfE number; manual override for changed numbers)
        pdf_dfe = eid
        if eid in manual.index and not pd.isna(manual.loc[eid, "pdf_dfe"]):
            pdf_dfe = int(manual.loc[eid, "pdf_dfe"])
            rec["pdf_match_method"] = "manual"
        else:
            rec["pdf_match_method"] = "dfe_number"
        hit = pdf[pdf["dfe"] == pdf_dfe]
        if len(hit):
            rec.update(pdf_dfe=pdf_dfe, name_pdf=hit["name_pdf"].iloc[0])
            rec["pdf_name_score"] = fuzz.token_sort_ratio(core(hit["name_pdf"].iloc[0]), core(name))
        else:
            rec.update(pdf_dfe=None, name_pdf=None, pdf_name_score=None, pdf_match_method="unmatched")
        out.append(rec)

    key = pd.DataFrame(out)
    # a survey school claimed by two listed schools: keep exact/manual, send fuzzy to review
    dup = key["name_survey"].notna() & key["name_survey"].duplicated(keep=False) \
        & key["survey_match_method"].isin(["fuzzy", "normalised"])
    for _, r in key[dup].iterrows():
        review.append({"EstablishmentNumber": r["EstablishmentNumber"], "school_list": r["name_school_list"],
                       "best_survey": r["name_survey"], "score": round(r["survey_match_score"]),
                       "runner_up": "DUPLICATE"})
    key.loc[dup, ["name_survey", "survey_match_method"]] = [None, "unmatched"]
    used = key["name_survey"].dropna()
    assert used.is_unique, f"survey name used twice: {used[used.duplicated()].tolist()}"

    if verbose:
        print("=== Survey matching")
        print(key["survey_match_method"].value_counts().to_string())
        fz = key[key["survey_match_method"] == "fuzzy"]
        if len(fz):
            print("\naccepted fuzzy matches (score >= %d):" % AUTO_ACCEPT)
            print(fz[["EstablishmentNumber", "name_school_list", "name_survey", "survey_match_score"]]
                  .round(0).to_string(index=False))
        if review:
            print("\nNEEDS MANUAL REVIEW (not accepted):")
            print(pd.DataFrame(review).to_string(index=False))
        print("\n=== Cut-off PDF matching")
        print(key["pdf_match_method"].value_counts().to_string())
        low = key[key["pdf_name_score"].fillna(100) < 70]
        print("number matches whose names look different (token score < 70) - checked by eye:")
        print(low[["EstablishmentNumber", "name_school_list", "name_pdf", "pdf_name_score"]]
              .round(0).to_string(index=False))
        unused_pdf = pdf[~pdf["dfe"].isin(key["pdf_dfe"].dropna().astype(int))]
        print("PDF rows not linked to any listed school:")
        print(unused_pdf[["dfe", "name_pdf", "distance_raw"]].to_string(index=False))
        unused_sv = sorted(set(survey_names) - set(used))
        print(f"\nsurvey schools not linked to any listed school: {len(unused_sv)}")
    return key, sv, pdf


if __name__ == "__main__":
    key, sv, pdf = build_key()
    cols = ["EstablishmentNumber", "name_school_list", "name_survey", "name_pdf", "primary",
            "survey_match_method", "survey_match_score", "pdf_dfe", "pdf_match_method", "pdf_name_score"]
    key[cols].to_csv(OUT / "school_key.csv", index=False)
    pdf.to_csv(OUT / "cutoff_parsed.csv", index=False)
    print("\nwrote outputs/school_key.csv and outputs/cutoff_parsed.csv")
