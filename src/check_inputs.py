"""
check_inputs.py - re-verify the facts we were told about the raw input files.

Run from the project folder:
    .venv/bin/python src/check_inputs.py

Nothing here is modelling: it only loads each file and prints counts so that
every number we quote later comes from code output.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"

# ---------------------------------------------------------------- postcodes
pc = pd.read_csv(DATA / "postcode locations.csv", dtype={"POSTCODE": str, "OA": str})
print("=== postcode locations.csv")
print("rows:", len(pc), "| unique postcodes:", pc.POSTCODE.nunique(),
      "| unique OAs:", pc.OA.nunique())
print("NaN per column:", pc.isna().sum().to_dict())
print("postcodes per OA: mean %.2f, min %d, max %d"
      % (pc.groupby("OA").size().mean(), pc.groupby("OA").size().min(), pc.groupby("OA").size().max()))
print("postcode areas:", pc.POSTCODE.str.split().str[0].str.extract(r"^([A-Z]+)")[0].value_counts().to_dict())
oas = set(pc.OA)

# ---------------------------------------------------------------- schools
sc = pd.read_csv(DATA / "school locations.csv")
print("\n=== school locations.csv")
print("rows:", len(sc), "| unique EstablishmentNumber:", sc.EstablishmentNumber.nunique())
print("missing NumberOfPupils:", sc.loc[sc.NumberOfPupils.isna(), "EstablishmentName"].tolist())
print("missing SchoolCapacity:", sc.SchoolCapacity.isna().sum())
over = sc[sc.NumberOfPupils > sc.SchoolCapacity]
print("pupils > capacity:", len(over))
print("type counts:", sc["TypeOfEstablishment (name)"].value_counts().to_dict())
print("largest rolls:\n", sc.nlargest(6, "NumberOfPupils")[["EstablishmentNumber", "EstablishmentName",
                                                            "SchoolCapacity", "NumberOfPupils"]].to_string(index=False))
print("total roll (all):", sc.NumberOfPupils.sum())
big = sc.EstablishmentName.str.contains("Roundhay School|Carr Manor", regex=True)
print("total roll without Roundhay School & Carr Manor:", sc.loc[~big, "NumberOfPupils"].sum(),
      "| excluded:", sc.loc[big, "EstablishmentName"].tolist())

# ---------------------------------------------------------------- car access
car = pd.read_csv(DATA / "access to car by oa.csv")
car.columns = ["OA", "cat", "n"]
print("\n=== access to car by oa.csv")
print("OAs in file:", car.OA.nunique(), "| our OAs missing from file:", len(oas - set(car.OA)))
car = car[car.OA.isin(oas)]
print("'Does not apply' total in our OAs:", car.loc[car.cat == "Does not apply", "n"].sum())
car_w = car.pivot(index="OA", columns="cat", values="n")
car_total = car_w.sum(axis=1)

# ---------------------------------------------------------------- households
hh = pd.read_csv(DATA / "number of households by oa.csv")
hh.columns = ["OA", "households"]
print("\n=== number of households by oa.csv")
print("OAs in file:", hh.OA.nunique(), "| our OAs missing:", len(oas - set(hh.OA)))
hh = hh[hh.OA.isin(oas)].set_index("OA").households
diff = (car_total - hh.reindex(car_total.index))
print("car-category sum minus households: min %d, max %d, mean abs %.2f, OAs with diff != 0: %d"
      % (diff.min(), diff.max(), diff.abs().mean(), (diff != 0).sum()))
print("total households in our OAs:", hh.sum())
no_car = car_w["No cars or vans in household"]
print("households with >=1 car / car-category total: %.4f" % (1 - no_car.sum() / car_total.sum()))

# ---------------------------------------------------------------- age bands
age = pd.read_csv(DATA / "agebands by oa.csv")
age.columns = ["OA", "band", "n"]
print("\n=== agebands by oa.csv")
print("OAs in file:", age.OA.nunique(), "| our OAs missing:", len(oas - set(age.OA)))
age = age[age.OA.isin(oas)]
tot = age.groupby("band").n.sum()
for b in ["Aged 4 years and under", "Aged 5 to 9 years", "Aged 10 to 14 years"]:
    print(f"{b}: {tot[b]}  (per single year of age ~ {tot[b] / 5:.0f})")
print("total population in our OAs:", tot.sum())

# ---------------------------------------------------------------- survey
# file is Windows-1252 encoded (curly apostrophes), not UTF-8
sv = pd.read_csv(DATA / "2025 travel to school survey.csv", encoding="cp1252")
print("\n=== 2025 travel to school survey.csv")
print("rows:", len(sv), "| last row:", sv.iloc[-1, 0])
body = sv[sv["School Name"] != "Grand Total"]
gt = sv[sv["School Name"] == "Grand Total"].iloc[0]
print("schools:", len(body), "| sum Car/Van (schools only):", body["Car/Van"].sum(),
      "| Grand Total row Car/Van:", gt["Car/Van"])
print("sum Grand Total column (schools only):", body["Grand Total"].sum(), "| Grand Total row:", gt["Grand Total"])
exact = set(body["School Name"].str.strip()) & set(sc.EstablishmentName.str.strip())
print("exact name matches with school list:", len(exact))
