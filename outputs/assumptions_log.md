# Assumptions log

Every number below is printed by the script of the stage in the "Stage" column (full printouts: `outputs/stageN_log.txt`).
Status: written before the main brief was available — re-check A1 (car access) and A8 (Method 1) against the brief.

| ID | Stage | Assumption | Reason | Effect on results |
|----|-------|------------|--------|-------------------|
| A1 | 1 | **Car access** = household has 1 or more cars/vans. `car_rate = 1 − "No cars" / (sum of the four car categories)` | Suggested by the Data Source Guide; one car is enough for a school run. The "Does not apply" category is 0 in every OA. | Household-weighted car_rate = 0.7128; OA range 0.024–1.000. |
| A2 | 1 | Denominator of car_rate = total of the car table, **not** the households file | Numerator and denominator come from one table, so the rate stays in [0, 1]. The two totals differ by −6…+5 per OA (1,235 OAs differ) because of census disclosure-control noise. | Totals 341,299 vs 341,359 households (difference 60, under 0.02%) — negligible. |
| A3 | 1 | **Primary age** = Reception to Year 6 on Census day (21 Mar 2021), i.e. ages 4y7m to 11y6m. Fractions: 1/12 of "0–4", all of "5–9", 19/60 of "10–14" | Exactly 7 school cohorts; assumes ages are evenly spread within each 5-year band. | 68,785 children. A flat 0.2/1/0.2 split gives 68,625 (−0.2%), so the choice barely matters for the total. |
| A4 | 1 | Each OA's children and households are split **equally** between the OA's postcodes | No postcode-level population is provided; household counts exist only per OA, so "split by households" is not possible inside an OA. Avoids giving each postcode the full OA count (about 8x inflation). | Sum is preserved exactly (tested). Local smearing: OAs with many non-residential postcodes (max 139 postcodes in one OA) spread their children thinly across them. |
| A5 | 1 | Census 2021 age structure is used for the 2024/25 school year | It is the only age data provided. | Unknown cohort change over 4 years; a limitation. |
| A6 | 1 | Sanity benchmark = rolls of the 223 listed schools excluding Roundhay School and Carr Manor (all-through) | Those two include secondary pupils. | Estimate / roll = 0.960 (68,785 / 71,682). 3 schools have no roll, and the list may miss schools or include pupils from outside Leeds, so a small gap is expected. |
