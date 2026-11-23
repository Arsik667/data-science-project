# Stage 5 – Fairness: method and status

All numbers quoted here are printed by `src/stage5_fairness.py` (log: `outputs/stage5_log.txt`).

## Principle

- Religion, and every other protected characteristic, is **not** a model input and is never used to change the estimates.
- It is used only to **audit** them.
- The estimates feed insurance pricing. The risk is therefore *proxy discrimination*: an input that is allowed, such as geography or car access, still produces estimates that are systematically wrong for a protected group.

## What was run (no religion data needed)

| Check | Data | Result |
|---|---|---|
| A. Error by school denomination | Survey (validation only) + school names | Catholic schools: mean error −0.167, 95% CI [−0.234, −0.101], n = 23. CofE: −0.001 [−0.049, 0.048], n = 29. No designation: +0.018 [−0.005, 0.039], n = 142. Kruskal–Wallis p < 0.0001. |
| A2. Controlling for catchment car access | OLS + bootstrap (seed 42) | Catholic gap −0.199 [−0.258, −0.143]: the gap is **not** explained by richer catchments. |
| B. Error by catchment car-access quintile | Census car access (proxy for income) | Over-estimate in the lowest quintile (+0.082), under-estimate in the highest (−0.093). Spearman −0.396. |
| C. Errors propagated to postcodes | Attendance matrix × school errors | 24.0% of children live in postcodes with \|propagated error\| > 0.10. Low-car postcodes lean towards over-estimates (Spearman −0.532). |

**Data note.** The `ReligiousEthos` field in `school locations.csv` marks none of the 27 Catholic schools (by name) as Catholic, and only 3 of the 37 Church of England schools. Denomination is therefore read from the school name.

### Interpretation

- **Catholic schools.** Catholic schools admit by faith first, so their pupils come from further away and are driven more: observed 52% against 36% predicted. Method 2 allocates by distance only, so it misses this.
  - Children at Catholic schools are under-counted as car users.
  - Because Catholic families are more likely to be Catholic, the error is plausibly correlated with religion. That is a proxy-discrimination risk, even though religion is not an input.
- **Church of England schools.** Most are community schools with local admissions, and they show no bias.
- **Car access.** The model is too flat (Stage 4), so it over-states car use in low-car-access, typically poorer areas and under-states it in high-car-access areas. In pricing terms, poorer areas would carry too much school-run exposure.
- **Caveat for C.** The model places Catholic-school pupils in postcodes *near* the school, but the real pupils live further away. The propagated-error map shows where estimates are least reliable; it does not show which households are mis-priced.

## What was NOT run: religion by OA

The file `religion by oa.csv` was not provided, so this check was **not run**. The code is ready: `religion_check()` in `src/stage5_fairness.py`. It runs automatically when the file is placed in `data/raw/`, and it is tested on synthetic data in `tests/test_stage5.py`.

**Expected input:** Census 2021 TS030 (religion) by OA. That means one OA-code column plus person counts for each category: No religion, Christian, Buddhist, Hindu, Jewish, Muslim, Sikh, Other religion, Not answered. The OA column is detected automatically, and "Total" columns are ignored.

**Steps**

1. Compute religion shares per OA from the counts.
2. Aggregate the postcode estimates to OA: children, car-driven children, and the children-weighted propagated error.
3. For each religion group g, compute a children-weighted mean estimated car rate:
   Σ_OA share_g · kids · car_rate_est / Σ_OA share_g · kids.
   This is an *ecological* assumption: children in an OA are assumed to share the OA's religious mix.
4. Report a **disparity ratio** for each group: the group mean divided by the all-children mean, where 1 means parity. The 0.8 "four-fifths" rule can serve as a flag, not a test.
5. Compute Spearman correlations across OAs between each group's share and:
   - the estimated car rate, which shows whether estimates differ by group;
   - the propagated survey error, which shows whether estimates are *wrong* differently by group.

   The second correlation is the fairness test that matters. A difference in estimates can be legitimate if car use really differs. A difference in error cannot.

**Limitations of the planned check**

- Ecological fallacy: OA shares are not individual religion.
- The survey is at school level, not OA level.
- The OA's whole population is not its school-age families.

## Recommendations (for the report)

1. Do not use Method 2 estimates for pricing without a correction for schools that do not admit by distance (faith schools, highly selective academies). A defensible approach is an **admission-type** feature, which is a school attribute rather than a family attribute. Any such feature must be tuned on held-out schools, never on the whole survey.
2. Report errors by car-access quintile and by school denomination every time the model is refreshed. These are fairness monitoring metrics.
3. Run `religion_check()` when the census religion table is available. Ask the client whether ethnicity (TS021) should be checked the same way.
