# Stage 6 – Limitations and sensitivity

Every number below is printed by a script. The source log is named in each row.

## 1. Headline: the level is uncertain, the ranking is fairly stable

| Quantity | Result | Source |
|---|---|---|
| Citywide car-driven share, main model | 35.4% (24,319 children) | stage3_log |
| Same, 90% Monte Carlo band (λ, d50, scale drawn jointly) | 7.9% – 54.1% (p50 29.6%) | stage6_log |
| Main driver of the level | d50 (Spearman with share −0.776), then λ (+0.620); scale negligible (+0.046) | stage6_log |
| Postcode-rank Spearman vs main across draws | median 0.930, p5 0.828, min 0.743 | stage6_log |
| Main top-10% postcodes that stay top-10% in ≥ 90% of draws | 64.2% | stage6_log |
| Postcode 90% band width (children-weighted) | median 47.0 pp | stage6_log |

**Interpretation**

- The model can say *which* postcodes have more school-run car use. It cannot say *how much* there is, because no input identifies d50.
- Changing one assumption at a time (stage6_log, `sensitivity_one_at_a_time.csv`):
  - λ and d50 move the total by 15–23 pp, but keep the postcode ranking (Spearman 0.84–0.93).
  - Route detour 1.3, g(d) steepness, capacity weight, capacity cap, all-through share and age fractions move the total by under 2 pp, and the ranking Spearman stays at or above 0.96.
- **Caveat.** This stability is the model agreeing with itself. It is not accuracy. Against the survey, the school ranking is only Spearman 0.50 (Stage 4).
- The Monte Carlo median (29.6%) is below the main value because the d50 range (500–2,000 m) lies mostly above the main 1,000 m. The ranges are judgement calls (A27).

## 2. Limitations, most important first

| # | Limitation | Evidence | Likely direction | What would fix it |
|---|---|---|---|---|
| L1 | The walk/drive threshold g(d) is assumed, not estimated | d50 changes the total from 53.0% (500 m) to 12.2% (2,000 m) | Unknown | Pupil-level mode data, or an external benchmark agreed with the client (not the validation survey) |
| L2 | Attendance depends on distance only; one λ for all schools | Cut-off Spearman −0.06; oversubscription AUC 0.475; λ weakly identified (folds choose 550 vs 1,000 m) | Catchments of popular and oversubscribed schools too wide; those of undersubscribed schools too narrow | Admissions data (applications, preferences); school-specific attractiveness tuned on held-out schools |
| L3 | Faith and other non-distance admissions are ignored | Catholic schools: error −0.167 [−0.234, −0.101] | Under-estimates car use for those families: a proxy-discrimination risk | Admission-type feature; see fairness_method.md |
| L4 | Too little spread between schools | sd of prediction 0.072 vs 0.177 observed; error falls with catchment car access (Spearman −0.396) | Over-estimates in low-car (poorer) areas, under-estimates in high-car areas | Car access of households **with children** (census table by household composition), not of all households |
| L5 | car_rate is for all households | Car table covers all households, including students and older people | Probably under-states car access for families in the city centre | As L4 |
| L6 | Equal split of OA children across postcodes | Up to 139 postcodes in one OA (stage1) | Smears children into non-residential postcodes; the OA total is right | Postcode-level household counts (e.g. address base) |
| L7 | Census 2021 ages used for school year 2024/25 | Estimate / primary roll = 0.934 (stage3) | Cohorts have changed size; about 7% fewer children than on the rolls | Mid-year population estimates for 2024 |
| L8 | Only the 225 listed schools; Leeds boundary ignored | 2,637 children in WF and 787 in BD postcodes (stage6_log, part C); 117 children live > 2 km from the nearest listed school | Border postcodes are sent to Leeds schools even if they attend schools outside Leeds | School lists for neighbouring authorities |
| L9 | Independent, special and home-educated children not separated | All estimated children go to state primary schools | Overstates state-school pupils slightly | Independent and SEN school rolls |
| L10 | Straight-line distance | A route detour of 1.3 changes the total by +1.9 pp and keeps ranking Spearman at 0.961 | Small | Network distances (OSRM) |
| L11 | Survey is self-reported, one day, 2025 | Median response rate 1.01 of the roll (stage4) | Good coverage; unknown day-to-day and weather variation | More survey years |
| L12 | Trip chaining, siblings, and parents driving on to work are not modelled | — | Distance matters less for car use than assumed | Out of scope |
| L13 | No brief available; definitions of Method 1/2 and the target are our own | — | — | Check against the brief once it arrives |

## 3. How the client should use the estimates

1. Use `est_car_rate` as a **relative** score (rank or decile) rather than an absolute count. The rank is far more robust than the level.
2. Ship the uncertainty band with the estimates: `estimates_postcode_uncertainty.csv` gives p5/p50/p95 per postcode and the share of draws in which the postcode is top-10% risk.
3. If an absolute level is needed, anchor d50 on an external source agreed with the client. If the 2025 survey is used for this, hold out half of the schools and report the result on the other half (prompt constraint 1).
4. Monitor errors by school denomination and by car-access quintile at every refresh (Stage 5).
