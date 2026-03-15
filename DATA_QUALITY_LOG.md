# Data Quality Log — Lab 1: F1 Top 10 Prediction (2022–2024)

**Team:** feligna  
**Sources:** Jolpica API (ergast), FastF1 SDK  
**Date:** March 14, 2026

---

## Issue 1: Grid Position = 0 (Pit Lane Start)

- **What:** The Jolpica API returns `grid='0'` for drivers who start from the pit lane (e.g., technical infringement penalty). This is a structural encoded value, not a missing value.
- **Where Found:** `results['grid']` column after API ingestion
- **Classification:** Data-encoding issue (MAR — Missing At Random, since not random; structure-dependent)
- **Frequency:** ~175 rows (~10% of 1700 rows in raw extract)
- **Impact:** If left as integer 0, would be interpreted as a legitimate pole position, causing incorrect modeling assumptions. Grid position 0 is not a valid F1 grid slot (valid range is 1–20).
- **Decision:** Convert all `grid='0'` to `NaN` during ingestion, then drop these rows in the modeling dataset for baseline rule (grid <= 10).
- **Justification:**
  - These are non-standard starts, not comparable to the normal grid positions (1–20).
  - For the baseline rule, these drivers are excluded because we cannot apply the `grid <= 10` heuristic to pit-lane starts without domain knowledge.
  - Dropping is valid because: (a) n is small (~10%), (b) baseline is simple one-feature rule, (c) pit-lane starters are rare edge cases.
  - In future models (Lab 2), these could be handled with a separate category (`grid_tier = 'pit_lane'`).

---

## Issue 2: Missing Position (Race Outcome)

- **What:** A small number of rows have missing values in the `position` field, which is the final race classification (1–classified, >25 or NaN = not classified).
- **Where Found:** `results['position']` column
- **Classification:** MNAR (Missing Not At Random) — missingness is outcome-dependent; drivers with no classification are those who did not finish or were excluded.
- **Frequency:** ~15 rows (~0.9% of raw extract)
- **Impact:** Cannot directly compute the target variable `top_10 = (position <= 10)` for these rows. Leaving as NaN risks dropping data; assuming position=999 risks wrong classification.
- **Decision:** Fill missing `position` with `999` (a sentinel value > 10), then map to `top_10 = 0` (not in top 10). Keep these rows in the dataset.
- **Justification:**
  - Drivers with missing position were not classified, so they definitely did not finish in top 10.
  - Using position=999 is a domain-appropriate encoding: any position > 10 → not top 10.
  - This avoids data loss while ensuring correct target assignment.

---

## Issue 3: Status Field — Leakage (Post-Race Information)

- **What:** The `status` column contains the Final Classification (e.g., "Finished", "+1 Lap", "DNF", "Accident", "Engine", etc.). This is only available **after the race**, making it post-race information.
- **Where Found:** `results['status']` column
- **Classification:** Leakage — not a data quality defect, but a modeling design issue.
- **Frequency:** 100% of rows have a status value
- **Impact:** Using `status` in any prediction model would constitute data leakage because we are using information from the race outcome to predict whether the driver finishes in the top 10.
- **Decision:** **Never use `status` as a predictive feature.** Flag it in feature audit (Section 2 of eda1.ipynb and baseline.ipynb).
- **Justification:**
  - `status` is revealed only during/after the race; we must predict before the race starts.
  - If `status == 'DNF'` strongly predicts `top_10 = 0`, that is circular reasoning.
  - Keep it in the dataset for reference and EDA, but exclude from any prediction pipeline.

---

## Issue 4: DOB (Date of Birth) — Sparse/Occasional Missing Values

- **What:** The `dob` field, recovered from Jolpica API, has occasional missing values. These are used to compute `age_at_race = season - dob.year`.
- **Where Found:** `results['dob']` column; downstream impact on `results['age_at_race']`
- **Classification:** MCAR (Missing Completely At Random) — likely metadata gaps in the API source, not related to performance or driver importance.
- **Frequency:** <1% in most years (varies by build)
- **Impact:** Rows with missing DOB cannot be used if age is a feature. For the baseline, age is not used, so impact is minimal.
- **Decision:** Keep DOB as-is (do not impute). For any future model that uses age, drop rows with missing DOB or use median imputation (mode imputation if age is categorical).
- **Justification:**
  - Missing rate is very low, drop-based cleanup is acceptable.
  - Imputation would require assumptions (e.g., rookie age, average driver age) that may not hold.
  - The baseline rule (`grid <= 10`) is age-independent, so this is not a blocker.

---

## Issue 5: Laps — Session/Circuit-Dependent Outliers

- **What:** The `laps` field (number of laps completed) has bimodal and wide variation driven by circuit length and race incidents. IQR-based outlier detection flags ~94 rows (~6.9% of 1,354 cleaned rows) as statistical outliers.
- **Where Found:** `results['laps']` column; flagged in Outlier Audit (Section 2.1 of eda1.ipynb)
- **Classification:** Outlier event, not a data error — represents real race dynamics (DNFs, red flag interruptions, lap-count variations).
- **Frequency:** ~94 outliers in cleaned dataset
- **Impact:** Minimal for the baseline rule (laps is not a baseline feature). Potential impact for future models: if laps predicts Top 10, outlier laps could skew coefficients without careful regularization.
- **Decision:** **Keep without clipping or transformation.** Do not winsorize or remove. Document as race-context variation, not measurement error.
- **Justification:**
  - Outlying lap counts represent real events (e.g., Monza's long races, Baku's mid-race red flags).
  - These are not errors; they are signal about circuit and race conditions.
  - For ML models, regularization and tree-based methods can handle these naturally.
  - For the simple baseline, laps is irrelevant anyway.

---

## Issue 6: Survivorship Bias — Driver Appearance Frequency

- **What:** Drivers with more race appearances (higher `n_races`) in the 2022–2024 dataset have statistically significantly higher Top 10 rates (Pearson r = 0.470, p = 0.0116). This suggests drivers who competed longer were retained because they were stronger.
- **Where Found:** Identified in Question 5 of eda1.ipynb via cross-tabulation of driver stats and top_10 rate.
- **Classification:** Selection bias (survivor bias) — not a column-level data quality issue, but a structural issue in how the data is sampled.
- **Frequency:** All ~157 drivers affect this bias to varying degrees.
- **Impact:** Using `n_races` as a feature would confound the model: apparent success of high-appearance drivers is partly due to retention (team kept them because they were good) rather than pure race-level skill. Models would learn "more races = better outcome" instead of "better preparation = better outcome."
- **Decision:** **Do not use `n_races` or any driver-aggregate feature in the baseline.** Keep baseline focused on **race-level pre-race features** (e.g., grid, constructor tier). For Lab 2 models that use driver-level aggregates, control for this bias explicitly (e.g., by stratifying or weighting).
- **Justification:**
  - This bias is real and explains ~22% of variance in Top 10 rate (r² = 0.22), significantly more than grid's ~3%.
  - However, it is not predictive for future races; it is historical selection.
  - The baseline rule (`grid <= 10`) avoids this by using race-level position, not driver history.

---

## Issue 7: FP3 → Qualifying Improvement (Question 3) — Distribution Mismatch: Mean vs. Median

- **What:** In 2024, the mean improvement (FP3 best lap minus Qualifying best lap) is **higher for Non-Top 10 drivers** (1.410s) than for Top 10 drivers (1.162s). However, the Spearman correlation is still **positive** (r = 0.228, p = 0.003), suggesting the relationship holds.
- **Where Found:** Question 3, Sections 3.1–3.3 of eda1.ipynb (FastF1 session-level data)
- **Classification:** Distribution asymmetry / outlier influence — not a data error, but a signal that the data is non-normal or contains outliers.
- **Frequency:** Localized to 2024 (n = 159); not present in 2022–2023.
- **Coverage:** 842 total driver-race observations across 3 seasons; 0 NaN values after merge (high data quality).
- **Impact:** The feature `improvement_s` is **weakly predictive** (r between 0.17–0.23 across years), and its relationship is **sensitive to distribution shape** (mean/median divergence in one year). This makes it unreliable as a standalone rule or primary feature.
- **Decision:** **Keep `improvement_s` as a secondary candidate feature, NOT as a primary baseline driver.** Use `grid` as the main baseline. In future models (Lab 2), allow regularized or tree-based methods to decide `improvement_s`'s weight with robust loss functions.
- **Justification:**
  - The positive correlation is real and consistent across 2022, 2023, 2024 (p < 0.01 in all years).
  - However, effect size is weak (r < 0.25 ≈ 3–5% variance explained), not strong enough for a simple decision rule.
  - Mean/median divergence in 2024 suggests outliers; Spearman is robust to outliers, so correlation is more trustworthy than mean.
  - Keeping in feature pool for future exploration is low-risk and may yield signal in multivariate models.

---

## Issue 8: Constructor Tier — Confounding of Grid Effect

- **What:** Constructor tier (Top 4 vs. Other) is highly predictive of Top 10 (82.0% for Top 4, 29.1% for Other). When grid is analyzed separately within each tier, the within-tier grid effect is weaker in Top 4 teams (r = -0.223) than in Other teams (r = -0.441). This indicates **confounding**: part of the grid effect is actually due to team quality.
- **Where Found:** Question 4, Section 4.3 of eda1.ipynb
- **Classification:** Confounding / collider — not a column-level defect, but a causal structure issue.
- **Frequency:** All ~1,344 rows (after grid cleaning) are affected.
- **Impact:** The simple baseline rule (`grid <= 10`) conflates grid position with team strength. A driver starting P11 for Mercedes may have higher Top 10 probability than a driver starting P10 for a mid-field team. The rule is still useful as a lower bound but is not fully causal.
- **Decision:** **Keep grid-based baseline for Lab 1 simplicity.** Document the confounder. For Lab 2, test a compound rule like "if grid <= 10 AND constructor_tier = 'Top 4' → Top 10" or use multivariate models that can separate the effects.
- **Justification:**
  - Grid is still strongly predictive (Spearman r = -0.571 overall), so the baseline is useful.
  - Confounding does not invalidate the baseline; it just explains why the effect is inflated.
  - For a first baseline, simplicity is valuable. Introducing `constructor_tier` would increase rule complexity without guaranteed improvement.
  - The data audit documented the issue, so future analysts are aware and can refine.

---

## Summary Table: Data Quality Issues & Decisions

| Issue                      | Type                         | Where                    | Rows Affected   | Decision                        | Rigor  |
| -------------------------- | ---------------------------- | ------------------------ | --------------- | ------------------------------- | ------ |
| 1. Grid = 0                | Encoding (MAR)               | `results['grid']`        | ~175 (10%)      | Convert to NaN → drop           | High   |
| 2. Position missing        | Outcome-dependent (MNAR)     | `results['position']`    | ~15 (0.9%)      | Fill 999 → top_10=0             | High   |
| 3. Status field            | Leakage (post-race)          | `results['status']`      | ~1,700 (100%)   | EXCLUDE from prediction         | High   |
| 4. DOB missing             | Sparse (MCAR)                | `results['dob']`         | ~5 (<1%)        | Keep; drop if needed in future  | Medium |
| 5. Laps outliers           | Event-based (real variation) | `results['laps']`        | ~94 (6.9%)      | Keep without clipping           | Medium |
| 6. Survivorship bias       | Selection bias (structural)  | Driver-level aggregates  | All drivers     | EXCLUDE `n_races` from baseline | High   |
| 7. Improvement_s (2024)    | Distribution asymmetry       | `improvement_s` (FastF1) | 159 (2024 only) | Keep as secondary feature       | Medium |
| 8. Constructor confounding | Confounding / collider       | interaction(grid, team)  | All rows        | Document; use in Lab 2          | Medium |

---

## Appendix: Data Sources & API Notes

### Jolpica (Ergast API)

- **Endpoint:** `https://api.jolpi.ca/ergast/f1/{year}/results.json`
- **Rate limit:** 1 request/second (enforced in code)
- **Retry logic:** Exponential backoff (2s, 4s, 8s) on failure
- **Data recovered:**
  - Season, round, race date, race name, circuit, country
  - Driver ID, code, name, DOB, nationality
  - Constructor, grid, position, status, points, laps

### FastF1 SDK

- **Data source:** Official F1 telemetry and session data (via ergast + embedded APIs)
- **Cache:** Local directory (`data/fastf1_cache/`) to avoid re-fetching
- **Session types used:** FP3 (Free Practice 3), Q (Qualifying)
- **Data recovered:**
  - Driver code (3-letter code, e.g., "HAM")
  - Lap times as timedeltas (converted to seconds)
  - Best lap per driver per session

### Known API Quirks

1. Jolpica sometimes delays new races; 2024 data coverage is incomplete (159 rows for ~23 races).
2. FastF1 requires internet/cache; offline mode is not supported.
3. Grid='0' is Jolpica's encoding for pit-lane starts; it is not a parsing error.

---

## Metadata

- **EDA Notebook:** eda1.ipynb (Sections 2.1–2.2 and Questions 3–5)
- **Baseline Notebook:** baseline.ipynb (Sections 1–2)
- **Merged Dataset:** `results_clean` (1,344 rows after grid cleaning)
- **Train/Val/Test:** 2022–2023 / Jan–May 2024 / Jun–Dec 2024 (temporal split)

---

**Completed by:** feligna  
**Date:** March 14, 2026  
**Status:** ✓ Ready for Lab 1 submission
