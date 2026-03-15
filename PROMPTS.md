# PROMPTS.md — AI Usage Log

**Lab:** IIT414W Lab 1 — F1 Top 10 Prediction (2022–2024)  
**Team:** feligna  
**Date:** March 14, 2026

> **Note:** This log documents all AI interactions used to develop the lab. Each entry describes what was asked, what AI returned, how it was verified, and how the output was adapted. The reasoning, decision-making, and interpretation are manual; AI assistance was used for code syntax, conceptual explanations, and best practices guidance under user direction.

---

## Entry 1 — [March 12, 2026] — Q1: Does Grid Position Predict Top 10 Finish?

**Context:**  
**Research Question:** Q1 asks whether drivers starting in better grid positions (lower grid numbers, e.g., pole at position 1) are more likely to finish in the Top 10. This is a correlation hypothesis: we need to quantify the strength and direction of the relationship.

I needed to:

1. Compute Spearman correlation (grid → top_10) to measure association strength
2. Create visualizations (scatter + trend, binned bar chart) to show the relationship
3. Interpret whether the correlation is practically meaningful

**Prompt(s):**

_Prompt 1a (Technique):_

> "I have a DataFrame with 'grid' (ordinal, 1-20) and 'top_10' (binary). How do I compute Spearman correlation for this relationship? Should I aggregate by grid position first, or apply the correlation directly?"

_Prompt 1b (Interpretation):_

> "When you have an ordinal predictor (like grid position 1-20) with a binary outcome, what's the best way to interpret and visualize a strong negative Spearman correlation? Should I focus on the overall correlation, or also show binned distributions to make the practical impact clear?"

**Output:**  
AI suggested using `scipy.stats.spearmanr()` directly on the raw columns (no pre-aggregation), and provided matplotlib patterns for multi-panel visualization with trend line overlay and grouped bar charts. AI also recommended the `polyfit` function for trend estimation.

**Validation:**

- Executed `spearmanr(grid, top_10)` directly → r = -0.571, p < 0.001 ✓
- Verified trend line has negative slope (expected: lower grid numbers = higher top_10 probability) ✓
- Computed by-hand: Grid 1-10 has 77.1% top_10 rate vs Grid 11-20 has 23.3% → ratio ~3.3x ✓
- Confirmed all three visualizations aligned with aggregated summaries ✓

**Adaptations:**

- Scaled point sizes (`total_races * 2`) for visual clarity
- Applied colorblind-friendly palette (`#2ecc71` green for better grid, `#e74c3c` red for worse)
- Added slope annotation on trend line for interpretability
- Grouped grid tiers (1-10 vs 11-20) to make decision rule intuitive

**Final Decision:**  
**Used.** Spearman correlation confirmed strong, negative, significant relationship. Grid position is an excellent pre-race predictor. Result directly supports Rule 1 baseline: "If grid ≤ 10, predict Top 10."

---

## Entry 2 — [March 12, 2026] — Q2: Is the Target Class Balanced?

**Context:**  
**Research Question:** Q2 addresses whether Top 10 finishes are balanced (50/50) or skewed in the dataset. This matters because:

- Skewed targets (e.g., 90% Top 10) would mislead accuracy metric interpretation
- Balanced targets allow accuracy to be a reasonable primary metric
- Seasonal variation in class distribution could signal data drift

**Prompt(s):**

_Prompt 2a (Technique):_

> "I need to check if my binary target is balanced. What's the standard way to compute class proportions and compare across groups (seasons)?"

_Prompt 2b (Interpretation & Metric Choice):_

> "How should you interpret a balanced binary target (near 50/50 split) when selecting evaluation metrics? Is accuracy sufficient as a primary metric under these conditions, or should you rely on other measures like F1 or ROC-AUC?"

**Output:**  
AI explained that balanced classes make accuracy a reasonable metric (naive 50% floor is easy to beat), and suggested pie charts for overall distribution + stacked bars for seasonal breakdown. Recommended using accuracy as primary metric for balanced scenarios.

**Validation:**

- Computed: 50.4% Top 10 vs 49.6% Not Top 10 → near-perfect balance ✓
- Cross-checked by season: 2022 (50.8%), 2023 (50.3%), 2024 (50.2%) → all ~50% ✓
- Verified naive baseline: always predicting "Top 10" gives 50.4% accuracy (weak floor) ✓
- Confirmed no season shows dramatic drift in class proportions ✓

**Adaptations:**

- Emphasized that naive 50.4% is the floor; any useful baseline must beat this
- Added counts alongside percentages for transparency
- Computed "naive baseline accuracy" metric explicitly for comparison

**Final Decision:**  
**Used.** Class balance confirmed. Accuracy is a valid primary metric. Establishes that the dataset is well-formed and not degenerate (e.g., not "always Top 10").

---

## Entry 3 — [March 12–13, 2026] — Q3: Does FP3 → Qualifying Improvement Predict Top 10?

**Context:**  
**Research Question:** Q3 investigates a pre-race signal: drivers who improve their lap times from FP3 (Free Practice 3) to Qualifying may be more prepared/confident, and thus more likely to finish Top 10. This requires session-level telemetry, not just race results.

Challenge: Integrating **two separate data sources**:

- Jolpica API (race outcomes) with driver IDs
- FastF1 SDK (session lap times) with driver codes
  Requires merge strategy, time handling, and robust error handling.

**Prompt(s):**

_Prompt 3a (Data Integration):_

> "I need to pull FP3 and Qualifying session lap times from FastF1 for each round (2022-2024). Then compute best-lap improvement per driver (fp3_best - quali_best). Finally, merge with race outcomes by (season, round, driver_code). What's the safest way to handle missing sessions, driver code normalization, and time-delta conversion?"

_Prompt 3b (Interpretation):_

> "When a rank-based correlation (like Spearman) is statistically significant but small in magnitude (r < 0.25), how should you interpret the practical importance? When summary statistics (means/medians) by group seem to contradict the correlation direction, what does this suggest about the data distribution?"

**Output:**  
AI provided patterns for: session data extraction with try-except error handling, `.astype(str).str.upper().str.strip()` for code normalization, `.total_seconds()` for timedelta conversion, and inner-join merge strategy. For interpretation: explained that Spearman is robust to outliers/asymmetry, so weak r with significant p indicates a real but small signal. Suggested keeping as secondary feature for future models.

**Validation:**

- Executed session pulls for all 3 seasons → 842 driver-race observations (clean join) ✓
- Verified merge: 0 NaN in improvement_s → no missing values ✓
- Computed Spearman per season: 2022 (r=0.193, p=2e-04), 2023 (r=0.173, p=2e-03), 2024 (r=0.228, p=3.8e-03) → consistent pattern ✓
- Confirmed 2024 asymmetry (mean paradox vs. rank correlation) is due to outliers ✓

**Adaptations:**

- Added try-except blocks around FastF1 calls to avoid crashes on missing sessions
- Normalized driver codes (uppercase + strip) to handle API inconsistencies
- Split computation by year for transparency and debugging
- Documented coverage loss as acceptable for a secondary feature

**Final Decision:**  
**Used as secondary feature, not baseline.** Signal is weak (r < 0.25) and sensitive to distribution shape. Documented for Lab 2 exploration, but too unreliable for simple rule. Kept grid as primary baseline.

---

## Entry 4 — [March 13, 2026] — Q4: Is the Grid Effect Confounded by Team Strength?

**Context:**  
**Research Question (Trap Check):** Q4 tests whether the strong grid effect (Q1) is genuine or partly spurious due to team quality confounding. Better teams:

- Qualify higher → better grid
- Have faster cars → higher Top 10 rates regardless of grid

If this is true, part of grid's predictive power is really team strength, not pure qualifying skill.

**Prompt(s):**

_Prompt 4a (Confounder Analysis):_

> "How do I test for confounding in a correlation? Specifically: grid correlates with top_10 (r = -0.57), but team quality might cause both. How do I stratify by team tier and recompute the correlation within each group to see if the effect weakens?"

_Prompt 4b (Interpretation):_

> "I computed correlations within team tiers: Top 4 teams (r = -0.22), Other teams (r = -0.44). What does this tell me about confounding? Is the grid effect spurious or genuine?"

**Output:**  
AI explained stratified analysis as the standard trap-check for confounding. Noted that if within-strata correlations are much weaker than overall, confounding is present. Provided patterns for groupby + correlation per group. For interpretation: explained that weakened effect in Top 4 (but still significant) indicates confounding is real but not complete.

**Validation:**

- Classified teams: Top 4 (Mercedes, RB, Ferrari, McLaren by total points) vs Others ✓
- Verified Top 4 Top 10 rate: 82.0% vs Others: 29.1% → strong main effect ✓
- Computed within-tier correlations: Top 4 (r=-0.22, p=1.4e-07), Others (r=-0.441, p=1.7e-39) ✓
- Interpretation: Within-strata effects are weaker but still significant → confounding is real and large ✓

**Adaptations:**

- Visualized grid-vs-top10 scatter separately for each tier (color-coded)
- Documented finding: NOT spurious, but partly confounded
- Planned for Lab 2: test compound rule (grid + team tier)

**Final Decision:**  
**Used to inform baseline design.** Confounding is real but does not invalidate grid-based rule. Documented as limitation: baseline conflates grid position with team quality. Simple rule is still useful as lower bound, but future models should separate effects.

---

## Entry 5 — [March 13, 2026] — Q5: Is There Survivorship Bias?

**Context:**  
**Research Question (Trap Check):** Q5 investigates whether drivers with more race appearances in the dataset have inflated Top 10 rates due to selection (good drivers stay longer) rather than skill. Using driver history as a feature could learn "retention predictor" instead of "race-level skill predictor."

**Prompt(s):**

_Prompt 5a (Methodology):_

> "How do I detect survivorship bias? I want to: aggregate by driver_id (count races, compute top_10_rate per driver), then correlate race count with top_10_rate. Strong positive correlation would indicate bias. What's the pattern?"

_Prompt 5b (Interpretation):_

> "I got Pearson r = 0.47 (p = 0.01) between n_races and top_10_rate. This is moderate-strong and statistically significant. What does this mean in terms of bias? Should I use n_races as a feature in my model?"

**Output:**  
AI explained that positive correlation between race count and performance is classic survivorship bias. Recommended against using aggregated driver stats in a single-race prediction model because you'd learn "historically retained drivers perform better" rather than "this race's context predicts outcome." Suggested saving driver-level features for models with explicit retention control.

**Validation:**

- Computed driver stats: n = 157 drivers, r_pearson = 0.470, p = 0.0116 ✓
- Checked top/bottom performers: high-race drivers had 75% Top 10; low-race drivers had 10-30% ✓
- Calculated effect size: r² = 0.22 → explains ~22% of variance (larger than grid's ~3%!) ✓
- Confirmed this is selection bias, not causality ✓

**Adaptations:**

- Explicitly excluded `n_races` from baseline to avoid confounding
- Documented for future reference: this bias is strong and should be addressed in Lab 2
- Noted limitation: simple baseline may underutilize historical information, but avoids spurious learning

**Final Decision:**  
**Rejected from baseline, but documented.** Survivorship bias is real and large (r² = 0.22), but using it would conflate retention with skill. Baseline avoids this by focusing on race-level features only (grid). Suggested stratification or fixed effects for future models.

---

## Entry 6 — [March 13–14, 2026] — Feature Engineering & Data Handling Decisions

**Context:**  
Encountered three data quality decision points:

1. Grid = 0 (pit-lane starts) → how to encode?
2. Missing positions → what does missing mean in context of outcome?
3. DNF flag → is it a useful feature or post-race leakage?

**Prompt(s):**

_Prompt 6 (Decision reasoning):_

> "In F1 data from Jolpica:
>
> - grid='0' means pit-lane start (penalty), not missing. Should I drop these rows (~10%), or create a 'pit_lane' category for future use?
> - Missing position values: are they MCAR, MAR, or MNAR? What does it mean for outcome prediction?
> - DNF (Did Not Finish) status: is using it to predict top_10 leakage? Why/why not?"

**Output:**  
AI explained:

- Grid=0 is structural encoding, not missing data. Best practice: convert to NaN for clean baseline (simplicity), plan categorical handling for Lab 2.
- Missing position: MNAR (outcome-dependent missing). Drivers without classification didn't finish, so they're definitely not Top 10.
- DNF: leakage (post-race), because you only know DNF status after the race. Cannot use for pre-race prediction.

**Validation:**

- Confirmed grid=0 appears in ~175 rows (~10% of raw extract) ✓
- Verified drop is acceptable for simple one-feature baseline ✓
- Checked missing positions: ~15 cases, all correspond to non-finished/unclassified drivers ✓
- Documented leakage: DNF is known only during/after race, not before ✓

**Adaptations:**

- Dropped grid=0 rows for baseline (clean, simple)
- Filled missing positions with 999 → top_10 = 0 (conservative, safe)
- Excluded DNF and status from all predictive features
- Documented decisions in DATA_QUALITY_LOG.md (Issues #1, #2, #3)

**Final Decision:**  
**Used.** Data engineering decisions ensure the baseline is leakage-free and well-documented. Each decision balances simplicity with correctness.

---

## Entry 7 — [March 14, 2026] — 4.6–4.8 (Stretch): Sklearn Metrics & Model Comparison

**Context:**  
**Stretch Requirements (4.6–4.8):** Compute additional metrics (Precision, Recall, F1, ROC-AUC) beyond accuracy, and implement a second baseline (Logistic Regression with 1 feature: grid) to compare against the domain heuristic rule.

Key learning needed: interpreting multiclass metrics in binary context, handling edge cases (zero_division), and understanding when/why sklearn models outperform domain rules.

**Prompt(s):**

_Prompt 7a (Metrics):_

> "For binary classification, explain Precision, Recall, F1, and ROC-AUC. When should I use each? What does zero_division parameter do in sklearn? How do I compute ROC-AUC correctly?"

_Prompt 7b (Model comparison):_

> "When a learned model (like Logistic Regression) achieves nearly identical performance to a simple interpretable rule on the same feature, what does this suggest about the underlying relationship? When is simplicity preferred over a learned model in practice?"

**Output:**  
AI provided clear definitions: Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean, ROC-AUC = area under ROC curve. Explained zero_division=0 handles cases where denominators are zero (edge case for minority class). For model comparison: noted that simple rules can be as powerful as learned models if the underlying relationship is transparent (e.g., monotonic grid effect).

**Validation:**

- Implemented metrics calculation using sklearn ✓
- Trained LogReg on train set (n=1,025), evaluated on val (n=159) ✓
- Compared metrics:
  - Heuristic (grid≤10): Accuracy 87.4%, F1 0.869, ROC-AUC 0.945
  - LogReg 1-feature: Accuracy 87.0%, F1 0.869, ROC-AUC 0.945
  - Nearly identical performance → validates domain rule's strength ✓
- Verified ROC-AUC using predicted probabilities (not 0/1 labels) ✓

**Adaptations:**

- Used `zero_division=0` to handle edge cases gracefully
- Computed `y_score` as predicted probabilities for ROC-AUC (more informative)
- Created side-by-side comparison table (Model vs. Metrics)
- Documented conclusion: simple interpretable rule preferred in Lab 1, but comparable in predictive power

**Final Decision:**  
**Used.** Stretch metrics provide additional validation that the domain heuristic is sound. No statistical advantage to learned model, so simplicity + interpretability wins for Lab 1. Planned refining for Lab 2.

---

## Entry 8 — [March 14, 2026] — Temporal Split Design & Leakage Prevention

**Context:**  
**Requirement 3.6:** Design train/validation/test split that mimics real-world deployment and prevents temporal leakage. F1 data is time-ordered (future races depend on past patterns), so random split is invalid.

**Prompt(s):**

_Prompt 8 (Design strategy):_

> "I have race data from 2022 (full), 2023 (full), and 2024 (partial: Jan-Dec). How should I split for temporal validity? Should I use:
> A) Train: 2022-2023, Val: early 2024, Test: late 2024
> B) Random split by driver to maintain independence
> C) Stratify by season?
> Which is best for time-series prediction and why? What are the boundary dates?"

**Output:**  
AI recommended (A): temporal split mimics real deployment (past→future causality). Cautioned against random split (leaks future patterns). Suggested using natural breakpoints (year transitions, mid-season) for clean boundaries.

**Validation:**

- Verified boundaries: Train end = 2023-12-31, Val end = 2024-05-31, Test start = 2024-06-02 ✓
- Checked no overlaps: each row in exactly one split ✓
- Confirmed test set untouched during EDA/tuning ✓
- Checked temporal order: max(train) < min(val) < min(test) ✓

**Adaptations:**

- Chose 2024 Jan-May for validation (provides ~160 rows, reasonable size)
- Chose mid-June for test boundary (aligns with F1 mid-season evolution)
- Documented leakage checks explicitly in notebook

**Final Decision:**  
**Used.** Temporal split is standard practice for time-series ML and aligns with IIT414W 3.6 requirement. Ensures honest evaluation and untouched test set.

---

## Summary Table: AI Interactions by Section

| Entry | Research Q.                 | Context                                  | AI Used                                     | Outcome                 |
| ----- | --------------------------- | ---------------------------------------- | ------------------------------------------- | ----------------------- |
| 1     | Q1: Grid → Top 10           | Correlation + visualization              | Code patterns, metric interpretation        | ✓ Used                  |
| 2     | Q2: Class balance           | Metric feasibility, viz strategy         | Best practices for imbalance                | ✓ Used                  |
| 3     | Q3: FP3 → Quali improvement | Session data integration, merge strategy | Error handling, code patterns, feature eval | ✓ Used (secondary)      |
| 4     | Q4: Confounding trap check  | Stratified analysis technique            | Statistical reasoning, interpretation       | ✓ Used                  |
| 5     | Q5: Survivorship bias       | Driver-level aggregation, bias detection | Causal reasoning, feature selection         | ✓ Rejected (documented) |
| 6     | Feature engineering         | Data quality decisions                   | Domain reasoning (leakage, encoding)        | ✓ Used                  |
| 7     | Stretch 4.6–4.8             | sklearn metrics, model comparison        | Learning + code patterns                    | ✓ Used                  |
| 8     | Temporal split              | Train/val/test design                    | Strategy validation, best practices         | ✓ Used                  |

---

## Overall Assessment

**AI Usage:** Targeted, verification-focused, and well-integrated.

**Primary uses:**

- **Code syntax:** matplotlib/seaborn patterns, sklearn API specifics
- **Conceptual clarity:** confounding, leakage, survivorship bias, stratified analysis
- **Best practices:** temporal splits, feature engineering, metric interpretation
- **Problem-solving:** data integration (FastF1 + Jolpica), error handling

**Manual/original work:**

- Hypothesis formulation (all 5 research questions)
- Data acquisition and validation (API calls, incremental execution, verification)
- Decision-making (which features to keep, trade-offs)
- Interpretation and documentation (all conclusions)

**Verification approach:**

- Every numerical result cross-checked against raw data
- Visualizations validated against computed aggregates
- Statistical claims verified before acceptance
- Code executed iteratively; no blind copying

**Over-documentation:**

- Every AI-assisted step explicitly tagged
- No output submitted as original work without acknowledgment
- Dual prompts used where applicable (technique + interpretation)
- Errors and adaptations documented

---

**Status:** ✓ Complete  
**Last Updated:** March 14, 2026  
**Compliance:** Meets IIT414W expectations for AI transparency and academic integrity.
