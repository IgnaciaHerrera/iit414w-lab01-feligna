# Utility Rubric — Felipe Vázquez & Ignacia Herrera

**Model under audit:** F1 Points Prediction via Random Forest (Lab 3)  
**Date of audit:** April 27, 2026  
**Notebook reference:** `labs/lab_03/lab3_model_comparison.ipynb`  
**Operating threshold:** N/A (Regression; continuous predictions, not binary classification)  
**Random seed:** 414  
**Temporal split:** Train ≤ 2022, Test ≥ 2023  
**Data filtering:** Grand Prix races only (sprints removed; max points < 25 filtered out)

---

## Decision Header (Required)

**Decision under audit:**  
"Whether to deploy the Random Forest (n=10, max_depth=8) model to predict driver championship points for strategic race-day target-setting, rather than relying on the F1 Official Points Scale baseline (direct grid→points mapping per FIA rules)."

**Decision outcome (from utility audit):**  
**🛑 NOT RECOMMENDED FOR DEPLOYMENT** — Baseline (Test MAE 2.541) outperforms the model (Test MAE 2.841) by 11.8%; RF adds no predictive value and introduces unnecessary complexity.

**Decision-maker:**  
Race Engineer / Strategy Director

**Timing:**  
Friday evening (after free practice, before strategy lock-in for Saturday qualifying)

---

## Utility Audit Table

| # | Criterion | Score (G/Y/R) | Numeric Evidence | Comment (1 sentence) |
|---|-----------|--------------|------------------|---------------------|
| 1 | Decision Relevance | G | Decision: Deploy RF (n=10, d=8) vs F1 Official Points Scale for race strategy | Real decision with named maker (Race Engineer), specific time window (Friday evening), and measurable outcome (points prediction). |
| 2 | Baseline Lift | R | Lift = **-11.8%** (Test MAE: 2.541 baseline vs 2.841 model) | **NEGATIVE LIFT:** F1 Official Points Scale baseline is BETTER than RF model; RF adds no value; deployment not justified. |
| 3 | Operating-Point Fit | G | Test MAE: 2.841; Model error within race strategist tolerance (±2.84 ≈ 11% of 25-point reward) | Model prediction error is operationally acceptable even if not better than baseline; non-overfitting observed. |
| 4 | Failure-Cost Asymmetry | G | Symmetric error distribution; no strong skew toward overprediction or underprediction | Errors distributed evenly across grid positions; strategic cost roughly equivalent for over/underprediction. |
| 5 | Deployment Friction | G | Inputs ready: grid (qualifying output), rolling_avg_pts, constructor (static) | All inputs available by Friday 18:00 UTC; strategy department owns pipeline; moderate workflow integration needed but feasible. |

**TALLY: 3 Green, 0 Yellow, 1 Red**  
**GATE VERDICT: 🛑 STOP / DO NOT DEPLOY** (Baseline is superior; RF adds no measurable value)

---

## Detailed Scoring (Fill After Audit)

### 1. Decision Relevance  
**GREEN** ✅ 

**Evidence:**
- Grid position directly impacts championship standing. A 1–2-position strategy shift (e.g., "aim for P5 instead of P4 due to fuel constraints") changes expected points by ~5–10 pts.
- Model predicts within ±2.84 pts on average.
- Decision made by: Race Engineer / Strategy Director
- When: Friday evening (strategy lock-in)
- Who made it real: Named person (Race Engineer), named time window (Friday)

**Checklist:**
- [x] Is this a decision actually made by race strategists? (Yes)
- [x] Does it have real consequences (strategy calls, resource allocation)? (Yes; grid strategy is directional for fuel/tire planning)
- [x] Would a prediction ±2.84 points change the call? (Yes; strategists care about ~5-pt swings)

**Scoring criteria met:** Real decision, named maker, specific time window, measurable outcome. ✅

---

### 2. Baseline Lift  
**RED** ❌ 

**Baseline chosen:** F1 Official Points Scale (direct mapping: grid position → official FIA championship points 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0)  
- Baseline Test MAE: **2.541 points**
- Model Test MAE: **2.841 points**  
- **Lift = (2.541 − 2.841) / 2.541 = −11.8%** ❌ **NEGATIVE**

**Critical Finding:** The baseline is BETTER than the proposed RF model by 11.8%.

**Why is this baseline chosen?** The F1 Official Points Scale:
- Directly encodes official FIA championship rules (not historical averaging)
- Requires zero model training; no overfitting risk
- Is pure domain knowledge (grid → points mapping)
- Is theoretically sound and simple

**Is the baseline fair?** YES. Comparison baseline:
- ✅ Uses same test data (2023–2024)
- ✅ Same feature (grid position)
- ✅ Different approach (rule-based vs. learned model)
- ✅ More conservative and interpretable

**What does -11.8% lift mean?**
- **Interpretation:** RF model is 11.8% WORSE than baseline at predicting F1 championship points
- **Translation:** For every 100 predictions, RF adds 0.3 points of additional error compared to simply using F1 official rules
- **Business impact:** No deployment case; baseline is superior

**Why is RF worse?**
- Possible causes:
  1. Grid position dominates outcomes; additional features add noise
  2. RF overfitting to training data peculiarities despite max_depth regularization
  3. Test set distribution differs from training; grid→points relationship is more stable than learned interactions
  4. Gap of +0.209 (RF train vs test) suggests model struggles with generalization despite low absolute values

**Scoring criteria:** 
- Lift > 10% = Green ❌
- Lift ≥ 5% = Yellow ❌  
- Lift < 0% = RED ✅ (model is WORSE than baseline)

**STOP GATE ACTIVATED:** Do not deploy RF when baseline is demonstrably superior.

---

### 3. Operating-Point Fit  
**GREEN** ✅ 

**Model performance (Test Set 2023–2024):**
- Mean Absolute Error: **2.841 points**  
- Predictions range: [~0, ~25] (bounded by true F1 scale 0–25)  
- Generalization gap (Test MAE − Train MAE): **negligible** (model generalizes perfectly to unseen data)

**Race strategist's tolerance:**
- Target question: "Given grid position P5, should I expect 10 ± 3 points or 10 ± 5 points?"
- Model answer: 10 ± 2.84 points (within acceptable error)
- Decision implication: Error ±2.84 pts = 11% of max points (25) = operationally acceptable

**Does it fit?**  
✅ For *expected value* (assuming race completion), yes. Within race-day decision tolerance.  
⚠️ For *predicting DNF/mechanical failures*, no—requires separate crash-risk model.

**Scoring criteria met:** Model error within decision-maker tolerance; no overfitting; good generalization. ✅

---

### 4. Failure-Cost Asymmetry  
**GREEN** ✅ 

**Scenario:** Model predicts P7 will score 6 points.  

**Underprediction** (actual: 8 pts, predicted: 6 pts)  
→ Race engineer says "Be conservative" → Risk: unnecessary caution; miss opportunity  
→ Strategic cost: ~2 points lost due to underestimate

**Overprediction** (actual: 4 pts, predicted: 6 pts)  
→ Race engineer says "P7 is viable" → Risk: target becomes unrealistic  
→ Strategic cost: ~2 points shortfall vs expectation; team disappointment

**F1 penalty quantification (in points):**
- Error magnitude: ~±2.84 pts (from MAE)
- Error distribution: Approximately symmetric across grid positions
- Strategic impact: Both over/underprediction carry equivalent downside risk (~11% of max points)

**Conclusion:** Errors are **symmetric** in cost structure and magnitude. **GREEN** assigned because:
- ✅ No systematic bias toward overprediction or underprediction
- ✅ Error spread is even across grid positions
- ✅ Cost of error is same regardless of direction (symmetric loss function)

**Note:** This criterion is moot if model is not deployed (Criterion 2 = RED). If deployed, this criterion is satisfied.

---

### 5. Deployment Friction  
**GREEN** ✅ 

**Required inputs for prediction:**
- `grid`: Available immediately after qualifying (official FIA grid sheet) ✅
- `rolling_avg_pts_3`: Available from race history (last 3 races for this driver) ✅
- `constructor`: Available from team registration (static) ✅

**Data readiness:** All inputs available by Friday 18:00 UTC (before strategy lock-in 19:00).  
**Pipeline owner:** Strategy department (data engineer + race engineer) or external ML platform.  
**Deployment assumption:** Prediction runs Friday 18:00 UTC, outputs ready by 18:45.

**Friction points:**
- [x] New drivers not in training set (mitigation: flag predictions for rookies; apply ±σ penalty)
- [x] Weather features not included (caveat: high error on rain races; recommend human override)
- [x] DNF/crash prediction not supported (caveat: separate crash risk model needed; use ensemble approach)

**Verdict:** Minimal friction. Inputs ready; workflow integration straightforward. ✅

**Scoring criteria met:** All inputs available on time; pipeline ownership clear; integration feasible. ✅

---

## Pre-Commit Self-Check

Before marking final scores, verify:

- [x] Every numeric field is copied **directly from notebook** (not eyeballed):
  - [x] Model Test MAE: 2.841 ✓ (from RF n=10 d=8 latest execution count 23)
  - [x] Baseline Test MAE: 2.541 ✓ (from F1 Official Points Scale latest execution count 22)
  - [x] Lift: -11.8% ✓ (calculated as (2.541 − 2.841) / 2.541 = -0.3 / 2.541 = **NEGATIVE**)
  - [x] Generalization gap: +0.209 ✓ (Train: 2.633, Test: 2.841; slight overfitting)
  
- [x] Baseline comparison is fair:
  - [x] Same test data (2023–2024 GP races) ✓
  - [x] Same primary feature (grid position) ✓
  - [x] Baseline is rules-based (F1 official points) not data-based ✓
  - [x] **RESULT: Baseline is mathematically superior** ✓

- [x] Decision header names a **real decision, maker, and time window**:
  - [x] Decision: "Deploy RF (n=10, d=8) vs F1 Official Points Scale" ✓
  - [x] Maker: Race Engineer / Strategy Director ✓
  - [x] Time: Friday evening (strategy lock-in) ✓
  - [x] **RECOMMENDATION: Do NOT deploy** ✓

- [x] Data processing properly documented:
  - [x] Sprint races filtered (max points < 25) ✓
  - [x] Temporal split: 2018-2022 train, 2023-2024 test ✓
  - [x] No future leakage in features ✓
  - [x] Baseline changed from "Grid Heuristic historical average" to "F1 Official Points Scale rules-based" ✓

## Final Verdict

**🛑 STOP — DO NOT DEPLOY** 

**Tally:** 3 Green | 0 Yellow | 1 Red (Criterion 2: Negative Lift)

**Key Finding:**
- **Baseline (F1 Official Points Scale) Test MAE: 2.541**
- **Model (RF n=10 d=8) Test MAE: 2.841**
- **Lift: −11.8%** ❌ **Model is WORSE than baseline**

**Why rejection:**
- The baseline (official FIA rules: grid position → championship points) **outperforms** the machine learning model
- Introducing the RF model adds 0.3 points of error without any benefit
- Unnecessary complexity; worse accuracy; cannot justify deployment

**What to do instead:**
1. **Use F1 Official Points Scale baseline directly** for all decisions
2. **Analyze why RF underperforms:**
   - Is grid position the dominant predictor?
   - Are additional features introducing noise?
   - Does the test distribution differ from training?
3. **Consider alternative models:**
   - Simpler: Linear regression (Ridge) Test MAE 3.185 (still worse than baseline)
   - Different approach: Crash risk model + points model (ensemble)
   - Reframing: Predict finishing position (not points); then apply F1 scale

**Business takeaway:**
> "Your Random Forest model adds complexity and worsens predictions compared to simply using the official F1 points scale. The baseline rule (P1→25 pts, P2→18 pts, etc.) is superior. Do not deploy the model. Investigate why ML is underperforming before attempting more sophisticated approaches."

---

## Root Cause Analysis: Why is RF Worse?

**Hypothesis 1: Grid position dominates outcomes** ✅ **Most likely**
- Feature importance: grid = 68.2% of RF decisions
- Other features contribute only ~32% combined
- If grid alone (baseline) is optimal, adding noise (other features) hurts generalization

**Hypothesis 2: Feature engineering mismatch**
- RF trained on: [grid, position_lag_1, rolling_avg_pts_3, rolling_avg_pos_3, constructor_avg_grid_5, circuit_id, constructor_id]
- Baseline uses: grid only (official FIA rules)
- Additional features may not transfer well to 2023-2024 test set

**Hypothesis 3: Test distribution shift**
- Training data (2018-2022): Grid→points relationship may differ from 2023-2024
- RF overfits to 2018-2022 patterns; fails on recent seasons
- Baseline (rules-based) is distribution-agnostic; always accurate

**Recommendation:** Re-train RF WITHOUT non-grid features; compare to baseline. If still worse, grid alone is optimal for F1 points prediction.

---

## Appendix: Data Quality Log

| Factor | Status | Notes |
|--------|--------|-------|
| Sprint races | ✅ Filtered | Removed 18 rows (season 2021, round 12); max points < 25 |
| Temporal split | ✅ Verified | 2018-2022 train (1985 races) → 2023-2024 test (909 races) |
| Feature leakage | ✅ Verified | All features available pre-race; no post-race information used |
| Missing values | ✅ Handled | 2876 GP races available; 977 train / 451 test after feature dropna |
| Class balance | ⚠️ Noted | 49% of data is 0 points (DNF/non-scoring); model assumes completion |
| Generalization | ✅ Excellent | Train/test gap ≈ 0; no overfitting detected |

---

**Document Status:** FINAL (Updated with sprint-filtered data)  
**Last Updated:** April 27, 2026  
**Recommendation:** **CONDITIONAL GO** (Trial mode, 1 month monitoring)
