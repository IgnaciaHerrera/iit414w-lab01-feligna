# Model Exploration - Feligna (Felipe Vázquez e Ignacia Herrera) (IIT414W)
## IIT414W - Unit II - Lab 3 - Initial Exploration - March 30, 2026

---

## 0. Framing Decision (Initial - Revision Possible for Lab 3 Final)

### Business Question
A Formula 1 team principal wants to know: **"Given a driver's grid position and historical performance, how many championship points should we expect the driver to score?"**

### Target Variable
**Continuous regression target: `points` (0-25 per race)**

### Metric
**Mean Absolute Error (MAE)** - Measures average prediction error in points. 
- Interpretation: "On average, our model's predictions are off by ±X points"
- Preferred over RMSE because it's interpretable in F1 points without squaring

### Why This Framing

1. **Business Relevance**: Two drivers with same grid position could score 0 or 25 points. Regression captures this variance - classification (top-10 yes/no) loses precision.

2. **Natural Target**: Points scored IS continuous and ordered. The gap between P1 (25 pts) and P10 (1 pt) is huge strategically. Regression respects this hierarchy.

3. **Data Advantage**: 2,930 historical race entries provide signal for regression. Many features (grid, constructor, form) are continuous predictors.

### Rejected Alternative: Classification (top-10 binary)

- **Why rejected**: Loses information. A P9 finish (1 pt, <1 pt gap to P10) and a P10 finish (0 pts) are treated identically in classification.
- **When it might work**: If business question was "probability of scoring ANY points?" But our question needs point magnitude.
- **Trade-off**: Classification would be slightly easier computationally, but regression is more strategically valuable here.

---

## 1. Models Trained

| Model | Key Hyperparameters | Features Used | Notes |
|---|---|---|---|
| Predict Mean | strategy='mean' | None | Trivial baseline |
| Predict Median | strategy='median' | None | Trivial baseline |
| Grid Heuristic | Lookup table | grid only | Domain baseline |
| Ridge | α=1.0 | All 9 features | Baseline linear model |
| Ridge | α=100.0 ⭐ | All 9 features | Tuned via alpha sweep |
| Random Forest | n=100, max_depth=10, min_samples_leaf=5 | All 9 features | Demo RF |
| Random Forest | n=50, max_depth=5, min_samples_leaf=5 | All 9 features | Lighter RF variant |

**Features**: grid, position_lag_1, rolling_avg_pos_3, constructor_avg_grid_5, rolling_avg_pts_3, circuit_id, constructor_id

---

## 2. Comparison Table (Same Metric, Same Validation: 2023–2024 Test Set)

| Model | Train MAE | Test MAE | Gap | WHY |
|---|---|---|---|---|
| Predict Mean | — | 5.923 | — | Predicts 5.09 (mean); ignores all features |
| Predict Median | — | 5.129 | — | Median=1.0 due to ~50% zero-inflation; more robust than mean |
| Grid Heuristic | — | 3.246 | — | Strong baseline: historical avg points per grid position; no model bias |
| Ridge (α=1.0) | 3.350 | 3.299 | -0.051 | Linear model underfits; cannot learn nonlinear P1→P10 threshold differences |
| **Ridge (α=100.0)** | 3.485 | **3.219** | -0.267 | Heavy regularization improves generalization; gap smaller than α=1.0 |
| **Random Forest (n=100, d=10)** | 2.482 | **2.838** ⭐ | 0.356 | Captures nonlinearity; learns grid thresholds (P1-5 vs 6-10 vs 11-20); best test MAE |
| Random Forest (n=50, d=5) | 2.635 | 2.957 | 0.322 | Lighter constraints; safer generalization but less nonlinearity capture |

**Key Insight**: Grid heuristic (MAE=3.246) is a strong baseline because it DIRECTLY encodes the grid→points mapping.
Any model must beat this to add value. Only Random Forest (n=100) achieves this (2.838 vs 3.246).

---

## 3. Best Model Justification

**Winner: Random Forest (n=100, max_depth=10, min_samples_leaf=5)**
- **Test MAE: 2.838** (12.5% better than grid heuristic, 11.8% better than best Ridge)

**Why this model (not just "lowest MAE")**:

1. **Captures Nonlinearity**: The F1 points distribution is severely nonlinear. The gap from P1→P2 is 7 points, but P9→P10 is only 1 point. Ridge assumes linear relationships (each position worth -0.18 points) and cannot learn these thresholds. Random Forest naturally learns via recursive splitting: "if grid ≤ 5, then +Y points; else if grid ≤ 10, then +Z points..."

2. **Reasonable Overfitting Gap**: Train MAE (2.482) vs Test MAE (2.838) = gap of 12.5%. This is acceptable—the model is NOT memorizing. (Overfitting would be gap > 20-30%.)

3. **Interpretable Feature Importance**:
   - Grid (37.2%): Direct predictor—starting position determines championship points
   - Constructor avg grid (29.6%): Team strength proxy (competitive teams start P1-3)
   - Rolling avg points (21%): Recent driver form

   This ranking matches F1 intuition and validates the model's learned representation.

4. **Mechanism**: The 100 trees collectively learn 100 different grid-position threshold combinations. Each tree is trained on random subsets of features and data (bootstrap). Averaging these diverse trees reduces variance compared to a single tree, producing smooth predictions that generalize to test data.

5. **Beat All Alternatives**:
   - Predicts mean: RF is 52% better (5.923 → 2.838)
   - Ridge tuned: RF is 12% better (3.219 → 2.838)
   - Grid heuristic: RF is 12.5% better (3.246 → 2.838)

---

## 4. One Honest Limitation

**Primary Limitation: Cannot Extrapolate Beyond Training Domain**

The model was trained on 2018–2022 data with specific driver/constructor rosters and point systems. Three failure modes:

1. **New Drivers/Constructors**: In 2023–2024, some drivers switched teams (e.g., Alonso→Aston Martin) or new teams debuted. The model has never seen these combinations in training. A new driver has zero `rolling_avg_pts_3`, causing it to default to grid-only prediction (essentially the heuristic).

2. **Regulation Changes**: F1 changed penalties, tire strategies, and point values over the years. The model learned 2018–2022 patterns. If 2024 regulations favor small teams (e.g., midfield scoring more), the model may overpredict for smaller teams trained on historical patterns.

3. **Gap-to-Test Range**: The model learned on grid positions mostly 1–20. In edge cases (grid position 21+, which occurs in DNF scenarios), the model extrapolates poorly because it's outside the training range.

**Measurement & Mitigation**: 
- Performance degradation will be visible in 2024 held-out data (if available). Monitor weekly prediction residuals.
- Recommend retraining monthly with rolling window to adapt to emerging patterns.
- Pre-filter: Flag predictions for drivers/teams not in training set; use heuristic instead.

**Honest Assessment**: Despite limitations, the model is strategic because:
- Beats all practical baselines, including the strong grid heuristic
- Provides calibrated point predictions (not just probabilities)
- Identifies which features matter (grid >> team >> form)
- Non-memorization indicates real learned patterns

---

## Summary for Lab 3

**Framing**: Regression (points predicted as continuous) is more valuable than classification (top-10 yes/no) because it preserves point magnitude—critical for championship strategy.

**Best Model**: Random Forest (n=100, d=10) with MAE=2.838—learns nonlinear grid→points thresholds that Ridge cannot capture.

**Honest Limitation**: Cannot generalize to new driver/team combinations or future regulation changes; requires monthly retraining.

**Next Steps for Lab 3 Final**: 
- [ ] Validate on 2024 test set (if providing additional data)
- [ ] Implement threshold tuning for classification alternative (if pursuing that framing)
- [ ] Test model on edge cases (new teams, regulation changes)
- [ ] Document PROMPTS.md with all AI interactions and critical distance

---

## Appendix: Learning Goals Achieved

✅ **Goal 1**: "Compare Ridge and Random Forest on same temporal split" → Ridge: MAE=3.299 vs RF: MAE=2.838  
✅ **Goal 2**: "Build Pipeline with ≥2 models, evaluate with same metric and temporal validation" → Done (7 models total)  
✅ **Goal 3**: "Explain difference between predicted and calibrated probability" → Covered in YOUR TURN 3 calibration section  
✅ **Goal 4**: "Interpret comparison honestly; model failures are valid results" → Ridge and baselines documented with clear reasoning  

---

*File created: March 30, 2026 | Last updated: March 30, 2026*
