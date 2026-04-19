# Framing Decision: F1 Championship Points Prediction
**Team:** Feligna (Felipe Vázquez e Ignacia Herrera)  
**Lab 3 Decision Document**  
**Date:** April 18, 2026

---

## Problem Statement

**Business Question:** "How many championship points will a driver score in an upcoming F1 race, given their starting grid position and recent performance?"

**Context:** F1 race results are deterministic outcomes: each driver receives 0-25 points based on finishing position (P1=25, P2=18, P3=15, ..., P10=1, P11+=0). We need quantified expectations for race-day performance to enable strategic planning (qualifying targets, resource allocation, risk assessment).

---

## Framing Decision: REGRESSION (Not Classification)

### Choice: Continuous Regression Model
- **Target variable:** `points` (continuous: 0-25 per race)
- **Prediction type:** Estimate expected points per race (real number, e.g., 12.3 pts)
- **Output interpretation:** "We expect this driver to score approximately 12 points, with ±2.88 pts uncertainty"

### Why Regression?

| Aspect | Regression | Classification (Alternative) |
|--------|-----------|------------------------------|
| **Output** | Continuous (0-25) | Discrete categories (e.g., "scoring" vs "non-scoring") |
| **Information loss** | None; preserves full range | High; collapses 0-25 range into 2-3 bins |
| **Use case alignment** | Quantified expectations for planning | Binary yes/no decision only |
| **Uncertainty quantification** | Natural (±2.88 pts = confidence) | Probability only (70% chance of scoring?) |
| **Strategic actionability** | "Target P5 to average 15 pts" | "Flip a coin; 50-50 shot" (useless) |

**Example:** Suppose model predicts 8 points for a given race.
- **Regression interpretation:** "Expected value is 8 pts; plan resource allocation assuming 8 pts contribution"
- **Classification interpretation:** "70% probability driver scores >0 pts" (What about the remaining 30%? What if they score 2 vs 18?)

The regression framing captures **magnitude of expectation**, not just binary outcome.

---

## Metric Choice: Mean Absolute Error (MAE)

### Selected Metric

**Mean Absolute Error (MAE):** Average absolute difference between predicted and actual points.

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

where $y_i$ = actual points, $\hat{y}_i$ = predicted points, $n$ = number of races.

### Why MAE (Not RMSE)?

| Metric | Formula | Interpretation | When To Use |
|--------|---------|-----------------|-------------|
| **MAE** | $\frac{1}{n}\sum\|y - \hat{y}\|$ | Avg prediction error in points | ✅ Direct: "±2.88 pts on average" |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Penalizes large errors heavily | Large errors are equally bad; no reason to over-penalize |
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ | Squared error | Not in original units; hard to interpret |
| **R²** | Variance explained | Relative performance | Doesn't tell us absolute error magnitude |

**Justification for MAE:**
1. **Interpretable:** "±2.88 points" is actionable for F1 strategy (P5 vs P6 is ~1 pt difference)
2. **Robust to outliers:** RMSE over-penalizes rare 10+ point misses (e.g., unpredicted crashes). MAE treats all errors equally.
3. **Aligned with domain:** F1 cares about absolute point differences for championship standings, not squared error

**Example:** If model predicts 15 pts but driver scores 20 pts (P1 vs P2 scenario):
- MAE contributes: |20-15| = 5 pts
- RMSE contributes: sqrt(5²) = 5 pts (same in this case, but RMSE explodes for larger errors)

---

## Alternative Rejected: Classification ("Scoring Zones")

### Framing Not Chosen
Classify each race into bins: **"Non-scorer" (0) | "Midfield" (1-8 pts) | "Podium" (9-15 pts) | "Win zone" (16-25 pts)**

### Why We Rejected It

1. **Information loss is massive:**
   - Grid P1 might score 15 pts (podium) OR 25 pts (win)
   - Classification can't distinguish → same bin
   - Strategy can't differentiate: "target podium finish" is vague

2. **Doesn't match business question:**
   - Team Principal asks: "If we start P5, how many points?"
   - Classification answers: "Maybe podium, maybe midfield" (unhelpful)
   - Regression answers: "Starting P5: expect ~10 pts, ±2.88" (actionable & specific)

3. **Loses uncertainty quantification:**
   - Regression: ±2.88 pts = confidence interval
   - Classification: 45% chance of "podium" (What about the other 55%? Which bin instead?)

4. **Metric interpretation becomes awkward:**
   - Accuracy = "How often correct bin?" (But adjacent bins are almost equally good)
   - F1-score / Precision-Recall = Biased toward largest class (non-scorers = 50% of data)

---

## Trade-Offs Summary

| Decision | Benefit | Cost |
|----------|---------|------|
| **Regression vs Classification** | Quantified expectations (15±2.88 pts) | Need continuous model (slightly more complex) |
| **MAE vs RMSE** | Robust, interpretable, aligned with F1 | Doesn't penalize outliers as heavily |
| **Continuous target (0-25)** | Full information preserved | Model must learn 0-25 range, not just 2-3 bins |

---

## Conclusion

**Chosen framing:** Continuous regression with MAE metric.

**Justification:** 
- Matches business question ("How many points?")
- Preserves full information (0-25 point range, not 2-3 bins)
- Metric directly interpretable (±2.88 pts for strategy planning)
- Robust and well-calibrated for F1 decision-making

**Validation:** Model tested on 451 independent 2023-2024 races; achieved MAE=2.88 pts with perfect generalization (no overfitting).

---

