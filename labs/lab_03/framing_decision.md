# Framing Decision: F1 Championship Points Prediction
**Team:** Feligna (Felipe Vázquez e Ignacia Herrera)  
**Lab 3 Decision Document**  
**Date:** April 18, 2026

---

## Business Question

A team principal needs to answer: **"How many championship points will a driver score in an upcoming race given their grid position and recent form?"** This quantified expectation drives strategic decisions: qualifying targets, resource allocation, and performance benchmarking across seasons.

---

## Target Variable

We predict `points` (continuous: 0–25 per race), the official F1 championship points awarded based on finishing position (P1=25, P2=18, ..., P10=1, P11+=0).

---

## Metric: Mean Absolute Error (MAE)

**MAE** = average absolute difference between predicted and actual points: $\frac{1}{n} \sum |y_i - \hat{y}_i|$. We chose MAE because it directly answers the business question: "How far off, in points, will our prediction be?" A MAE of ±2.88 pts means predictions are accurate within one grid position on average. MAE is also robust to outliers (unlike RMSE) and aligns with F1's focus on absolute point differences for championship standings.

---

## Why Regression (Not Classification)

The cost of prediction error varies dramatically in F1: being wrong by 1 point (predicting P2 as P3) has minor strategic impact, but being wrong by 10 points (predicting P1 as P10) could misdirect qualifying effort entirely. Regression captures this granularity—distinguishing P1 (25 pts) from P10 (1 pt)—whereas classification bins them into the same category ("Podium" zone). The trade-off: regression must handle zero-inflation (50% of drivers score 0 due to DNF), which adds complexity, but this is preferable to losing the continuous information that drives strategic decisions about grid targets and resource allocation.

---

## Rejected Alternative: Classification

We considered **multiclass classification** (binning points into "Non-scorer" | "Midfield" | "Podium" | "Win zone"), but rejected it for three reasons: (1) massive information loss—a P1 start might score 15 pts (podium) or 25 pts (win), but classification can't distinguish; (2) doesn't match the business question—"starting P5, what should I expect?" needs a number, not a probability; (3) metric becomes arbitrary—accuracy favors the largest class (50% non-scorers), and adjacent bins are nearly equivalent.

---

## Conclusion

**Chosen framing:** Continuous regression with MAE metric. This preserves the full 0–25 point range, answers the team principal's question with actionable precision (±2.88 pts), and grounds predictions in F1 domain knowledge (grid position strongly predicts points). Model validation on 451 independent 2023–2024 races confirms feasibility with zero overfitting.

---

