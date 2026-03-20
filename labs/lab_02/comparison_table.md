# Lab 1 vs Lab 2 — Comparison Table

## [Team: Feligna, Felipe & Ignacia]

| Model / Baseline | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------------------------|----------|-----------|--------|-------|---------|
| Majority class (Lab 1) | 0.503 | 0.503 | 1.000 | 0.670 | 0.500 |
| Domain heuristic grid≤10 (Lab 1)| 0.874 | 0.875 | 0.875 | 0.875 | 0.874 |
| Lab 2 model (LogReg) | 0.849 | 0.878 | 0.812 | 0.844 | 0.924 |

---

## Primary metric: **F1-Score** 
(Justification: Balanced measure of precision and recall. In F1 racing, both false positives [predicting top-10 when driver finishes outside] and false negatives [missing a top-10 finish] are costly errors. F1-score penalizes both equally, making it the right choice over accuracy alone.)

---

## Interpretation 

The engineered features did not surpass the grid≤10 baseline (F1=0.875 vs 0.844)—a 0.031 gap that tells us grid position dominates F1 top-10 prediction because drivers in P1-P10 finish top-10 ~88% of the time, while rolling form features alone cannot capture this structural advantage. Our 9 false positives and 15 false negatives cluster in mid-field (P11–P20), where grid is less predictive but our features also fail. To beat this baseline, we would need pre-qualifying features (practice data, pit strategy, car performance) rather than just race form. Despite losing to the baseline, the model's strong ROC-AUC (0.924) and precision (0.878) show our features have value for probability-based ranking or ensembles in future work.




