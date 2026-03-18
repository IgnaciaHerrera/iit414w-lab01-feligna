# Lab 1 Baseline Report
## F1 Top-10 Prediction (2022-2024)

> **Note:** This log documents all AI interactions used to develop the lab. Each entry describes what was asked, what AI returned, how it was verified, and how the output was adapted. The reasoning, decision-making, and interpretation are manual; AI assistance was used for code syntax, conceptual explanations, and best practices guidance under user direction.

---

## 1. Data Overview

**Dataset:** F1 race results via Jolpica API (2022-2023-2024)  
**Target:** `top_10` (binary: 1 if finishing position ≤ 10, else 0)  
**Validation Split:** 
- Train: 2022-2023 (all races through Dec 31, 2023)
- Validation: 2024 Jan 1 — May 31 (early/mid-season races)
- Test: 2024 Jun 1 onwards (reserved, not evaluated yet)

**Class Distribution (Validation):**
- Positive (Top-10): ~50%
- Negative (Outside Top-10): ~50%

---

## 2. Majority-Class Baseline

**Rule:** Always predict the most common class (Top-10 = 1).

| Metric | Value |
|--------|-------|
| Accuracy | 0.503 |
| Precision | 0.503 |
| Recall | 1.000 |
| F1 Score | 0.670 |
| ROC-AUC | 0.500 |

**Interpretation:**  
This baseline catches all actual Top-10 finishers (Recall = 1.0) but is useless for practical prediction—it predicts everyone as Top-10. This is our lower bound: any real model must beat 67% F1.

---

## 3. Domain Heuristic Baseline

**Rule:** If `grid` (starting position) ≤ 10, predict Top-10; else predict Not Top-10.

| Metric | Value |
|--------|-------|
| Accuracy | 0.874 |
| Precision | 0.875 |
| Recall | 0.875 |
| F1 Score | 0.875 |
| ROC-AUC | 0.874 |

**Confusion Matrix (Validation: 159 rows):**
- TP = 70 (correctly predicted Top-10)
- TN = 69 (correctly predicted Not Top-10)
- FP = 10 (incorrectly predicted Top-10)
- FN = 10 (incorrectly predicted Not Top-10)

**Interpretation:**  
This simple rule is surprisingly strong: F1 = 0.875 vs. 0.670 for majority-class. The rule is balanced (TP ≈ TN, FP ≈ FN), suggesting grid position is a genuinely predictive signal. This becomes the target metric for Lab 2: any new model must beat **F1 = 0.875** to justify added complexity.

---

## 4. Metric Justification

**Primary Metric Choice:** F1 Score

**Why F1?**
1. **Class Balance:** Top-10 is ~50% in our data, so accuracy is not misleading, but F1 is still more robust.
2. **Error Type Balance:** Both false positives (predicting Top-10 wrongly) and false negatives (missing a Top-10 finisher) are equally costly in this context—a team wants to know who will score points.
3. **Interpretability:** F1 is the harmonic mean of precision and recall, punishing extreme imbalances. At 0.875, it tells us the model is reliable at both finding and not overstating Top-10 performance.

**Alternative Metrics:**
- **Accuracy:** Acceptable here (balanced classes) but less informative than F1.
- **ROC-AUC:** Good for comparing models probabilistically, but we need a decision threshold.
- **Precision/Recall:** Could be used separately, but F1 integrates both.

---

## 5. Leakage Guard Checklist

Applied to domain heuristic baseline. All checks passed ✅

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Grid is available pre-race (post qualifying) | ✅ | Grid position set before race starts |
| 2 | Train/Val/Test split is strictly temporal | ✅ | No future data in training |
| 3 | Target does not leak into features | ✅ | Grid is independent of position |
| 4 | No post-race fields used (position, status, points, laps) | ✅ | Only `grid` is used |
| 5 | Rolling/lag features use `.shift(1)` | N/A | No lag features in baseline |
| 6 | Categorical encodings fitted on train only | N/A | No categorical encoding in baseline |
| 7 | Hyperparameters not tuned on test | ✅ | Heuristic is domain-driven, not tuned |
| 8 | No data snooping in test set | ✅ | Test set untouched |
| 9 | RANDOM_SEED = 414 set | ✅ | Reproducible |
| 10 | Notebook runs top-to-bottom | ✅ | Verified |

**Notable Finding:**  
The feature audit in baseline.ipynb explicitly classified all columns by pre/post-race availability. This forced explicit thinking about leakage risk before building features. Lesson: audit features *before* using them.

---

## 6. Baseline Interpretation

**Key Findings:**

1. **Grid position is predictive.** The domain heuristic at F1 = 0.875 outperforms the majority-class baseline (0.670) by +30.5 percentage points. This suggests that drivers starting in the top 10 genuinely have a higher probability of finishing in the top 10—which aligns with F1 domain knowledge (grid determines tire strategy, track position, and race-day advantage).

2. **Balance in errors.** The confusion matrix shows TP ≈ TN and FP ≈ FN (both ~10 errors each in validation). This balance indicates the rule doesn't systematically over-predict or under-predict; it simply captures a real pattern without bias.

3. **What's missing.** F1 = 0.875 is strong, but is not perfect. The 20 errors (FP + FN) suggest that pure grid position isn't the whole story. Drivers can overcome a poor grid through strategy, and drivers on pole can DNF or drop back. This is where Lab 2 comes in: can we add features (driver form, constructor tier, weather context) to narrow the gap?

4. **Hardest part of this section:** The hardest part was deciding whether 87.4% F1 counts as "good" without seeing what competitors achieve—I had to reason from first principles that a simple one-feature rule beating the majority-class baseline by 30+ points is substantially better, even if room remains.

---

## 7. Next Steps (Lab 2)

**Challenge:** Engineer 3+ new features to beat F1 = 0.875.

**Candidate Features to Explore:**
- **Lag feature:** Previous race finishing position (proxy for driver consistency)
- **Rolling aggregate:** Average finishing position over last 3 races
- **Constructor tier:** Encode constructor quality (Top 4 vs. Other)
- **Interaction:** Driver past performance at specific circuit types

**Success Criteria:**  
Any model (Logistic Regression or Decision Tree) that achieves F1 > 0.875 on the same validation set will have learned something new. If Lab 2 model ≤ 0.875, the heuristic remains the best choice (principle: stick with simpler models unless forced to upgrade).

---

