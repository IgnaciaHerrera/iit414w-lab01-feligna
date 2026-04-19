# Lab 3: Model Comparison Table

**Team:** Feligna (Felipe Vázquez e Ignacia Herrera)  
**Course:** IIT414W - Unit II  
**Date:** April 2026

---

## Overview

Honest comparison of 7 models (3 baselines + 2 Ridge variants + 2 Random Forest variants) applied to F1 championship points prediction.

**Validation Method:** Temporal split (2018-2022 train, 2023-2024 test)  
**Metric:** Mean Absolute Error (MAE) in points  
**Test Set:** 451 race entries from seasons 2023-2024 
**Primary Ranking:** Test MAE (ascending = best)

---

## Main Comparison Table (All Models)

| Rank | Model | Type | Train MAE | Test MAE | Gap | Benchmark | WHY |
|------|-------|------|-----------|----------|-----|-----------|-----|
| **1** ⭐ | **RF (n=50, d=5)** | **Tree Ensemble** | **2.953** | **2.882** | **-0.071** | **WINNER** | Learns nonlinear grid thresholds (P1-5 high, P6-10 medium, P11+ low). Shallow depth (5) prevents overfitting. Beats RF(n=100,d=10) by 0.003 MAE with better generalization (gap -0.071 vs +0.389). |
| 2 | RF (n=100, d=10) | Tree Ensemble | 2.496 | 2.885 | +0.389 | — | Captures same nonlinearity but 100 trees + depth=10 leads to overfitting. Gap +13.5% shows memorization cost. Test MAE marginally worse (+0.003 vs n=50). |
| 3 | Grid Heuristic | Domain Baseline | 3.510 | 3.099 | -0.411 | Strong | Direct lookup: grid→avg_points. Encodes F1 knowledge but can't adapt per driver. RF beats this by 0.217 MAE (7%), proving ML adds value beyond domain heuristic. |
| 4 | Ridge (α=1.0) | Linear Model | 3.676 | 3.184 | -0.492 | Weak | Linear assumption fails on discrete points. Assumes smooth decline; reality has cliffs. Underfitting: Test MAE worse than Heuristic by 0.085 MAE. |
| 5 | Ridge (α=100.0) | Linear Model | 3.727 | 3.224 | -0.503 | Weaker | Strong regularization worsens fit (vs α=1.0). Test MAE +0.040 worse. Problem isn't variance; it's model misspecification (linear can't capture cliff edges). |
| 6 | Predict Median | Trivial Baseline | 5.505 | 4.803 | — | Floor | Always predicts median (1.0 pt). Robust to outliers but ignores all features. 17% better than mean due to zero-inflation distribution. |
| 7 | Predict Mean | Trivial Baseline | 6.252 | 5.784 | — | **Absolute Floor** | Always predicts mean (5.534 pts). Captures zero information. Establishes the bare minimum bar to beat. All ML models beat this by >50%. |

---

## Model Rankings by Test MAE

```
Rank 1: RF (n=50, d=5)          MAE = 2.882  ✅ USE THIS
         ├─ 7% better than Grid Heuristic (2.882 vs 3.099)
         ├─ 10% better than Ridge (α=1.0)
         └─ 50% better than Predict Mean

Rank 2: RF (n=100, d=10)        MAE = 2.885
         ├─ 0.1% worse than n=50 (essentially tied, but overfits)
         └─ 7% better than Grid Heuristic

Rank 3: Grid Heuristic          MAE = 3.099
         ├─ Domain knowledge baseline
         └─ 3% better than Ridge (α=1.0)

Rank 4: Ridge (α=1.0)           MAE = 3.184
         ├─ Linear assumption breaks down
         └─ 1.3% worse than Heuristic

Rank 5: Ridge (α=100.0)         MAE = 3.224
         ├─ Stronger regularization hurts
         └─ 1.3% worse than Ridge(α=1.0)

Rank 6: Predict Median          MAE = 4.803
         ├─ Trivial baseline
         └─ All ML models beat this

Rank 7: Predict Mean            MAE = 5.784
         └─ Absolute floor (establishes necessity for ML)
```

---

## Detailed Model Explanations

### **Winner: RF (n=50, d=5) — Test MAE = 2.882**

**Why it wins:**
- Captures **nonlinearity** in F1 points (not linear decline)
- Learns that grid ≤ 5 → high scores (10-20 pts), grid 6-10 → medium (2-8 pts), grid 11+ → low (0-1 pt)
- 50 trees + shallow depth (5) = **conservative ensemble**, avoids memorizing training noise
- **Generalization gap -0.071** (train 2.953 vs test 2.882) = *slight underfitting* (better than overfitting!)
- Beats RF (n=100,d=10) with same test MAE (2.885) but LESS overfitting → more reliable

**Feature importance:**
- Grid: 73.8% (starting position is destiny)
- Rolling avg points (last 3 races): 14.5% (recent form matters)
- Others: <4% (team effects minor)

**Production recommendation:** ✅ Use this model

---

### **2nd Place: RF (n=100, d=10) — Test MAE = 2.885**

**Why it's close but not better:**
- Same nonlinearity learning as n=50
- BUT: 100 trees + depth 10 allow **overfitting** (train MAE 2.496 vs test MAE 2.885 = gap +0.389 = overfitting)
- Model memorizes training patterns that don't generalize well (gap is 5x worse than n=50's gap)
- Test MAE only 0.003 points better than n=50, which doesn't justify the overfitting risk

**Trade-off:** Not worth the overfitting; n=50 is more conservative and stable


---

### **3rd: Grid Heuristic — Test MAE = 3.099**

**Strengths:**
- Pure domain knowledge (no ML needed for this)
- Perfectly interpretable: "Grid 1 → avg 18.98 pts, Grid 5 → avg 7.40 pts, Grid 20 → avg 0.1 pts"
- Good generalization: Gap -0.411 (test < train) suggests robust mapping

**Why RF beats it by 7%:**
- Heuristic treats all P1 starters identically (predict 18.98 pts always)
- RF learns: "This P1 starter is a rookie (low rolling avg) → predict 16 pts; that P1 starter is a veteran (high rolling avg) → predict 24 pts"
- Individual variance + recent form = the 0.217 MAE improvement

**When to use:** Fallback if RF model fails; transparent for stakeholders who want domain logic

---

### **4th: Ridge (α=1.0) — Test MAE = 3.184**

**The problem: Linear assumption on nonlinear data**
- Ridge learns: points ≈ 5.534 - 0.38 × grid (simplified, scales features)
- This predicts: Grid 1 → ~17 pts, Grid 5 → ~14.6 pts, Grid 10 → ~11.7 pts, Grid 15 → ~8.8 pts, Grid 20 → ~6 pts
- **Reality:** Grid 1 drivers score 25 pts (P1), Grid 20 drivers score 0 pts (no points finish)
- **Gap in Ridge's logic:** Assumes smooth linear drop; doesn't capture the discrete cliff edges between grid bands

**Underfitting indicator:** Test MAE (3.184) slightly worse than Heuristic (3.099) = linear model too simple, even with regularization

---

### **5th: Ridge (α=100.0) — Test MAE = 3.224**

**Why stronger regularization made it worse:**
- α=100 restricts coefficients even more than α=1.0
- Results in even smoother predictions (coefficients closer to 0) = even more underfitting
- Test MAE worsened (3.224 vs 3.184 for α=1.0) = +0.040 worse

**Lesson:** The problem isn't overfitting (ridge's strength to control); it's **model class mismatch** (linear can't fit nonlinearity). Regularization can't fix a broken model type.

---

### **6th: Predict Median — Test MAE = 4.803**

**Mechanism:** Always outputs y_train.median() = 1.0 point

**Why it's better than mean:**
- F1 data is zero-inflated (49.5% of drivers score 0)
- Median (1.0) closer to typical value than mean (5.534)
- Test MAE 4.803 vs 5.784 for mean = 17% improvement

**Use case:** Sanity check only. Any real model should beat this.

---

### **7th: Predict Mean — Test MAE = 5.784**

**Mechanism:** Always outputs y_train.mean() = 5.534 points

**Why it's the floor:**
- Captures zero information about individual races
- Treats P1 starter and P20 starter identically
- Establishes the **absolute minimum bar** any model must exceed
- All 6 competitors beat this by >50%

**Use case:** Proves necessity for feature-based models. If your model is worse than this, it's broken.

---

## Cross-Validation Notes

**Why temporal validation (not random k-fold)?**
- F1 season structure matters: teams, drivers, regulations change year-to-year
- Random shuffle would leak future information into training (violates causality)
- Temporal split (2018-2022 train → 2023-2024 test) mimics real prediction: "Given history, predict future"

**Test set characteristics:**
- 451 race entries from 2023-2024 seasons
- ~49.5% score 0 points (DNFs, non-scoring finishes) = realistic distribution
- Similar to training set mean (5.18 pts) and std dev (7.27 pts) = no distribution shift
- Feature matrix after NaN drop: Train 977 rows, Test 451 rows

---

## Key Insights for Lab 3

| Insight | Evidence | Implication |
|---------|----------|-------------|
| **Nonlinearity Matters** | RF beats Ridge by 7.5% on test MAE | F1 points aren't linear; discrete cliffs matter (grid bands) |
| **ML Beats Domain** | RF beats Grid Heuristic by 7% | But only slightly; domain knowledge is 93% of the story |
| **Simpler Generalizes Better** | RF (n=50, gap -0.071) beats RF (n=100, gap +0.389) on generalization | Conservative ensemble > complex overfitting (n=50 preferred) |
| **Linear Insufficient** | Ridge Test MAE 3.184 vs Heuristic 3.099 | Model class wrong for problem (underfits) |
| **Overfitting Visible** | RF(n=100): Train 2.496, Test 2.885 = gap +0.389 | Gap indicator prevents deployment of unstable model |

---

## Recommendation Summary

**For Team Principal:**
> "Use Random Forest (n=50, max_depth=5) for predicting race points. It predicts within ±2.88 points on average. Understand: it works because it learns that grid position is destiny in F1 (73.8% feature importance), but it can't predict crashes (50% of zero scores) or weather surprises. Use for strategic planning, not for guarantees."

**For next iteration:**
- Add weather features (wet/dry/mixed)
- Add crash probability model (separate binary classifier)
- Retrain monthly with rolling window (regulations change)
- Monitor feature drift (new teams, drivers)

---

**Submission Date:** April 18, 2026  
**Last Updated:** April 18, 2026  
**Status:** ✅ Ready for review
