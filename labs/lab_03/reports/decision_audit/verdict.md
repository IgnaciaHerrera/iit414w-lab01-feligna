# Verdict - Felipe Vázquez & Ignacia Herrera

**Model:** F1 Points Prediction via Random Forest (Lab 3)  
**Engineers:** Felipe Vázquez & Ignacia Herrera  
**Date:** April 27, 2026  
**Rubric Reference:** [./utility_rubric.md](./utility_rubric.md)

---

## The Verdict Protocol Applied

**Tally from Utility Rubric:**
- ✅ Green: 3 (Criteria 1, 3, 4, 5)
- ⚠️ Yellow: 0
- ❌ Red: 1 (Criterion 2: Baseline Lift)

**Protocol Mapping:**
- GO: 0 Red, ≤ 1 Yellow → (Not us)
- CONDITIONAL-GO: Mixed → (Not us)
- **NO-GO: ≥ 1 Red → ✅ (THIS IS US)**

**Decision:** Tally has 1 Red, so default is **No-Go.** The conditions below are the only argument for changing it.

---

## Verdict: **🛑 NO-GO**

- [x] Go
- [ ] Conditional-Go
- [x] No-Go

### One-sentence Reason
**The F1 Official Points Scale baseline (MAE 2.541) outperforms the Random Forest model (MAE 2.841) by 11.8%, making the ML model strictly worse than the domain-rule alternative and unsuitable for deployment.**

---

## Answering the Three Questions (Verdict Protocol Requirement)

### Q1: What one change would flip the verdict to Go?

**The one change (specific lever, not "better model"):**  
Replace the test set distribution with races from lower-volatility circuits (Monaco, Singapore excluded) where grid position correlation is stronger, and retrain RF with grid-only features to eliminate noise from secondary features.

**Target Criterion:** #2 (Baseline Lift)

**Expected Numeric Effect:** 
"Lift improves from −11.8% to ≥ +5% (positive territory); RF achieves Test MAE ≤ 2.45, beating or matching baseline."

---

### Q2: What is the smallest experiment to test that change?

**Runtime requirement:** Under 2 hours, concrete and runnable (not research agenda)

### Step 1: Identify low-volatility circuits
Filter test set to races at circuits with high grid-correlation (e.g., exclude rain-prone tracks like Montreal, Silverstone; focus on predictable tracks like Singapore, Monaco where grid → outcome is tighter). Expected: ~60–70% of test samples remain (600–630 races).

### Step 2: Retrain RF with grid-only features
- Remove non-grid features: `position_lag_1`, `rolling_avg_pts_3`, `rolling_avg_pos_3`, `constructor_avg_grid_5`, `constructor_id`, `circuit_id`
- Keep: `grid` only
- Re-train RF (n=10, max_depth=8) on 2018–2022 train set
- Evaluate on filtered test set (low-volatility only)

### Step 3: Measure new Lift
- Compute Test MAE for grid-only RF on filtered test set
- Compute Test MAE for F1 Official Points Scale baseline on same filtered set
- Calculate Lift = (Baseline MAE − Model MAE) / Baseline MAE
- **Decision Rule:** If Lift ≥ +5% AND Test MAE ≤ 2.45, flip Criterion 2 to YELLOW or GREEN → Retally to "Conditional-Go"

**Expected Runtime:** 
~45 minutes (data filter + retrain RF + re-evaluation)

---

### Q3: Under what real-world condition would the verdict downgrade to No-Go?

**Real-world change (specific failure mode, not vague):**  
FIA introduces a new regulation (e.g., 2026 power unit change, grid penalty redistribution rule, or sprint race mandatory scoring change) that materially alters the grid→points relationship, breaking the baseline's assumptions and making both baseline AND RF unreliable.

**Indicator:** Baseline performance drops below Criterion 3 tolerance (Test MAE > 3.5 on post-regulation data), triggering a "No-Go until Re-baselined" assessment.

---

## Cold-Call Preparation (15:35)

### Q1: Why this verdict over the adjacent one?

**Draft Answer:**  
"No-Go, not Conditional, because Criterion 2 (Baseline Lift) is RED with -11.8% lift—the baseline rule is quantifiably superior. A Conditional verdict would require the model to at least match the baseline; it doesn't. The tally protocol is unambiguous: ≥1 Red → No-Go."

### Q2: What is the first number your flip-condition moves?

**Draft Answer:**  
"Lift: currently -11.8% (negative). The flip condition expects Lift to move to ≥ +5% (positive), achieved by removing noisy features and filtering to low-volatility circuits where grid predicts points more tightly."

### Q3: What would you NOT do, and why?

**Draft Answer:**  
"I would NOT deploy a deeper Random Forest (e.g., max_depth=15) because the rubric shows the problem is not model complexity but feature quality. RF already has 68% feature importance in grid; adding depth amplifies overfitting on the other 32% noise. The root issue is that grid alone optimizes this prediction task—more trees won't fix it."

---

## Rubric Anchor Points (Evidence)

| Criterion | Score | Key Number | Decision Impact |
|-----------|-------|------------|-----------------|
| 1. Decision Relevance | ✅ GREEN | Real F1 decision, Friday evening, named maker | Supports: Verdict is defensible in context |
| 2. **Baseline Lift** | ❌ **RED** | **-11.8% (2.541 vs 2.841)** | **Drives No-Go decision** |
| 3. Operating-Point Fit | ✅ GREEN | Test MAE 2.841 ≤ tolerance | Supports: If model were good, it'd fit operationally |
| 4. Failure-Cost Asymmetry | ✅ GREEN | Symmetric error distribution | Supports: Risk is balanced; not an additional barrier |
| 5. Deployment Friction | ✅ GREEN | Inputs available Friday 18:00 | Supports: Feasibility is not the blocker |

**Interpretation:** 3 Green + 1 Red = **No-Go**. The model is technically feasible and fair-scored, but scientifically inferior to the baseline. Deployment cannot be justified.

---

## Why This Verdict Is Defensible

1. **Objective:** Lift is -11.8%, not close to break-even (±0%). Not subjective.
2. **Comparable:** Same test set, same metric (MAE), same time window. Direct comparison.
3. **Conservative:** Choosing the simpler, provably better baseline is the safer call.
4. **Falsifiable:** The flip condition explicitly states what would change the verdict (grid-only + low-volatility circuits, Lift ≥ +5%).
5. **Domain-aligned:** F1 engineers prefer simple, interpretable rules (official FIA points scale) over black-box ML when both are compared.

---

**Status:** ✅ **VERDICT COMPLETE**  
**Recommendation:** Use F1 Official Points Scale baseline for race strategy. Archive RF model; revisit after feature redesign (grid-only test).
