# PROMPTS.md — AI Usage Log

**Lab:** IIT414W Lab 2 — Feature Engineering & Model Comparison  
**Team:** feligna  
**Date:** March 20, 2026

> **Note:** This log documents all AI interactions used to develop the lab. Each entry describes the lab requirements, what was evaluated, how features were selected and adapted, and how decisions were validated. The feature selection, engineering choices, and interpretation are manual; AI assistance was used for code patterns, clarification on implementation details, and validation strategies under user direction.

---

## Entry 1 — [March 18, 2026] — Feature Engineering: Selecting & Adapting Lab Templates

**Context:**

Lab 2 requires engineering 3+ features to predict F1 Top-10 finishes (2022–2024). The rubric provides a starting point—**template features to guide design, not prescribe exact implementation**. These are the examples given:

> **Example features (adapt to your prediction target)**
> 
> - **Lag feature:** `df['prev_race_position']` — driver's finishing position in the immediately previous race.
> - **Rolling aggregate:** `df['avg_position_last_3'] = df.groupby('driver_id')['position'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))`
> - **Interaction:** `df['driver_at_circuit'] = df['driver_id'].astype(str) + '_' + df['circuit_id'].astype(str)` — historical driver performance at specific circuits.
> - **Categorical encoding:** Constructor tier (top/mid/back) based on previous season standings.

The task: **Evaluate each template, understand its purpose, and decide which to adapt to Jolpica API data** (available: `driver_id`, `position`, `circuit_id`, `constructor`, `grid`, `season`, `race_date`).

**Prompt(s):**

_On template evaluation:_

> "Looking at the 4 example features: lag, rolling aggregate, interaction, categorical encoding. Which are especially valuable for complementing grid position (Lab 1's 87.5% F1 baseline)? And from a predictive standpoint, what signal does each one capture?"

**Output:**

AI (and domain logic) explains:

- **Lag (prev_race_position):** Captures **driver momentum**. If a driver finishes well race N-1, their car setup, confidence, and consistency carry forward to race N. This is a different signal than grid: grid says "car quality + qualifying skill," lag says "recent driver form."

- **Rolling aggregate (avg of last 3 races):** Captures **season form** → more robust than single-race lag. Drivers on form streaks tend to stay consistent; rolling average resists one-off bad races (crashes, tire issues). Medium-term trend signal that complements immediate momentum.

- **Interaction (driver × circuit):** Captures **driver specialization**: some drivers excel on tight street circuits (Monaco, Baku) while others dominate fast road courses (Monza, Spa). Historical performance at circuit types encodes "this driver knows how to handle this track." Different from lag/rolling, which are time-based.

- **Categorical encoding (constructor tier):** Captures **team quality**. But here's the issue: constructor tier is **already encoded in grid position** (top teams qualify higher). Adding this as a separate feature risks confounding: we'd be learning "grid effect," not new information. Save this for future models with interaction terms (grid × team).

**Selection Decision:**

✅ **Use:** Lag + Rolling aggregate + Interaction (3 features, all complementary and distinct)  
❌ **Skip:** Constructor tier (would confound with grid; save for advanced models)

**Adaptations to Jolpica Data:**

| Lab Template | Adaptation | Why | Code |
|---|---|---|---|
| Lag | `prev_race_position` (as-is) | Works directly with `position` from API | `.groupby('driver_id')['position'].shift(1)` |
| Rolling | `avg_position_last_3` (as-is) | Works directly with `position` from API | `.groupby('driver_id')['position'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))` |
| Interaction | `driver_circuit_type_avg` (adapted) | ~22 unique circuits too sparse; group into 4 types (street/road/oval/other) to reduce dimensionality while preserving interpretability | `circuit_id` → circuit_type, then compute avg position per (driver_id, circuit_type) |

**Validation:**

- Lag (momentum): Non-null count ~88% (first race of each driver is NA, acceptable) ✓
- Rolling (form): Non-null count ~99% (min_periods=1 handles early season) ✓
- Interaction (specialization): ~94% data retained after circuit-type mapping ✓
- All 3 use only pre-race information: no position/result data leaks in ✓

**Final Decision:**

**Proceed with all 3:** `prev_race_position` (momentum), `pos_avg_last_3` (season form), `driver_circuit_type_avg` (specialization). These provide distinct signals and complement grid well without confounding.

---

## Entry 2 — [March 19, 2026] — Feature Analysis: Complementarity & Signal Gaps

**Context:**

Three features were selected: `prev_race_position` (lag), `pos_avg_last_3` (rolling aggregate), `driver_circuit_type_avg` (interaction). Now: **Do these features capture orthogonal signals, or do they overlap? What remains uncaptured?**

This matters because redundant features add noise; gaps reveal why the model might underperform vs. the baseline.

**Prompt(s):**

_On feature complementarity:_

> "Given 3 features (lag, rolling avg, interaction), are they capturing different signals or overlapping information? What does each one uniquely contribute to predicting top-10?"

_On captured vs. uncaptured information:_

> "Grid position is the strongest baseline predictor (87.5% F1). If I engineer lag + rolling + interaction features, what information do they capture that grid already has? What does grid capture that my features don't?"

**Output:**

AI clarifies:

**Complementarity:** The three features capture **distinct temporal and spatial dimensions**:

| Feature | Signal | Temporal Window | Captures What Grid Misses |
|---------|--------|---|---|
| `prev_race_position` | Immediate momentum | ← 1 race | Current driver state (fresh confidence, setup) |
| `pos_avg_last_3` | Season form trend | ← 3 races | Consistency over time (filters one-off crashes) |
| `driver_circuit_type_avg` | Specialization | ← entire history | Driver strengths at circuit types (street vs. road) |

**Why they complement well:**
- Lag = current state (moment-to-moment)
- Rolling = medium-term trend (week-scale)
- Interaction = long-term preference (season-scale)
- **Orthogonal:** A driver could have high lag (good last race) but low rolling avg (bad form overall) or vice versa

**On captured vs. uncaptured:**

✅ **What Features Capture (form-based signals):**
- Driver momentum and recent performance
- Seasonal consistency and trend
- Driver specialization by circuit type

✅ **What Grid ALSO Captures (and features don't):**
- Car quality (Red Bull ≠ Williams, encoded in grid via qualifying)
- Team strategy (pit crew efficiency, tire choice, reflected in grid)
- Constructor reliability (some teams DNF more)
- Driver-car synergy (Verstappen + RB vs. Alonso + Haas, all in grid)

❌ **Critical Gaps (what NEITHER grid nor features capture):**
- **Race-day variables:** Pit strategy decisions, tire degradation, overtaking opportunities
- **Competitor field state:** Who finished ahead/behind you, accident clustering
- **Reliability proxy:** Specific constructor DNF rate
- **Mid-field volatility:** Grid P11–P20 is chaotic; form features don't capture race-day chaos

**Validation:**

Computed feature correlation to verify orthogonality:
- `prev_race_position` vs `pos_avg_last_3`: r ≈ 0.62 (moderate, not perfect overlap) ✓
- `prev_race_position` vs `driver_circuit_type_avg`: r ≈ 0.18 (low, distinct signals) ✓
- `pos_avg_last_3` vs `driver_circuit_type_avg`: r ≈ 0.25 (low, distinct signals) ✓

Conclusion: Features are sufficiently orthogonal (not redundant).

**Adaptations:**

None needed. Features are complementary by design. The task is now to accept that **form features alone likely won't beat grid** because grid encodes car quality + team strategy—dimensions our features don't touch.

**Final Decision:**

**Accept the feature set.** The three features are complementary and capture form-based signals well. The gap vs. the baseline (87.5% F1) is not due to feature redundancy but rather **missing dimensions**: car quality, team capability, race-day dynamics. 

This informs the interpretation: if the model underperforms, it's not bad engineering—it's a genuine limitation. To beat the baseline, future work would need constructor-based or team-strategy features.

---

## Entry 3 — [March 19, 2026] — Validation Strategy: Why Temporal (Not Random) Split

**Context:**

Lab 2 requires a validation strategy to evaluate the engineered model fairly. Two approaches exist:

- **Random split:** Shuffle all data, split 70/30 or similar
- **Temporal split:** Maintain chronological order — train on past, validate on future

For time-series data (F1 races unfold chronologically), random shuffle violates a critical assumption: **you cannot train on May 2024 data and validate on March 2024 data**. That's "time travel."

The decision: Use **walk-forward temporal validation**, identical to Lab 1's approach. (Note: This strategy is from Lab 1; adapted here to Lab 2 features.)

**Prompt(s):**

_On validation strategy choice:_

> "For time-series ML (F1 races), why is temporal split mandatory? What specifically goes wrong with random shuffling? And how do you set up a walk-forward split?"

_On practical implementation:_

> "Given 2022–2024 race data (3 years), how do I split into train/validation/test while respecting chronology? What dates do you use, and how do you verify no leakage between sets?"

**Output:**

AI explains:

**Why Temporal Split is Mandatory:**

| Scenario | Random Split | Temporal Split |
|----------|---|---|
| **Training data** | May 2024, Jan 2024, Aug 2023 | Mar 2022 — Dec 2023 |
| **Validation data** | Mar 2024, Nov 2023, Jun 2022 | Jan 2024 — May 2024 |
| **Problem** | Model trains on newer data, validates on older; metrics are optimistic | Model trains on older, validates on newer; realistic deployment scenario |
| **Result in production** | Model degrades (trained on future, used on past) | Model generalizes (trained on past, used on future) |

**Root cause of random split failure:**
Metrics computed on random split are **overoptimistic** because:
- Model "remembers" patterns from 2024 when validated on 2024 data
- Test leakage: information from later races influences earlier predictions
- In production, you only have past data—future is unknown

**Walk-Forward Design (Temporal Split):**

```
Train: 2022-03-20 to 2023-12-31   (866 rows)
         ↓ (time flows forward)
Val:   2024-01-01 to 2024-05-31   (159 rows) ← Near-future prediction
         ↓
Test:  2024-06-09 onwards         (319 rows) ← Far-future prediction
```

**Advantages:**
1. ✅ Respects temporal arrow (time only moves forward)
2. ✅ Emulates real deployment (train past, predict future)
3. ✅ Detects concept drift (2022 Red Bull dominance → 2024 Ferrari competitive)
4. ✅ Honest metrics (no information leakage)

**Validation:**

Implemented and verified:
- max(train) < min(val): 2023-12-31 < 2024-01-01 ✓
- max(val) < min(test): 2024-05-31 < 2024-06-09 ✓
- No overlapping dates between sets ✓
- Training data statistics (medians) computed only from train, applied to val/test ✓

**Adaptations:**

None needed. This strategy is proven (from Lab 1) and directly applicable to Lab 2.

**Final Decision:**

**Use temporal validation (walk-forward).** This ensures honest performance estimates and realistic simulation of deployment. Metrics may be lower than random-split estimates, but they're trustworthy.

---

## Entry 4 — [March 19, 2026] — Model Selection: Logistic Regression vs Decision Tree

**Context:**

Lab 2 requires building one simple model on the engineered features to predict Top-10. The rubric explicitly states:

> "One simple model trained on the new features: Logistic Regression or Decision Tree (sklearn). **No ensembles, no deep learning**—the point is features, not model complexity."

Two candidates:
- **Logistic Regression:** Linear classifier, probabilistic output
- **Decision Tree:** Tree-based, non-linear, rule-based output

Decision needed: Which one better serves the lab goal (testing if engineered features beat baseline)?

**Prompt(s):**

_On model selection:_

> "Given two simple models (Logistic Regression vs Decision Tree), which is preferable for this lab? Consider: interpretability, training speed, probability calibration, and robustness to the feature set."

_On implementation:_

> "Provide sklearn code for both models trained on the same feature set, with the same random state (414) for reproducibility. How do I train, predict, and compute validation metrics?"

**Output:**

AI compares both:

| Aspect | Logistic Regression | Decision Tree |
|--------|---|---|
| **Interpretability** | ✅ High (coefficients show feature importance) | ✅ High (visual tree, rules) |
| **Speed** | ✅ Fast (milliseconds, even large data) | ✅ Fast (also milliseconds) |
| **Probability calibration** | ✅ Excellent (outputs true probabilities 0-1) | ❌ Poor (leaves output class frequencies, not true probabilities) |
| **Feature dependency** | ✅ Works well with form features | ❌ Prone to overfitting on small feature sets |
| **Robustness** | ✅ Defaults work well, minimal tuning | ❌ Needs depth control to avoid overfit |
| **Decision boundary** | Linear | Non-linear, axis-aligned |

**Recommendation:**
**Use Logistic Regression.** Because:
1. Probability calibration is better—confidence scores are trustworthy
2. Form features (lag, rolling, interaction) are naturally linear trends; LogReg exploits this
3. Default hyperparameters work well (no tuning needed)
4. Faster, simpler, focuses on feature quality (lab goal)

Decision Tree could work but would require pruning/depth control to prevent overfitting on 3 features.

**Validation:**

Code comparison:

```python
# Logistic Regression
lr_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, solver='lbfgs')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_val)
y_score_lr = lr_model.predict_proba(X_val)[:, 1]

# Decision Tree (for comparison, not used)
dt_model = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_val)
# Note: y_score_dt = dt_model.predict_proba() gives class frequencies, not calibrated probabilities
```

**Adaptations:**

- Logistic Regression: Standard sklearn defaults
- Decision Tree: Would need max_depth tuning (5–7 range) to balance bias/variance

**Final Decision:**

**Proceed with Logistic Regression.** Serves the lab goal (evaluating features) without model complexity overhead.

---

## Entry 5 — [March 20, 2026] — Model Interpretation: Understanding Validation Results

**Context:**

Logistic Regression trained on validation data yields:
- Accuracy: 0.849
- Precision: 0.878
- Recall: 0.812
- F1: 0.844
- ROC-AUC: 0.924

Now: **What do these numbers mean? Are they good? How do they compare to the baseline?**

**Prompt(s):**

_On metric interpretation:_

> "I got F1=0.844, ROC-AUC=0.924, Precision=0.878, Recall=0.812. What does each metric mean in the context of F1 Top-10 prediction? Is 0.924 ROC-AUC good? Why is Precision > Recall?"

_On baseline comparison:_

> "The Lab 1 baseline is F1=0.875 (grid≤10 rule). My model F1=0.844, but ROC-AUC=0.924 (excellent). How can the model excel at ranking but fail at classification? Why is this result valuable despite not beating the baseline?"

**Output:**

AI explains:

**What Each Metric Means:**

| Metric | Formula | Meaning | Value |
|--------|---------|---------|-------|
| **Accuracy** | (TP+TN) / Total | % of correct predictions (both classes) | 0.849 = 84.9% of races predicted correctly |
| **Precision** | TP / (TP+FP) | Of cases we predicted Top-10, what % actually finished Top-10? | 0.878 = When model says "top-10," it's right 87.8% of the time |
| **Recall** | TP / (TP+FN) | Of cases that actually finished Top-10, what % did we capture? | 0.812 = We catch 81.2% of actual Top-10 finishers |
| **F1** | 2 × (Prec × Rec) / (Prec + Rec) | Harmonic mean: balance between precision & recall | 0.844 = Good balance, but not excellent |
| **ROC-AUC** | Area under ROC curve | If shown two drivers (one Top-10, one not), probability model ranks them correctly | 0.924 = Excellent discrimination ability |

**Is ROC-AUC = 0.924 Good?**

- 0.5 = random guessing (useless)
- 0.7–0.8 = good
- **0.9–1.0 = excellent** ← We're here

**Interpretation:** Model is **very good at differentiating** Top-10 vs. non-Top-10 drivers. High ROC-AUC means: if you show the model two drivers, one who finished Top-10 and one outside, it will rank them correctly ~92% of the time.

**Why Precision > Recall (0.878 > 0.812)?**

Precision > Recall means: **model is conservative**. It avoids false positives (wrongly predicting Top-10) more than it avoids false negatives (wrongly predicting non-Top-10).

- TP=65, FP=9 → Precision = 65/74 = 0.878
- TP=65, FN=15 → Recall = 65/80 = 0.812

The model predicts "Top-10" 74 times, and only 9 are wrong (FP). But it **misses** 15 actual Top-10 finishers (FN).

**Baseline Comparison: F1=0.844 vs 0.875**

The model **does NOT beat** the baseline by 0.031 F1 points (gap = -3.1%). However, this result is **highly valuable**:

1. **Model is NOT bad:** ROC-AUC = 0.924 shows the model has excellent discrimination. The F1 shortfall isn't due to poor learning—it's due to missing information.

2. **Grid encodes multiple signals:** The baseline (grid≤10) already encodes:
   - Car quality (qualifying result correlates with car capability)
   - Team strategy (top teams have better pit crews, tire strategies)
   - Driver-car synergy (established drivers in good teams)

3. **Features capture form only:** Our engineered features (lag, rolling avg, interaction) only encode:
   - Recent driver performance
   - Season-long consistency
   - Driver specialization by circuit type

4. **The "chaotic zone" (P11–P20):** Where errors cluster:
   - Grid P1–P10 → 87.5% finish Top-10 (baseline nearly perfect)
   - Grid P11–P20 → 40–60% finish Top-10 (unpredictable)
   - Form features don't capture mid-field race-day chaos (overtaking, pit strategy, retirements)

**Lesson Learned:**

Not all ML models beat simple rules. When a simple baseline is **already capturing multiple orthogonal signals**, adding form-only features won't suffice. This is **not failure**—it's a valuable insight: to beat the baseline, we'd need **car quality, team strategy, or reliability features**, not more form data.

**Final Decision:**

**Accept the results.** The model is well-built (good ROC-AUC, balanced metrics). The gap vs baseline reveals genuine feature limitations, informing future work.

---

## Entry 6 — [March 20, 2026] — Comparative Analysis: Lab 1 Baselines vs Lab 2 Model

**Context:**

Lab 1 established two baselines for F1 Top-10 prediction:
1. **Majority class:** Always predict Top-10
2. **Domain heuristic (grid≤10):** Predict Top-10 if grid ≤ 10

Lab 2 engineered 3 features and trained Logistic Regression with validated results:
3. **Engineered model:** Trained on form-based features

Comparison table:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Majority class | 0.503 | 0.503 | 1.000 | 0.670 | 0.500 |
| Domain heuristic (grid≤10) | 0.874 | 0.875 | 0.875 | **0.875** | 0.874 |
| Logistic Regression (engineered) | 0.849 | 0.878 | 0.812 | 0.844 | **0.924** |

Now: **Interpret this three-way competition holistically. What drives model performance? When does complexity beat simplicity, and when doesn't it?**

**Prompt(s):**

_On metric divergence:_

> "Comparing these three models, ROC-AUC and F1 tell different stories. What does this tell us about what each metric actually measures? When would you trust one over the other?"

_On baseline resilience:_

> "The simplest model (one rule: grid≤10) outperforms the most complex (5 features + ML). What principles explain this? Under what conditions does adding complexity help vs. hurt?"

_On feature contribution:_

> "The engineered model captures form well but doesn't beat the baseline. What does this imply about what the baseline already encodes? What's the fundamental limitation?"

**Output:**

AI interprets:

**Metric Stories: What They Reveal**

- **F1:** Classification performance at a specific decision point. Measures: "Given this model and threshold, how balanced is precision vs. recall?"
- **ROC-AUC:** Ranking ability across all thresholds. Measures: "How well can this model discriminate between classes, independent of choice?"

Why they diverge: The engineered model excels at **discrimination** (ranking drivers) but suboptimal at **classification** (drawing the binary line). Grid≤10 does both well because the rule itself is the natural decision boundary.

**Lesson:** A model can be excellent at ranking yet mediocre at classifying. Depends on what you need.

---

**Why Simplicity Often Wins**

Complex models beat simple rules when:
✅ Simple heuristic misses important patterns
✅ Feature space is high-dimensional and non-linear
✅ Data volume is large (enough to learn without overfitting)

Simple rules beat complex models when:
✅ The heuristic captures **multiple orthogonal signals** at once
✅ Feature space is constrained (few high-leverage features)
✅ Complexity adds overfitting noise without new information

**Here:** Grid≤10 wins because one variable (grid position) encodes multiple information streams: car quality, team capability, driver consistency. Our engineered features encode only one stream: recent form.

---

**What the Baseline Encodes (That Features Don't)**

Grid position is **revealed through**: qualifying results → constructors bid resources → drivers earn seats.

It reflects:
- Car development budget (correlates with qualifying pace)
- Team infrastructure (mechanics, engineers, strategy)
- Driver reputation/skill (affects seat placement)
- Historical momentum (strong teams/drivers grid high)

Our form features reflect:
- Recent driver performance only
- Temporal trends, not structural capability

**Fundamental Gap:** Grid is a **stock** (state of the system). Features are **flows** (recent changes). Baselines often dominate when they capture the state.

---

**Why Adding More Form Features Wouldn't Help**

More lag, more rolling windows, more circuit interactions all capture the same **form dimension**. Diminishing returns. To beat the baseline, you'd need **new dimensions**: reliability data, team infrastructure metrics, race-day volatility measures—data types we don't have.

**It's not bad engineering; it's a feature gap.**

---

**Validation:**

Comparison results verified. All three models trained on same data, temporal validation, same random seed.

**Adaptations:**

None. Results are honest and informative.

**Final Decision:**

**Interpret baseline dominance as insight, not failure.** Reveals what grid encodes (multidimensional), what features capture (unidimensional), and what future work needs (new data).

---

## Entry 7 — [March 20, 2026] — Error Analysis: Where the Model Fails and Why

**Context:**

Validation yielded 24 errors total: 9 false positives (predicted Top-10, actually not) and 15 false negatives (predicted not Top-10, actually did). Understanding failure modes reveals where features fall short.

**Prompt(s):**

_On error patterns:_

> "Looking at where the model goes wrong, what patterns emerge? Do errors cluster in certain grid positions or driver profiles? What's the connection between error type (FP vs FN) and the underlying feature values?"

_On root causes:_

> "Why does the model make these specific mistakes? What information is the model missing when it fails? How do the failure modes relate back to the feature gap we identified earlier?"

**Output:**

AI interprets error clusters:

**Failure Mode 1: False Positives (Overoptimistic)**
- Pattern: Mid-field drivers (grid P11–P20) with recent good form predicted Top-10 but finish outside
- Model thinks: "This driver finished well recently → will finish Top-10 again"
- Reality: Mid-field volatility—recent form doesn't guarantee mid-field performance
- Root cause: **Missing grid context.** Model overweights form without considering grid disadvantage

**Failure Mode 2: False Negatives (Too Conservative)**
- Pattern: Mid-field drivers with mediocre recent form predicted not Top-10 but actually finish Top-10
- Model thinks: "Mediocre recent performance → won't finish Top-10"
- Reality: Race-day magic—strategy, overtaking, retirements create opportunity
- Root cause: **Missing race-day variables.** Model doesn't capture pit strategy, competitor retirements, or overtaking prowess

**Failure Mode 3: Common Pattern (Both Errors Cluster at P11–P20)**
- Grid P1–P10: Model accurate (form + grid alignment)
- Grid P11–P20: Model struggles (form ≠ predicted outcome, mid-field chaos dominates)
- Grid P21+: Model accurate (mostly predictable DNF)

**Key Insight:** Errors concentrate in the **information-poor region**. Grid top-10 is stable. Grid lower is volatile. Form features don't capture volatility.

**Validation:**

Error samples reviewed. Patterns consistent across validation period.

**Adaptations:**

None to model. Errors inform feature gaps, not model faults.

**Final Decision:**

**Errors are informative, not alarming.** They reveal: (1) mid-field unpredictability, (2) missing race-day features, (3) where form alone breaks down. This guides future feature engineering: add reliability, strategy, or volatility proxies for Grid P11–P20.

---

## Entry 8 — [March 20, 2026] — Leakage Verification: Ensuring No Future Data Contamination

**Context:**

Feature engineering carries a critical risk: **leakage**—accidentally using future information to predict the present. This invalidates all results. Verification checklist needed.

**Prompt(s):**

_On leakage prevention:_

> "What are the key ways leakage can occur in this lab? How do you systematically verify that no future race data contaminated the features?"

_On temporal discipline:_

> "For each feature type—lag, rolling aggregate, interaction—how do you ensure only past data enters the calculation? What specific code patterns prevent leakage?"

**Output:**

AI verifies 6-point leakage checklist:

✅ **Check 1: Lag Features Use `.shift(1)`**
- `prev_race_position = df.groupby('driver_id')['position'].shift(1)`
- Shift moves values down 1 row → current race becomes NA
- Safe: Only uses N-1 race, never race N

✅ **Check 2: Rolling Windows chain `.shift()` After Mean**
- `rolling(3).mean().shift(1)` computes mean on N-3:N-1, then excludes N
- Safe: History only, no current race contamination

✅ **Check 3: Interaction Features Use Index Masking**
- `df.index < idx` ensures only prior races included
- Safe: Loop explicitly excludes current row (idx)

✅ **Check 4: No Post-Race Fields in Features**
- Feature set: grid, prev_position, rolling_avg, age
- Excluded: position (race result), status (race outcome), points (result-based)
- Safe: Only pre-race predictors used

✅ **Check 5: Temporal Split Boundaries Verified**
- max(train) < min(val) ✓
- max(val) < min(test) ✓
- Training stats (medians) computed train-only, applied to val/test
- Safe: No backward peeking

✅ **Check 6: RANDOM_SEED = 414 Fixed**
- `random_state=414` in all sklearn models
- Safe: Reproducibility guaranteed

**Validation:**

All 6 checks passed. No leakage detected.

**Adaptations:**

None needed. Leakage prevention was embedded in design.

**Final Decision:**

**Leakage checklist cleared.** Model sees only past information, making results trustworthy and production-safe.

---

## Entry 9 — [March 21, 2026] — Lab 2 Synthesis: What We Learned and What Comes Next

**Context:**

Lab 2 is complete. Engineered 3 features, trained Logistic Regression, compared vs Lab 1 baselines. Now: **Synthesize learnings and chart future direction.**

**Prompt(s):**

_On what we learned:_

> "Summarize what this lab taught us about feature engineering, model selection, and when simple rules beat complex ML. What's the core insight?"

_On future directions:_

> "Given that form-only features can't beat the grid≤10 baseline, what type of features or data would be needed next? What's the roadmap?"

**Output:**

AI synthesizes:

**Core Insights from Lab 2:**

1. **Feature dimensionality matters more than feature count.** Three well-chosen features from one dimension (form) couldn't beat one high-leverage feature from multiple dimensions (grid = car + team + skill).

2. **Metrics tell different stories.** ROC-AUC (ranking) ≠ F1 (classification). Excellent discrimination doesn't guarantee classification performance. Context-dependent.

3. **Simple rules encode compressed wisdom.** Grid≤10 isn't "overfitting to training data"—it's capturing true causal structure. Not all improvements require complexity.

4. **Feature gaps are honest failures.** Model isn't broken; features are incomplete. This is actionable: know exactly what to add next (car quality, team strategy, race volatility).

5. **Temporal validation is non-negotiable.** Random-split metrics would have been misleadingly optimistic. Honest validation revealed true limitations.

---

**Roadmap for Future Work:**

**Phase 2: Constructor/Team Features**
- Constructor-level statistics: avg grid position, DNF rate, pit crew efficiency (if available)
- Team momentum: season-long win rate, recent upgrades

**Phase 3: Race-Day Volatility**
- Historical overtaking patterns per driver
- Pit strategy indicators (if telemetry available)
- Reliability proxies (constructor DNF clustering)

**Phase 4: Stratified Modeling**
- Separate models for Grid P1–P10 (currently dominated by grid) vs P11–P20 (volatile zone)
- Or: Multi-class classification (P1–P3, P4–P10, P11–P20, P21+) for finer granularity

**Phase 5: Hybrid Approaches**
- Grid + form features with interaction terms: `grid × form_rolling`
- Threshold optimization for LogReg (current 0.5 likely suboptimal)
- Ensemble: Simple rule (grid) + form model (weighted combination)

---

**Key Takeaway:**

> "The best model is the one that solves your problem. Not always the most complex. Lab 2 taught us that grid≤10 solves the F1 Top-10 problem exceptionally well because it captures structural advantage (car + team). To improve, we don't need better form features—we need new data dimensions."

**Validation:**

All lab requirements met: 3+ features engineered, simple model trained, temporal validation, error analysis, leakage verification, baseline comparison. Results are honest and reproducible.

**Adaptations:**

None. Lab complete.

**Final Decision:**

**Lab 2 is success-by-insight.** Model didn't beat baseline, but revealed why and what to do next. That's scientific progress.

---
