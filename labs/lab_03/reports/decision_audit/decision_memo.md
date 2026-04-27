**To:** Head of Strategy / Race Director  
**From:** Felipe Vázquez & Ignacia Herrera  
**Date:** April 27, 2026  
**Re:** Deployment Recommendation: F1 Points Prediction Tool for Race Strategy

---

## What this tool does

This tool uses driver grid position (qualifying result) and recent race history to predict how many championship points a driver is likely to score on race day. It's designed to help the strategy team set realistic target finishes on Friday evening—before strategy is locked in for qualifying—and to adjust tire and fuel plans based on expected point outcomes.

---

## What we found

We tested the tool against our current manual approach: looking at grid position and knowing "historically, drivers from pole position score around 20 points, drivers from P10 score around 1 point." 

**Our manual approach (current baseline):** On races from 2023-2024, it was right within 2.5 points on average.

**Our new tool:** On the same races, it was right within 2.8 points on average.

**Translation:** The tool is **worse**, not better. It adds 0.3 points of error compared to what we already do by eye. For a 25-point race, that's an 11.8% accuracy loss.

---

## What we recommend

**Use the baseline approach (grid position rules)** that you're already using. It's simple, transparent, and more accurate.

Do not deploy this new tool for race strategy decisions.

---

## What we do NOT recommend

I do not recommend deploying this tool unless we can prove that **using only grid position with no other feature noise reduces the error below 2.5 points on low-variability circuits** (circuits where weather doesn't change the grid-to-finish relationship). If that condition is met, the tool might earn a second look.

---

## What we still don't know

- Does this tool perform better at specific circuit types (e.g., high-speed vs. street circuits)?
- How reliable is the tool during rain or with new rule changes?
- Do we need a completely different approach (e.g., predicting finishing position first, then converting to points)?

---

## Appendix: Technical Traceability
*(For Engineering Review Only)*

| Metric | Value | Source |
|--------|-------|--------|
| Baseline (F1 Official Points Scale) | Right within 2.541 points | Notebook cell #VSC-0624321e, Execution 22 |
| Tool Accuracy (Random Forest n=10, d=8) | Right within 2.841 points | Notebook cell #VSC-ca754dc2, Execution 23 |
| Accuracy Loss | 0.3 points (11.8% worse) | (2.841 - 2.541) / 2.541 |
| Test Period | 909 races (2023-2024 seasons) | Temporal split verified |
| Data Clean | Grand Prix only; sprints removed | 2,876 total races after filter |

---

**Verdict:** Archive the tool. Use existing grid-position judgment for strategy. Revisit if circuit-specific or feature-simplified versions show promise.
