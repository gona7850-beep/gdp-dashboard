---
description: Validate a list of candidate designs against feasibility, domain-of-applicability, and calibration trust.
argument-hint: <session_id> <candidates_csv>
---

You are running the **validate-design** workflow for AlloyForge.

## Inputs
- Session: `$1`
- Candidates CSV: `$2`  (composition columns + optional process columns,
  one row per candidate)

## Steps

1. Load the session. Recover the fitted model, calibrator, and training data.
2. Read the candidates CSV. Verify all element columns are recognized.
3. For each candidate:
   - **Feasibility:** run `default_checker.check(...)`. Record all violations
     with severity.
   - **Domain-of-applicability:** compute NN distance to the *training feature
     space*. Flag if > training 95th percentile.
   - **Predicted properties:** `ForwardModel.predict(...)` → `(μ, σ)`.
   - **Calibrated interval:** apply the conformal calibrator at α=0.1.
   - **Risk-adjusted score** vs the spec, if a spec was used to generate them.
4. Cross-check: for any candidate with `feasibility.passed == True` AND
   `DoA flag == True`, write a strong warning — the candidate looks fine
   physically but the model has no evidence base for it.
5. Hand the full list to `LLMAssistant.review_candidates` for a written
   metallurgical critique.

## Output

A markdown report at `reports/validation_<timestamp>.md` with one row per
candidate and four columns:
- **Predicted properties** with 90% intervals.
- **Feasibility status** (pass / which constraints violated).
- **Domain status** (in / out of training distribution).
- **Recommendation** (synthesize / re-design / collect data first).

Plus a JSON summary at `reports/validation_<timestamp>.json` for programmatic
downstream use.
