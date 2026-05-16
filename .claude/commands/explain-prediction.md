---
description: Produce SHAP-based local + global explanations for a candidate, plus an LLM-written metallurgical interpretation.
argument-hint: <session_id> <composition_json> [target]
---

You are running the **explain-prediction** workflow for AlloyForge.

## Inputs
- Session: `$1`
- Composition: `$2`  (JSON dict of element → atomic fraction)
- Target: `$3`  (optional; if omitted, explain all targets)

## Steps

1. Load the session.
2. Build a `pd.DataFrame` from the composition. Verify it sums to 1.0 (warn
   if off by > 0.5%; renormalize).
3. Run `Explainer.explain(composition)` to get SHAP local values.
4. Run `Explainer.global_importance()` for context.
5. Run `Explainer.counterfactual(...)` to find the smallest composition tweak
   that flips the prediction across a user-specified threshold (default:
   ±1σ shift in the predicted mean).
6. Compose a payload of {prediction, intervals, SHAP top-5, counterfactual}
   and pass to `LLMAssistant.interpret_prediction(...)` for a written
   metallurgical narrative.

## Output

A markdown report at `reports/explain_<timestamp>.md` with:
- The composition and predicted properties (with 90% intervals).
- SHAP top-5 contributing features per target (signed).
- Counterfactual: smallest atomic-fraction tweak to move the prediction.
- LLM interpretation: 2–3 paragraphs linking SHAP signals to known
  metallurgy (carbide formers → secondary hardening, etc.).

If the candidate is out-of-domain (NN distance flag), prefix the LLM output
with a warning that the interpretation is speculative.
