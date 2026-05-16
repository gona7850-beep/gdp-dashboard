---
description: Run multi-objective inverse design using a fitted session and a target spec.
argument-hint: <session_id> <design_spec_json>
---

You are running the **inverse-search** workflow for AlloyForge.

## Inputs
- Session: `$1`  (must point to a fitted `ForwardModel` + calibrator)
- DesignSpec JSON: `$2`  — path to a JSON file matching the `DesignSpec`
  dataclass: `{objectives, element_bounds, target_values, weights,
  process_bounds, risk_lambda}`.

## Steps

1. Load the session. Verify `ForwardModel.fitted_ == True`. If not, exit and
   tell the user to run `/run-prediction` first.
2. Load the DesignSpec and validate:
   - All target keys exist in the model's targets.
   - All elements in `element_bounds` exist in the training composition columns.
     If a new element is requested, warn that it is **out of domain**.
   - `risk_lambda` is set; if missing, default to 1.0 and warn.
3. Run `InverseDesigner.run_nsga2(pop_size=64, n_gen=80)`. For larger problems
   (> 6 design variables), bump to `pop_size=128, n_gen=120`.
4. Apply the `default_checker` feasibility filter on the final population.
5. Rank by risk-adjusted aggregate score.
6. For the top 10 feasible candidates, compute SHAP local explanations and a
   counterfactual ("what's the smallest change that drops it off the Pareto front?").
7. Hand the top 5 to the LLM design reviewer (`LLMAssistant.review_candidates`)
   for a metallurgical sanity check.

## Output

A markdown report at `reports/inverse_<session_id>_<timestamp>.md` containing:
- The Pareto front (compositions + predicted properties + intervals).
- Feasibility status per candidate.
- SHAP top-3 features per candidate.
- LLM reviewer notes per candidate.
- A recommended top-3 for synthesis with risk callouts.

Also write a CSV at `reports/inverse_<session_id>_<timestamp>_candidates.csv`
for downstream use by `/generate-doe`.
