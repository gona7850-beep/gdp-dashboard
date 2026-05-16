---
description: Propose the next batch of experiments using active learning over a fitted session.
argument-hint: <session_id> <batch_size> [region_json]
---

You are running the **generate-doe** workflow for AlloyForge.

## Inputs
- Session: `$1`
- Batch size: `$2`  (typical: 4–8)
- Region filter JSON: `$3`  (optional; `{element_bounds, process_bounds}` to
  restrict the search region)

## Steps

1. Load the session. Verify the validator has run and ECE < 0.15. If not,
   warn the user that AL proposals may be unreliable.
2. Initialize an `ActiveLearner` from the session.
3. Choose acquisition:
   - **Single objective:** `sample_uncertainty` with diversity penalty.
   - **Multi-objective:** `sample_pareto_improvement` (qEHVI) with the
     training-set Pareto front as the reference.
4. If a region filter is provided, apply it during candidate generation.
5. Propose `$2` candidates. For each, compute the NN distance to the closest
   training point.
6. Run the feasibility checker. If > 50% of proposals are infeasible, loosen
   the diversity penalty or widen the region — print a diagnostic.

## Output

A CSV at `reports/doe_batch_<timestamp>.csv` with columns:
`composition_<element>...`, `process_<param>...`, `acquisition_score`,
`predicted_<target>_mean`, `predicted_<target>_std`, `nn_distance`,
`feasible`. Sorted by `acquisition_score` descending.

Plus a markdown brief at `reports/doe_batch_<timestamp>.md` summarizing why
each candidate was chosen (uncertainty? Pareto frontier? diversity?).
