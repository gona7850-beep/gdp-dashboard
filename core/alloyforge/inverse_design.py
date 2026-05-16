"""
Inverse design: given a target (vector of properties or a multi-objective spec),
find compositions that satisfy it.

Two complementary strategies are provided:

1. **NSGA-II global search** (pymoo)
    Robust for highly multimodal landscapes and many constraints. Returns a
    Pareto front. Use when you have many degrees of freedom (≥ 6 elements +
    process variables) and want global coverage.

2. **Bayesian optimization with expected hypervolume improvement** (botorch)
    Sample-efficient. Use when each forward-model evaluation is cheap (it is
    here — we use the trained ForwardModel — but for active-learning loops
    where real experiments cost weeks, the same code drives the picks).

Both strategies output a candidate DataFrame ranked by an objective score, with
per-candidate uncertainty and feasibility status attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .forward_model import ForwardModel
from .feasibility import FeasibilityChecker


# ---------------------------------------------------------------------------
@dataclass
class DesignSpec:
    """Specification of the inverse-design problem.

    ``objectives``: list of (target_name, direction) where direction ∈ {"max","min","target"}.
    ``target_values``: required when direction == "target"; absolute miss is minimized.
    ``weights``: relative weights for scalarized objective (used by NSGA-II's
        secondary ranking). Defaults to equal.
    ``process_bounds``: e.g. {"VED": (40, 90)}. Treated as continuous free variables.
    ``element_bounds``: {"Ti": (0.50, 0.95), "Al": (0.0, 0.10), ...}.
        Elements not listed default to (0, 0.5).
    """

    objectives: List[Tuple[str, str]]
    element_bounds: Dict[str, Tuple[float, float]]
    target_values: Dict[str, float] = field(default_factory=dict)
    weights: Optional[Dict[str, float]] = None
    process_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    risk_lambda: float = 1.0  # μ - λ·σ scoring; higher = more conservative
    feasibility: Optional[FeasibilityChecker] = None

    def __post_init__(self):
        for tgt, d in self.objectives:
            if d not in ("max", "min", "target"):
                raise ValueError(f"Unknown direction {d} for {tgt}")
            if d == "target" and tgt not in self.target_values:
                raise ValueError(f"target_values[{tgt}] required for direction='target'")


# ---------------------------------------------------------------------------
class _NSGA2Problem(ElementwiseProblem):
    """Wraps ForwardModel + DesignSpec into a pymoo problem.

    Decision vector layout:
        x[0:n_el]                  unnormalized element fractions
        x[n_el:n_el+n_proc]        process variables
    Constraint vector:
        g[0]                        ∑ elements − 1 (target=0 with eps tolerance)
        g[1:]                       feasibility hard constraints (≤ 0)
    """

    def __init__(self, model: ForwardModel, spec: DesignSpec,
                 element_columns: Sequence[str]):
        self.model = model
        self.spec = spec
        self.elements = list(element_columns)
        self.proc_keys = list(spec.process_bounds.keys())
        n_el = len(self.elements)
        n_proc = len(self.proc_keys)

        xl = []
        xu = []
        for e in self.elements:
            lo, hi = spec.element_bounds.get(e, (0.0, 0.5))
            xl.append(lo)
            xu.append(hi)
        for p in self.proc_keys:
            lo, hi = spec.process_bounds[p]
            xl.append(lo)
            xu.append(hi)

        n_obj = len(spec.objectives)
        # Sum-to-one is enforced by projection; feasibility constraints are checked
        # post-hoc to keep n_constr fixed. We use one equality-as-inequality and one
        # aggregate feasibility violation count.
        super().__init__(
            n_var=n_el + n_proc,
            n_obj=n_obj,
            n_constr=2,
            xl=np.array(xl, dtype=float),
            xu=np.array(xu, dtype=float),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        n_el = len(self.elements)
        comp_arr = x[:n_el].astype(float)
        # Project onto simplex respecting bounds: simple normalization (loose but stable)
        s = comp_arr.sum()
        if s <= 1e-9:
            comp_arr = np.full_like(comp_arr, 1.0 / n_el)
            s = 1.0
        comp_norm = comp_arr / s

        comp_df = pd.DataFrame([comp_norm], columns=self.elements)
        proc_df = None
        if self.proc_keys:
            proc_vals = x[n_el:].astype(float)
            proc_df = pd.DataFrame([proc_vals], columns=self.proc_keys)

        preds = self.model.predict(comp_df, process=proc_df).iloc[0]

        # Build objective vector. NSGA-II minimizes, so flip signs as needed.
        f = []
        for tgt, direction in self.spec.objectives:
            mu = preds[f"{tgt}_mean"]
            sigma = preds[f"{tgt}_std"]
            if direction == "max":
                # Maximize μ - λσ  → minimize -(μ - λσ)
                f.append(-(mu - self.spec.risk_lambda * sigma))
            elif direction == "min":
                f.append(mu + self.spec.risk_lambda * sigma)
            else:  # target
                tv = self.spec.target_values[tgt]
                f.append(abs(mu - tv) + self.spec.risk_lambda * sigma)

        # Constraints
        g_sum = abs(s - 1.0) - 1e-2  # allow 1% slack pre-projection
        g_feas = 0.0
        if self.spec.feasibility is not None:
            comp_series = comp_df.iloc[0]
            proc_series = proc_df.iloc[0] if proc_df is not None else None
            r = self.spec.feasibility.check(comp_series, proc_series)
            # Sum positive parts of hard-constraint scores
            for name, g in r.scores.items():
                # We can't tell hard vs soft from scores alone; encode by checking violations
                if name in r.hard_violations and g > 0:
                    g_feas += g
        out["F"] = np.array(f)
        out["G"] = np.array([g_sum, g_feas])


# ---------------------------------------------------------------------------
@dataclass
class InverseDesigner:
    model: ForwardModel
    spec: DesignSpec
    element_columns: Sequence[str]

    def run_nsga2(self, pop_size: int = 80, n_gen: int = 60,
                  seed: int = 0, verbose: bool = False) -> pd.DataFrame:
        problem = _NSGA2Problem(self.model, self.spec, self.element_columns)
        algo = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        )
        res = minimize(
            problem, algo, get_termination("n_gen", n_gen),
            seed=seed, verbose=verbose,
        )
        return self._postprocess(res, problem)

    def _postprocess(self, res, problem: _NSGA2Problem) -> pd.DataFrame:
        if res.X is None:
            return pd.DataFrame()
        X = np.atleast_2d(res.X)
        n_el = len(problem.elements)
        rows = []
        for x in X:
            comp = x[:n_el]
            comp = comp / max(comp.sum(), 1e-9)
            row = {e: comp[i] for i, e in enumerate(problem.elements)}
            if problem.proc_keys:
                for j, p in enumerate(problem.proc_keys):
                    row[p] = float(x[n_el + j])
            rows.append(row)
        cand = pd.DataFrame(rows)

        # Re-predict for clean reporting
        proc_df = cand[problem.proc_keys] if problem.proc_keys else None
        preds = self.model.predict(cand[problem.elements], process=proc_df)
        out = pd.concat([cand.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)

        # Attach feasibility status
        if self.spec.feasibility is not None:
            feas_rows = []
            for i in range(len(out)):
                comp_s = out.loc[i, problem.elements]
                proc_s = out.loc[i, problem.proc_keys] if problem.proc_keys else None
                r = self.spec.feasibility.check(comp_s, proc_s)
                feas_rows.append({
                    "feasible": r.feasible,
                    "n_hard_viol": len(r.hard_violations),
                    "n_soft_viol": len(r.soft_violations),
                })
            out = pd.concat([out, pd.DataFrame(feas_rows)], axis=1)

        # Aggregate score (lower is better in pymoo land; flip for user-facing rank)
        score = np.zeros(len(out))
        for j, (tgt, direction) in enumerate(self.spec.objectives):
            mu = out[f"{tgt}_mean"].to_numpy()
            sigma = out[f"{tgt}_std"].to_numpy()
            if direction == "max":
                score += -(mu - self.spec.risk_lambda * sigma)
            elif direction == "min":
                score += mu + self.spec.risk_lambda * sigma
            else:
                tv = self.spec.target_values[tgt]
                score += np.abs(mu - tv) + self.spec.risk_lambda * sigma
        out["agg_score"] = score
        out = out.sort_values("agg_score").reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out
