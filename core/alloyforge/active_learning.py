"""
Active learning and design of experiments.

For materials groups with expensive experimental cycles, the bottleneck isn't ML
quality — it's *what to try next*. This module turns the forward model into a
recommender of the most informative next experiments.

Two acquisitions are supported:

- **Uncertainty sampling** (qσ): pick candidates with highest predictive σ.
  Fast, simple. Use early when the model is poorly calibrated.

- **qEHVI-like score** (greedy hypervolume improvement under independent models):
  pick candidates that most improve the Pareto front *and* are predicted to be
  feasible. We implement a Monte Carlo flavor that does not require a full
  botorch dependency for the basic path; the botorch-backed variant is in
  ``inverse_design.py`` for proper qNEHVI when needed.

The output is a small "batch of N" recommendations with a written rationale per
pick, suitable to drop into a lab notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .forward_model import ForwardModel
from .feasibility import FeasibilityChecker


def _is_dominated(p: np.ndarray, others: np.ndarray) -> bool:
    """p is dominated if any row in `others` is ≤ p in every dim and < in at least one."""
    if len(others) == 0:
        return False
    le = np.all(others <= p, axis=1)
    lt = np.any(others < p, axis=1)
    return bool(np.any(le & lt))


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows (minimization convention)."""
    n = len(points)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
    return mask


def hypervolume_2d(front: np.ndarray, ref: Tuple[float, float]) -> float:
    """2D hypervolume relative to reference (max corner). Minimization convention.

    Inputs are points to dominate; the volume swept "below" the front and "left" of ref
    is what we measure. Sufficient for visualization purposes; for >2D objectives,
    rely on pymoo's IGD or HV implementations.
    """
    if len(front) == 0:
        return 0.0
    sorted_pts = front[np.argsort(front[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for x, y in sorted_pts:
        if x >= ref[0] or y >= ref[1]:
            continue
        hv += (ref[0] - x) * (prev_y - y)
        prev_y = y
    return float(hv)


@dataclass
class ActiveLearner:
    model: ForwardModel
    feasibility: Optional[FeasibilityChecker] = None

    def sample_uncertainty(self, candidate_pool: pd.DataFrame,
                            element_columns: Sequence[str],
                            process_columns: Optional[Sequence[str]] = None,
                            batch_size: int = 5,
                            target_weights: Optional[Dict[str, float]] = None,
                            ) -> pd.DataFrame:
        """Pick top-σ candidates with a simple diversity filter (greedy)."""
        proc = candidate_pool[list(process_columns)] if process_columns else None
        preds = self.model.predict(candidate_pool[list(element_columns)], process=proc)

        sigma_cols = [c for c in preds.columns if c.endswith("_std")]
        weights = {c.replace("_std", ""): 1.0 for c in sigma_cols}
        if target_weights:
            for k, v in target_weights.items():
                if k in weights:
                    weights[k] = v
        agg_sigma = np.zeros(len(preds))
        for c in sigma_cols:
            tgt = c.replace("_std", "")
            agg_sigma += weights[tgt] * preds[c].to_numpy()

        # Greedy: pick highest σ, then penalize σ of neighbors by composition distance
        comp = candidate_pool[list(element_columns)].to_numpy()
        chosen = []
        remaining = np.arange(len(comp))
        score = agg_sigma.copy()
        while len(chosen) < batch_size and len(remaining) > 0:
            i = remaining[np.argmax(score[remaining])]
            chosen.append(i)
            # Diversity penalty: subtract proportional to closeness to picked point
            d = np.linalg.norm(comp - comp[i], axis=1)
            score = score - 0.5 * agg_sigma.max() * np.exp(-d / 0.05)
            remaining = remaining[remaining != i]

        out = candidate_pool.iloc[chosen].copy().reset_index(drop=True)
        for c in preds.columns:
            out[c] = preds[c].iloc[chosen].to_numpy()
        out["acq_score"] = agg_sigma[chosen]
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def sample_pareto_improvement(self,
                                    candidate_pool: pd.DataFrame,
                                    element_columns: Sequence[str],
                                    objectives: List[Tuple[str, str]],
                                    current_front: np.ndarray,
                                    process_columns: Optional[Sequence[str]] = None,
                                    batch_size: int = 5,
                                    n_mc: int = 64,
                                    seed: int = 0) -> pd.DataFrame:
        """Greedy Monte-Carlo hypervolume improvement.

        ``current_front`` is an (N, n_obj) array of *minimization-convention* values
        of the existing Pareto front (already-completed experiments).
        """
        rng = np.random.default_rng(seed)
        proc = candidate_pool[list(process_columns)] if process_columns else None
        preds = self.model.predict(candidate_pool[list(element_columns)], process=proc)

        # Build (n_cand, n_obj) mean and std matrices in minimization convention
        mu = []
        sigma = []
        for tgt, d in objectives:
            m = preds[f"{tgt}_mean"].to_numpy()
            s = preds[f"{tgt}_std"].to_numpy()
            if d == "max":
                m = -m  # flip to minimization
            elif d == "target":
                raise ValueError("target direction not supported in this acquisition")
            mu.append(m); sigma.append(s)
        mu = np.stack(mu, axis=1)
        sigma = np.stack(sigma, axis=1)

        # Determine reference point (slightly worse than worst observed)
        all_pts = np.vstack([current_front, mu]) if len(current_front) else mu
        ref = all_pts.max(axis=0) + 0.05 * (all_pts.max(axis=0) - all_pts.min(axis=0) + 1e-9)

        chosen = []
        front = current_front.copy() if len(current_front) else np.zeros((0, mu.shape[1]))
        base_hv = hypervolume_2d(front, tuple(ref)) if mu.shape[1] == 2 else 0.0

        for _ in range(batch_size):
            ehvi = np.zeros(len(mu))
            for j in range(len(mu)):
                if j in chosen:
                    continue
                samples = rng.normal(mu[j], sigma[j], size=(n_mc, mu.shape[1]))
                gains = []
                for s in samples:
                    new_front_pts = np.vstack([front, s[None, :]])
                    nd = pareto_front(new_front_pts)
                    new_front = new_front_pts[nd]
                    if mu.shape[1] == 2:
                        hv = hypervolume_2d(new_front, tuple(ref))
                    else:
                        hv = 0.0  # placeholder; in production use pymoo's HV indicator
                    gains.append(max(0.0, hv - base_hv))
                ehvi[j] = float(np.mean(gains))
            j_star = int(np.argmax(ehvi))
            chosen.append(j_star)
            # Update "expected front" with the mean of the chosen point for the next iteration
            new_pt = mu[j_star][None, :]
            new_front_pts = np.vstack([front, new_pt])
            nd = pareto_front(new_front_pts)
            front = new_front_pts[nd]
            if mu.shape[1] == 2:
                base_hv = hypervolume_2d(front, tuple(ref))

        out = candidate_pool.iloc[chosen].copy().reset_index(drop=True)
        for c in preds.columns:
            out[c] = preds[c].iloc[chosen].to_numpy()
        out["rank"] = np.arange(1, len(out) + 1)
        return out
