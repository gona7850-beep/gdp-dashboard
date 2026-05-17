"""Unified active-learning planner.

Combines three signals to recommend the next batch of experiments:

* **Predictive uncertainty (σ)** — high σ ⇒ model knows little ⇒
  experiment teaches the most.
* **HTS compound score** — composition whose element set matches a
  high-scoring compound is metallurgically promising.
* **Domain-of-applicability (DoA)** — distance to nearest training point
  in feature space. Picks that lie just outside the training cloud are
  best for active learning; far-out picks are extrapolations.

The acquisition function is::

    acq = w_sigma · σ_norm   +  w_hts · hts_norm  +  w_doa · doa_band

where each term is normalised to [0, 1] within the candidate pool, and
``doa_band`` is a triangular kernel that peaks at the 0.8-1.2 DoA
percentile (just outside the training cloud) and decays to zero at 0
(redundant) and at 3+ (extrapolation).

This sits on top of the existing :class:`ActiveLearner` in
``core/alloyforge/active_learning.py`` — that one handles the pure
uncertainty + Pareto-improvement axis. Here we add HTS and DoA so the
recommendation reflects the full design pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_pipeline import Dataset
from .hts_descriptor import HTSScoreFeaturizer
from .validation import DomainOfApplicability


@dataclass
class PlannerWeights:
    sigma: float = 1.0
    hts: float = 0.5
    doa: float = 0.5


@dataclass
class ExperimentPlanner:
    """Pick the next ``batch_size`` experiments to run.

    Parameters
    ----------
    model
        Trained forward model (v1 or v2). Must expose
        ``predict(comp, process=None)`` returning a DataFrame with
        ``<target>_mean`` and ``<target>_std`` columns.
    host_symbol
        Host matrix for the HTS scorer (default ``"Nb"``).
    weights
        Acquisition weights.
    """

    model: object
    host_symbol: str = "Nb"
    weights: PlannerWeights = field(default_factory=PlannerWeights)
    doa_: Optional[DomainOfApplicability] = None
    hts_: Optional[HTSScoreFeaturizer] = None
    last_acquisition_: Optional[pd.DataFrame] = None

    def fit(self, training_dataset: Dataset) -> "ExperimentPlanner":
        """Fit the DoA on the training set; build the HTS featurizer."""
        self.doa_ = DomainOfApplicability().fit(self.model, training_dataset)
        self.hts_ = HTSScoreFeaturizer(host_symbol=self.host_symbol)
        return self

    def propose(
        self,
        candidate_pool: pd.DataFrame,
        targets: Sequence[str],
        process_pool: Optional[pd.DataFrame] = None,
        batch_size: int = 5,
        diversity_decay: float = 0.5,
    ) -> pd.DataFrame:
        """Score each candidate, then greedily pick a diverse batch.

        Parameters
        ----------
        candidate_pool
            DataFrame of compositions to choose from (one per row).
        targets
            Property names to compute σ on (averaged across targets).
        process_pool
            Optional process variables for each candidate.
        batch_size
            Number of picks to return.
        diversity_decay
            After each pick we down-weight nearby candidates by a
            Gaussian of the composition distance. ``0`` = pure greedy on
            acquisition, ``1`` = aggressive diversity.

        Returns
        -------
        pandas.DataFrame
            Picks ranked 1-N with columns: every candidate column plus
            ``sigma_avg``, ``hts_score``, ``doa_score``, ``acq``, ``rank``.
        """
        el_cols = list(candidate_pool.columns)

        # ----- σ averaged across targets -----
        preds = self.model.predict(candidate_pool, process=process_pool)
        sigma_cols = [f"{t}_std" for t in targets if f"{t}_std" in preds.columns]
        if not sigma_cols:
            raise ValueError(
                f"Model predictions lack any of {[t + '_std' for t in targets]}"
            )
        sigma_avg = preds[sigma_cols].mean(axis=1).to_numpy()

        # ----- HTS best-total per row -----
        if self.hts_ is None:
            hts_score = np.zeros(len(candidate_pool))
        else:
            hts_score = self.hts_.transform(candidate_pool)["hts_max_total"].to_numpy()

        # ----- DoA score (in-domain → 0; just outside training → 1; far → drops) -----
        if self.doa_ is not None:
            X_feat = self._featurize_for_doa(candidate_pool)
            doa_raw = self.doa_.score(X_feat)
        else:
            doa_raw = np.ones(len(candidate_pool))
        # Triangular kernel peaking at 1.0
        doa_band = np.clip(
            1.0 - np.abs(doa_raw - 1.0) / 2.0, 0.0, 1.0,
        )

        # ----- Normalise each signal within the candidate pool -----
        sigma_n = _minmax_normalize(sigma_avg)
        hts_n = _minmax_normalize(hts_score)
        doa_n = _minmax_normalize(doa_band)

        acq_base = (
            self.weights.sigma * sigma_n
            + self.weights.hts * hts_n
            + self.weights.doa * doa_n
        )

        # ----- Greedy diversity-aware selection -----
        comp_arr = candidate_pool[el_cols].to_numpy(dtype=float)
        chosen: List[int] = []
        acq_dyn = acq_base.copy()
        remaining = np.arange(len(comp_arr))
        while len(chosen) < min(batch_size, len(comp_arr)):
            j = int(remaining[np.argmax(acq_dyn[remaining])])
            chosen.append(j)
            # diversity penalty: subtract proportional to closeness in
            # composition space
            d = np.linalg.norm(comp_arr - comp_arr[j], axis=1)
            penalty = diversity_decay * acq_base.max() * np.exp(-d / 0.05)
            acq_dyn = acq_dyn - penalty
            remaining = remaining[remaining != j]

        out = candidate_pool.iloc[chosen].copy().reset_index(drop=True)
        out["sigma_avg"] = sigma_avg[chosen]
        out["hts_score"] = hts_score[chosen]
        out["doa_score"] = doa_raw[chosen]
        out["acq"] = acq_base[chosen]
        out["rank"] = np.arange(1, len(out) + 1)
        # Attach predicted means for each target for convenience
        for t in targets:
            mean_col = f"{t}_mean"
            if mean_col in preds.columns:
                out[mean_col] = preds.iloc[chosen][mean_col].to_numpy()
        self.last_acquisition_ = out
        return out

    def _featurize_for_doa(self, comp_df: pd.DataFrame) -> np.ndarray:
        """Transform compositions through the same featurizer + preproc as the
        underlying model."""
        feat = self.model.featurizer
        X = feat.transform(comp_df)
        first = next(iter(self.model.models_.values()))
        return first.preproc.transform(X[first.feature_names])


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


__all__ = ["ExperimentPlanner", "PlannerWeights"]
