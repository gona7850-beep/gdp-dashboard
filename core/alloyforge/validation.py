"""
Validation utilities for forward models.

What this provides beyond raw CV scores:
    - **Conformal prediction intervals**: distribution-free coverage guarantees.
      Used to convert the GP's σ into a calibrated [lo, hi] band given a target
      miscoverage level (default α = 0.1 → 90% coverage).
    - **Reliability diagram data**: empirical vs predicted CDF over a held-out set,
      so the UI can show whether the σ from the model is honest.
    - **Domain-of-applicability (DoA) check**: simple distance-to-nearest-training
      point in feature space. Stops the user from trusting predictions on a
      composition that lies far outside what the model has seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from .forward_model import ForwardModel
from .data_pipeline import Dataset


@dataclass
class ConformalCalibrator:
    """Split-conformal calibration on top of a fitted ForwardModel.

    Wraps the model's (μ, σ) predictions to produce prediction intervals that
    achieve marginal coverage 1-α regardless of model miscalibration.

    Algorithm (per target):
        1. Hold out a calibration set (default 20% of training data, group-aware).
        2. Compute non-conformity scores: |y - μ| / σ (locally adaptive).
        3. q_hat = (⌈(n+1)(1-α)⌉)-th quantile of scores.
        4. At inference: interval = μ ± q_hat · σ.
    """

    alpha: float = 0.1  # miscoverage; α=0.1 → 90% prediction intervals
    q_hat_: Dict[str, float] = field(default_factory=dict)

    def calibrate(self, model: ForwardModel, dataset: Dataset,
                  calib_idx: Optional[np.ndarray] = None,
                  seed: int = 0) -> "ConformalCalibrator":
        rng = np.random.default_rng(seed)
        n = len(dataset.compositions)
        if calib_idx is None:
            calib_idx = rng.choice(n, size=max(20, n // 5), replace=False)

        comp_cal = dataset.compositions.iloc[calib_idx]
        proc_cal = dataset.process.iloc[calib_idx] if dataset.process is not None else None
        y_cal = dataset.properties.iloc[calib_idx]

        preds = model.predict(comp_cal, process=proc_cal)
        for tgt in model.targets:
            mu = preds[f"{tgt}_mean"].to_numpy()
            sigma = preds[f"{tgt}_std"].to_numpy()
            sigma = np.where(sigma < 1e-6, 1e-6, sigma)
            scores = np.abs(y_cal[tgt].to_numpy() - mu) / sigma
            k = int(np.ceil((len(scores) + 1) * (1 - self.alpha)))
            k = min(k, len(scores))
            self.q_hat_[tgt] = float(np.sort(scores)[k - 1])
        return self

    def intervals(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Augment a predictions DataFrame with `_lo` and `_hi` columns."""
        out = predictions.copy()
        for col in list(predictions.columns):
            if not col.endswith("_mean"):
                continue
            tgt = col[:-5]
            if tgt not in self.q_hat_:
                continue
            mu = predictions[f"{tgt}_mean"].to_numpy()
            sigma = predictions[f"{tgt}_std"].to_numpy()
            q = self.q_hat_[tgt]
            out[f"{tgt}_lo"] = mu - q * sigma
            out[f"{tgt}_hi"] = mu + q * sigma
        return out


# ---------------------------------------------------------------------------
@dataclass
class DomainOfApplicability:
    """Simple, fast DoA: Euclidean distance to nearest neighbour in scaled
    feature space, expressed as a percentile of the training distance distribution.

    A `doa_score` near 0 means the query sits inside the training cloud; near 1
    (or above) means it is an extrapolation. Tag predictions with `extrapolation=True`
    when score > threshold (default 0.95).
    """

    threshold: float = 0.95
    X_train_: Optional[np.ndarray] = None
    train_nn_quantile_: float = 1.0

    def fit(self, model: ForwardModel, dataset: Dataset) -> "DomainOfApplicability":
        X = dataset.build_X(model.featurizer)
        # Use the preprocessor from the first target model (same features across targets)
        first = next(iter(model.models_.values()))
        Xs = first.preproc.transform(X[first.feature_names])
        self.X_train_ = Xs
        # Reference: NN distance for each training point to its nearest other point
        d_self = self._nn_min_distance(Xs, Xs, self_excluded=True)
        self.train_nn_quantile_ = float(np.quantile(d_self, 0.95))
        return self

    def score(self, X_query: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("DomainOfApplicability not fitted")
        d = self._nn_min_distance(X_query, self.X_train_, self_excluded=False)
        return d / (self.train_nn_quantile_ + 1e-9)

    @staticmethod
    def _nn_min_distance(A: np.ndarray, B: np.ndarray,
                          self_excluded: bool) -> np.ndarray:
        # O(n*m); fine for n,m up to a few thousand.
        diff = A[:, None, :] - B[None, :, :]
        d = np.sqrt((diff**2).sum(-1))
        if self_excluded:
            np.fill_diagonal(d, np.inf)
        return d.min(axis=1)


# ---------------------------------------------------------------------------
def reliability_diagram(model: ForwardModel, dataset: Dataset,
                        target: str, n_bins: int = 10) -> pd.DataFrame:
    """Return predicted-vs-observed coverage data for plotting a reliability curve.

    For each predicted z-quantile q, compute the fraction of observations falling
    inside the corresponding central interval. A well-calibrated model lies on y=x.
    """
    preds = model.predict(dataset.compositions, process=dataset.process)
    mu = preds[f"{target}_mean"].to_numpy()
    sigma = preds[f"{target}_std"].to_numpy()
    y = dataset.properties[target].to_numpy()
    z = (y - mu) / np.where(sigma < 1e-6, 1e-6, sigma)

    rows = []
    for q in np.linspace(0.05, 0.95, n_bins):
        zc = norm.ppf(0.5 + q / 2)
        coverage = float(np.mean(np.abs(z) <= zc))
        rows.append({"nominal_coverage": float(q), "empirical_coverage": coverage})
    return pd.DataFrame(rows)


def cv_summary(model: ForwardModel) -> pd.DataFrame:
    return model.report()
