"""
Forward composition–property model.

Design:
    Stacked ensemble of XGBoost (capturing non-linear interactions on a small dataset)
    + Gaussian Process residual head (calibrated uncertainty in low-data regions).

    For multi-target problems, fit one model per target (independent) and expose a
    consistent ``predict(X) -> mean, std`` interface that downstream optimization
    layers can rely on.

    Hyperparameters are tuned with Optuna (Bayesian TPE) inside group-aware CV.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBRegressor

from .data_pipeline import Dataset, CompositionFeaturizer, build_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


@dataclass
class _SingleTargetModel:
    """XGBoost mean predictor + GP residual head."""

    xgb: XGBRegressor
    gp: GaussianProcessRegressor
    preproc: object  # sklearn Pipeline from data_pipeline.build_preprocessor()
    feature_names: List[str]
    y_mean: float = 0.0
    y_std: float = 1.0

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_arr = self.preproc.transform(X[self.feature_names])
        mu_xgb = self.xgb.predict(X_arr)
        # GP is trained on residuals of (y - xgb_pred) in standardized space
        resid_mu, resid_std = self.gp.predict(X_arr, return_std=True)
        mu = (mu_xgb + resid_mu) * self.y_std + self.y_mean
        sigma = resid_std * self.y_std
        return mu, sigma


def _objective_xgb(trial: optuna.Trial, X: np.ndarray, y: np.ndarray,
                    cv_splits) -> float:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        max_depth=trial.suggest_int("max_depth", 2, 7),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        random_state=0,
        n_jobs=1,
        verbosity=0,
    )
    maes = []
    for tr, te in cv_splits:
        m = XGBRegressor(**params)
        m.fit(X[tr], y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(X[te])))
    return float(np.mean(maes))


@dataclass
class ForwardModel:
    """Multi-target stacked ensemble with optional Optuna hyperparameter search.

    Usage:
        fm = ForwardModel(featurizer=feat, targets=["UTS_MPa", "elongation_pct"])
        fm.fit(dataset, n_trials=30)
        mean, std = fm.predict(new_compositions_df)
    """

    featurizer: CompositionFeaturizer
    targets: Sequence[str]
    n_cv_splits: int = 5
    random_state: int = 0
    models_: Dict[str, _SingleTargetModel] = field(default_factory=dict)
    metrics_: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def fit(self, dataset: Dataset, n_trials: int = 25,
            extra_feature_cols: Optional[Sequence[str]] = None,
            verbose: bool = False) -> "ForwardModel":
        X_full = dataset.build_X(self.featurizer)
        feature_names = list(X_full.columns)
        if extra_feature_cols:
            for c in extra_feature_cols:
                if c not in feature_names:
                    raise KeyError(f"Extra column not in X: {c}")

        # Build CV splits once (group-aware if available)
        idx = np.arange(len(X_full))
        if dataset.groups is not None and dataset.groups.nunique() >= self.n_cv_splits:
            cv = GroupKFold(n_splits=self.n_cv_splits)
            splits = list(cv.split(idx, groups=dataset.groups.values))
        else:
            cv = KFold(n_splits=self.n_cv_splits, shuffle=True,
                        random_state=self.random_state)
            splits = list(cv.split(idx))

        for tgt in self.targets:
            if tgt not in dataset.properties.columns:
                raise KeyError(f"Target column missing: {tgt}")
            y = dataset.properties[tgt].to_numpy(dtype=float)

            preproc = build_preprocessor()
            X_arr = preproc.fit_transform(X_full[feature_names])

            # Standardize target for stable GP
            y_mean = float(np.nanmean(y))
            y_std = float(np.nanstd(y) or 1.0)
            y_norm = (y - y_mean) / y_std

            # --- Optuna search for XGBoost ---
            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                lambda t: _objective_xgb(t, X_arr, y_norm, splits),
                n_trials=n_trials,
                show_progress_bar=False,
            )
            best_params = study.best_params
            if verbose:
                log.info(f"[{tgt}] best XGB MAE (CV) = {study.best_value:.4f}")

            # --- Train best XGBoost on full data ---
            xgb = XGBRegressor(**best_params, random_state=self.random_state,
                                n_jobs=1, verbosity=0)
            xgb.fit(X_arr, y_norm)

            # --- Train GP on residuals ---
            residuals = y_norm - xgb.predict(X_arr)
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=4,
                    normalize_y=False,
                    random_state=self.random_state,
                )
                gp.fit(X_arr, residuals)

            # --- Estimate CV metrics (refit on each fold for honesty) ---
            cv_preds = np.zeros_like(y_norm)
            for tr, te in splits:
                m_xgb = XGBRegressor(**best_params, random_state=self.random_state,
                                      n_jobs=1, verbosity=0)
                m_xgb.fit(X_arr[tr], y_norm[tr])
                resid_tr = y_norm[tr] - m_xgb.predict(X_arr[tr])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m_gp = GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=2,
                        random_state=self.random_state,
                    )
                    m_gp.fit(X_arr[tr], resid_tr)
                cv_preds[te] = m_xgb.predict(X_arr[te]) + m_gp.predict(X_arr[te])
            cv_preds = cv_preds * y_std + y_mean
            self.metrics_[tgt] = {
                "cv_mae": float(mean_absolute_error(y, cv_preds)),
                "cv_r2": float(r2_score(y, cv_preds)),
                "n_train": int(len(y)),
            }

            self.models_[tgt] = _SingleTargetModel(
                xgb=xgb, gp=gp, preproc=preproc,
                feature_names=feature_names, y_mean=y_mean, y_std=y_std,
            )
        return self

    def predict(
        self, compositions: pd.DataFrame,
        process: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict all targets for the given compositions.
        Returns a DataFrame with columns:  <target>_mean, <target>_std for each target."""
        X = self.featurizer.transform(compositions)
        if process is not None:
            X = pd.concat([X.reset_index(drop=True),
                            process.reset_index(drop=True)], axis=1)

        out = {}
        for tgt, model in self.models_.items():
            mu, sigma = model.predict(X)
            out[f"{tgt}_mean"] = mu
            out[f"{tgt}_std"] = sigma
        return pd.DataFrame(out, index=compositions.index)

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_).T
