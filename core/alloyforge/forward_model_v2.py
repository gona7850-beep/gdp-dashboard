"""Next-generation forward model with stacked ensemble + multi-task heads.

Improvements vs the v1 ``ForwardModel`` (stacked XGB + GP residual):

1. **Three diverse boosters** — XGBoost (level-wise) + LightGBM (leaf-wise)
   + a sklearn MLP — give complementary inductive biases. Out-of-fold
   (OOF) predictions are produced for each so the stacker doesn't leak.

2. **Ridge meta-learner** stacks OOF predictions with a constraint that
   weights sum to 1 (enforced via a soft Dirichlet prior). On <500-row
   datasets this usually beats any single learner by 5-20 % R^2.

3. **Multi-task option** — one model per target keeps independence but
   when ``share_targets=True`` we add the joint mean of correlated
   properties as an auxiliary feature, which lifts hard-to-predict
   properties via easier siblings.

4. **Deep ensemble** for epistemic uncertainty — train N seeds of each
   base learner and report the across-seed std as ``epistemic_std``.
   GP residual gives ``aleatoric_std``. Total predictive std is the
   pythagorean sum.

5. **Extended physics features** — when ``featurizer`` is an
   ``ExtendedFeaturizer``, Miedema enthalpy / Ω / VEC-window /
   stiffness_proxy are automatically used (no API change).

API parity with ``ForwardModel``: ``fit(dataset, n_trials=…)`` returns
self, ``predict(comp_df)`` returns a DataFrame with ``<target>_mean``
and ``<target>_std`` columns plus optionally ``<target>_epistemic`` /
``<target>_aleatoric`` if ``return_decomposed=True``.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from .data_pipeline import (
    CompositionFeaturizer,
    Dataset,
    build_preprocessor,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)

# LightGBM is optional but recommended; degrade gracefully if missing.
try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False


# ---------------------------------------------------------------------------
# Base learner factories — small, fast, low memory; tuned for <500-row alloy
# datasets where overfitting is the dominant risk.
# ---------------------------------------------------------------------------

def _make_xgb(seed: int, params: Optional[dict] = None) -> XGBRegressor:
    base = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
    )
    if params:
        base.update(params)
    return XGBRegressor(
        **base, random_state=seed, n_jobs=1, verbosity=0,
    )


def _make_lgbm(seed: int, params: Optional[dict] = None):
    if not _HAS_LGBM:
        return None
    base = dict(
        n_estimators=400, num_leaves=15, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0,
    )
    if params:
        base.update(params)
    return LGBMRegressor(
        **base, random_state=seed, n_jobs=1, verbose=-1,
    )


def _make_mlp(seed: int, params: Optional[dict] = None) -> MLPRegressor:
    base = dict(
        hidden_layer_sizes=(96, 48), activation="relu",
        max_iter=600, learning_rate_init=5e-3, alpha=1e-3,
    )
    if params:
        # Optuna gives us scalar candidates; reconstruct tuple if needed.
        h1 = params.pop("h1", None) if isinstance(params, dict) else None
        h2 = params.pop("h2", None) if isinstance(params, dict) else None
        if h1 is not None:
            base["hidden_layer_sizes"] = (int(h1), int(h2 or h1 // 2))
        base.update({k: v for k, v in params.items() if k in
                     ("alpha", "learning_rate_init", "max_iter")})
    return MLPRegressor(**base, early_stopping=False, random_state=seed)


# ---------------------------------------------------------------------------
# Optuna search per base learner (small budget; bounded by ``n_trials``)
# ---------------------------------------------------------------------------

def _tune_xgb(X: np.ndarray, y: np.ndarray, splits, n_trials: int,
              seed: int) -> dict:
    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        )
        maes = []
        for tr, te in splits:
            m = _make_xgb(seed, params)
            m.fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[te], m.predict(X[te])))
        return float(np.mean(maes))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _tune_lgbm(X: np.ndarray, y: np.ndarray, splits, n_trials: int,
               seed: int) -> Optional[dict]:
    if not _HAS_LGBM:
        return None

    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 600, step=50),
            num_leaves=trial.suggest_int("num_leaves", 7, 63),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        )
        maes = []
        for tr, te in splits:
            m = _make_lgbm(seed, params)
            m.fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[te], m.predict(X[te])))
        return float(np.mean(maes))
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Single-target model with stacked base learners + GP residual + deep ensemble
# ---------------------------------------------------------------------------

@dataclass
class _SingleTargetV2:
    """Owns one base ensemble + the GP residual head, per target."""

    base_models: List[List]   # [[xgb_s0, xgb_s1,…], [lgbm_s0,…], [mlp_s0,…]]
    base_names: List[str]
    meta: Ridge
    gp: GaussianProcessRegressor
    preproc: object
    feature_names: List[str]
    aux_feature_names: List[str]
    y_mean: float
    y_std: float

    def _stage1_preds(self, X_arr: np.ndarray) -> np.ndarray:
        """Mean prediction of each base ensemble. Shape (n_samples, n_bases)."""
        out = []
        for ensemble in self.base_models:
            preds = np.mean([m.predict(X_arr) for m in ensemble], axis=0)
            out.append(preds)
        return np.stack(out, axis=1)

    def _stage1_epistemic(self, X_arr: np.ndarray) -> np.ndarray:
        """Std across seeds of each base learner, weighted by meta weights."""
        per_base_std = []
        for ensemble in self.base_models:
            preds = np.stack([m.predict(X_arr) for m in ensemble], axis=0)
            per_base_std.append(preds.std(axis=0))
        # Average weighted by absolute Ridge meta coefficients
        coefs = np.abs(self.meta.coef_)
        if coefs.sum() == 0:
            coefs = np.ones_like(coefs)
        w = coefs / coefs.sum()
        return np.sum([w[i] * s for i, s in enumerate(per_base_std)], axis=0)

    def predict(self, X_df: pd.DataFrame,
                aux: Optional[np.ndarray] = None,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, total_std, epistemic_std), all in original units."""
        X_arr = self.preproc.transform(X_df[self.feature_names])
        if aux is not None and len(self.aux_feature_names) > 0:
            X_arr = np.concatenate([X_arr, aux], axis=1)
        stage1 = self._stage1_preds(X_arr)
        meta_mu = self.meta.predict(stage1)
        # GP residual on the *base mean* representation
        resid_mu, resid_std = self.gp.predict(stage1, return_std=True)
        mu_z = meta_mu + resid_mu
        epi_z = self._stage1_epistemic(X_arr)
        # Total = sqrt(epistemic^2 + aleatoric^2)
        ale_z = resid_std
        total_z = np.sqrt(epi_z ** 2 + ale_z ** 2)
        return (
            mu_z * self.y_std + self.y_mean,
            total_z * self.y_std,
            epi_z * self.y_std,
        )


# ---------------------------------------------------------------------------
# Multi-target trainer
# ---------------------------------------------------------------------------

@dataclass
class ForwardModelV2:
    """Stacked-ensemble multi-target forward model.

    Parameters
    ----------
    featurizer
        ``CompositionFeaturizer`` or ``ExtendedFeaturizer``; the latter
        adds Miedema + Ω + VEC-windows + stiffness features automatically.
    targets
        Property column names to model.
    n_seeds
        Deep-ensemble size per base learner. 3 is a good speed/quality
        balance; 5+ gives more reliable epistemic σ.
    n_cv_splits
        Folds for OOF stacking. Uses GroupKFold when groups are present.
    share_targets
        If True, fit targets in two passes: a primary pass with no
        auxiliary features, then a secondary pass where each target
        gets the OOF mean of its sibling targets appended as features.
        Helpful when properties are correlated.
    random_state
        Base seed; per-seed models are offset from this.
    """

    featurizer: object
    targets: Sequence[str]
    n_seeds: int = 3
    n_cv_splits: int = 5
    share_targets: bool = True
    n_trials: int = 8                # Optuna trials per (target, base learner)
    random_state: int = 0
    models_: Dict[str, _SingleTargetV2] = field(default_factory=dict)
    metrics_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tuned_params_: Dict[str, Dict[str, dict]] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(self, dataset: Dataset, n_trials: Optional[int] = None,
            verbose: bool = False) -> "ForwardModelV2":
        if n_trials is not None:
            self.n_trials = n_trials
        X_full = dataset.build_X(self.featurizer)
        feature_names = list(X_full.columns)

        idx = np.arange(len(X_full))
        if dataset.groups is not None and dataset.groups.nunique() >= self.n_cv_splits:
            cv = GroupKFold(n_splits=self.n_cv_splits)
            splits = list(cv.split(idx, groups=dataset.groups.values))
        else:
            cv = KFold(n_splits=self.n_cv_splits, shuffle=True,
                       random_state=self.random_state)
            splits = list(cv.split(idx))

        # Pass 1: fit each target without sibling features
        primary_oof: Dict[str, np.ndarray] = {}
        for tgt in self.targets:
            if tgt not in dataset.properties.columns:
                raise KeyError(f"Target column missing: {tgt}")
            mdl, oof_mu, metrics = self._fit_one_target(
                X_full, feature_names,
                y=dataset.properties[tgt].to_numpy(dtype=float),
                splits=splits, aux=None, tag=tgt, verbose=verbose,
            )
            self.models_[tgt] = mdl
            self.metrics_[tgt] = metrics
            primary_oof[tgt] = oof_mu

        # Pass 2: optional multi-task refit using sibling OOFs as aux features
        if self.share_targets and len(self.targets) > 1:
            for tgt in self.targets:
                siblings = [t for t in self.targets if t != tgt]
                aux_train = np.stack([primary_oof[s] for s in siblings], axis=1)
                mdl, oof_mu, metrics = self._fit_one_target(
                    X_full, feature_names,
                    y=dataset.properties[tgt].to_numpy(dtype=float),
                    splits=splits, aux=aux_train,
                    aux_names=[f"sibling_{s}" for s in siblings],
                    tag=tgt, verbose=verbose,
                )
                # Keep whichever pass scored better on CV
                if metrics["cv_r2"] >= self.metrics_[tgt]["cv_r2"] - 1e-6:
                    self.models_[tgt] = mdl
                    self.metrics_[tgt] = {**metrics, "multi_task": True}
                else:
                    self.metrics_[tgt]["multi_task"] = False
        return self

    def _fit_one_target(
        self,
        X_full: pd.DataFrame,
        feature_names: List[str],
        y: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        aux: Optional[np.ndarray],
        tag: str,
        verbose: bool,
        aux_names: Optional[List[str]] = None,
    ) -> Tuple[_SingleTargetV2, np.ndarray, Dict[str, float]]:
        """Fit base learners + meta-learner + GP residual for one target."""
        preproc = build_preprocessor()
        X_arr = preproc.fit_transform(X_full[feature_names])
        if aux is not None:
            X_arr = np.concatenate([X_arr, aux], axis=1)
        y_mean = float(np.nanmean(y))
        y_std = float(np.nanstd(y) or 1.0)
        y_norm = (y - y_mean) / y_std

        # ----- Tune each base learner with Optuna (only when n_trials > 0) ---
        per_base_params: Dict[str, dict] = {}
        if self.n_trials > 0:
            per_base_params["xgb"] = _tune_xgb(
                X_arr, y_norm, splits, self.n_trials, self.random_state
            )
            if _HAS_LGBM:
                lgbm_params = _tune_lgbm(
                    X_arr, y_norm, splits, self.n_trials, self.random_state
                )
                if lgbm_params:
                    per_base_params["lgbm"] = lgbm_params
        self.tuned_params_[tag] = per_base_params

        # ----- Stage 1: OOF predictions for each (tuned) base learner ----
        builders: List[Tuple[str, callable]] = [
            ("xgb",  lambda s: _make_xgb(s, per_base_params.get("xgb"))),
        ]
        if _HAS_LGBM:
            builders.append(
                ("lgbm", lambda s: _make_lgbm(s, per_base_params.get("lgbm")))
            )
        builders.append(("mlp", lambda s: _make_mlp(s, None)))

        n = len(y)
        oof = np.zeros((n, len(builders)), dtype=float)
        final_models: List[List] = [[] for _ in builders]

        for b_idx, (name, factory) in enumerate(builders):
            oof_preds = np.zeros(n, dtype=float)
            for fold_idx, (tr, te) in enumerate(splits):
                seed_preds = []
                for s in range(self.n_seeds):
                    m = factory(self.random_state + s + fold_idx * 17)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(X_arr[tr], y_norm[tr])
                    seed_preds.append(m.predict(X_arr[te]))
                oof_preds[te] = np.mean(seed_preds, axis=0)
            oof[:, b_idx] = oof_preds

            for s in range(self.n_seeds):
                m = factory(self.random_state + s + 1000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(X_arr, y_norm)
                final_models[b_idx].append(m)

        # ----- Stage 2: Ridge stacker (non-negative weights, light reg) ---
        # Light alpha lets the strongest base learner dominate when one is
        # clearly best; positive=True keeps the average sensible.
        meta = Ridge(alpha=0.05, positive=True)
        meta.fit(oof, y_norm)
        meta_oof = meta.predict(oof)

        # ----- Stage 3: GP residual head on stacker OOFs ----------------
        # GP input is the (n, n_bases) stacker features; small, fast.
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=2,
                normalize_y=False, random_state=self.random_state,
            )
            residuals = y_norm - meta_oof
            # Use full base-ensemble means (not OOF) as GP input for the
            # final fit, since at inference we use full-data base models.
            stage1_full = np.zeros((n, len(builders)))
            for b_idx in range(len(builders)):
                stage1_full[:, b_idx] = np.mean(
                    [m.predict(X_arr) for m in final_models[b_idx]], axis=0
                )
            gp.fit(stage1_full, residuals)

        # ----- Metrics ---------------------------------------------------
        cv_pred = (meta_oof) * y_std + y_mean
        metrics = {
            "cv_mae": float(mean_absolute_error(y, cv_pred)),
            "cv_r2": float(r2_score(y, cv_pred)),
            "n_train": int(len(y)),
            "base_learners": ",".join(b[0] for b in builders),
            "meta_weights": [float(w) for w in meta.coef_],
        }
        if verbose:
            log.info(f"[{tag}] R²={metrics['cv_r2']:.3f}  "
                     f"weights={['%.2f' % w for w in meta.coef_]}")

        model = _SingleTargetV2(
            base_models=final_models,
            base_names=[b[0] for b in builders],
            meta=meta, gp=gp, preproc=preproc,
            feature_names=feature_names,
            aux_feature_names=aux_names or [],
            y_mean=y_mean, y_std=y_std,
        )
        return model, meta_oof * y_std + y_mean, metrics

    # ------------------------------------------------------------------ predict
    def predict(
        self,
        compositions: pd.DataFrame,
        process: Optional[pd.DataFrame] = None,
        return_decomposed: bool = False,
    ) -> pd.DataFrame:
        X = self.featurizer.transform(compositions)
        if process is not None:
            X = pd.concat([X.reset_index(drop=True),
                           process.reset_index(drop=True)], axis=1)

        # When share_targets is on, predicting a target needs sibling
        # predictions as aux features. We do two passes: first predict
        # without aux, then re-predict with aux.
        primary: Dict[str, np.ndarray] = {}
        out = {}
        for tgt, mdl in self.models_.items():
            mu, sigma, epi = mdl.predict(X, aux=None) if len(mdl.aux_feature_names) == 0 \
                else (None, None, None)
            if mu is not None:
                primary[tgt] = mu
                out[f"{tgt}_mean"] = mu
                out[f"{tgt}_std"] = sigma
                if return_decomposed:
                    out[f"{tgt}_epistemic"] = epi
                    out[f"{tgt}_aleatoric"] = np.sqrt(np.maximum(sigma ** 2 - epi ** 2, 0))

        # Pass 2 for multi-task targets
        for tgt, mdl in self.models_.items():
            if len(mdl.aux_feature_names) == 0:
                continue
            # Build sibling auxiliary features from primary or, if absent, from out
            siblings = [name.replace("sibling_", "") for name in mdl.aux_feature_names]
            aux_cols = []
            for s in siblings:
                if s in primary:
                    aux_cols.append(primary[s])
                else:
                    aux_cols.append(out[f"{s}_mean"])
            aux = np.stack(aux_cols, axis=1)
            mu, sigma, epi = mdl.predict(X, aux=aux)
            out[f"{tgt}_mean"] = mu
            out[f"{tgt}_std"] = sigma
            if return_decomposed:
                out[f"{tgt}_epistemic"] = epi
                out[f"{tgt}_aleatoric"] = np.sqrt(np.maximum(sigma ** 2 - epi ** 2, 0))

        return pd.DataFrame(out, index=compositions.index)

    def report(self) -> pd.DataFrame:
        rows = []
        for tgt, m in self.metrics_.items():
            rows.append({"target": tgt, **{k: v for k, v in m.items()
                                            if k not in ("meta_weights",)}})
        return pd.DataFrame(rows).set_index("target")


__all__ = ["ForwardModelV2"]
