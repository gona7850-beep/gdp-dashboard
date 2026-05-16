"""Composition design / verification / inverse design / property prediction.

An extended, framework-independent reimplementation of the reference
``composition_platform.py`` sketch. Compared to that sketch, this module:

* Supports several scikit-learn estimators (Random Forest, Gradient Boosting,
  Ridge, MLP) selected by name, plus a custom estimator hook.
* Reports per-property validation R^2 *and* k-fold mean / std so reviewers
  can judge stability rather than a single train/val split.
* Adds **element constraints** (per-element min/max and fixed-value pins)
  enforced during inverse design sampling, plus an optional weighted MSE so
  some target properties can be prioritised.
* Adds a **genetic-algorithm** inverse-design strategy in addition to the
  Dirichlet Monte Carlo baseline, and lets the caller pick.
* Adds **uncertainty estimation** for Random-Forest predictions via the
  std-dev across tree predictions (one std per property), so candidates
  with low predicted error but high model uncertainty can be down-ranked.
* Persists trained models with joblib (full estimator state), not just a
  JSON of column names.

Nothing in this file is framework-specific: the Streamlit page in
``app/pages/7_조성설계_플랫폼.py`` and the FastAPI router in
``backend/routers/composition.py`` both consume this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Predicted properties for a single composition."""
    properties: dict[str, float]
    uncertainty: dict[str, float] | None = None
    model_r2: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingReport:
    """Summary returned by :meth:`PropertyPredictor.train`."""
    feature_columns: list[str]
    property_columns: list[str]
    n_samples: int
    n_train: int
    n_val: int
    val_r2: dict[str, float]
    val_mae: dict[str, float]
    cv_r2_mean: dict[str, float]
    cv_r2_std: dict[str, float]
    estimator_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DesignConstraints:
    """Per-element bounds enforced during inverse design.

    ``min_fraction`` / ``max_fraction`` clip every sampled composition into
    the requested range and then renormalise so the row still sums to 1.
    ``fixed`` pins specific elements to a constant fraction (the remaining
    budget is distributed among the others by the chosen sampler).
    """
    min_fraction: dict[str, float] = field(default_factory=dict)
    max_fraction: dict[str, float] = field(default_factory=dict)
    fixed: dict[str, float] = field(default_factory=dict)

    def validate(self, feature_columns: list[str]) -> None:
        for name, table in [("min", self.min_fraction),
                            ("max", self.max_fraction),
                            ("fixed", self.fixed)]:
            unknown = [k for k in table if k not in feature_columns]
            if unknown:
                raise ValueError(f"Unknown elements in {name} constraints: {unknown}")
        fixed_sum = sum(self.fixed.values())
        if fixed_sum > 1.0 + 1e-9:
            raise ValueError(f"Fixed fractions sum to {fixed_sum:.3f} > 1.0")
        for el, lo in self.min_fraction.items():
            hi = self.max_fraction.get(el, 1.0)
            if lo > hi:
                raise ValueError(f"min_fraction[{el}]={lo} > max_fraction[{el}]={hi}")


@dataclass
class DesignCandidate:
    """A single inverse-design candidate."""
    composition: dict[str, float]
    predicted: dict[str, float]
    uncertainty: dict[str, float] | None
    score: float                     # weighted MSE (lower is better)
    rel_errors: dict[str, float]     # |pred - target| / |target|

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Estimator zoo
# ---------------------------------------------------------------------------

def _build_estimator(name: str, random_state: int = 42):
    name = name.lower()
    if name in ("rf", "randomforest", "random_forest"):
        return RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    if name in ("gbr", "gradientboosting", "gradient_boosting"):
        return MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=300, random_state=random_state)
        )
    if name in ("ridge",):
        return Ridge(alpha=1.0, random_state=random_state)
    if name in ("mlp", "neural", "nn"):
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            random_state=random_state,
        )
    raise ValueError(
        f"Unknown estimator '{name}'. Choose from: rf, gbr, ridge, mlp."
    )


AVAILABLE_ESTIMATORS = ("rf", "gbr", "ridge", "mlp")


# ---------------------------------------------------------------------------
# Property predictor
# ---------------------------------------------------------------------------

class PropertyPredictor:
    """Train an sklearn pipeline and predict per-property values.

    Designed to be reusable across UIs (Streamlit / FastAPI / CLI). After
    ``train()`` it exposes :attr:`feature_columns`, :attr:`property_columns`,
    and the validation report. The pipeline is always ``StandardScaler ->
    estimator`` so callers can pass raw element fractions without worrying
    about scaling.
    """

    def __init__(
        self,
        estimator: str | object = "rf",
        random_state: int = 42,
    ) -> None:
        if isinstance(estimator, str):
            self.estimator_name = estimator.lower()
            self._estimator = _build_estimator(estimator, random_state=random_state)
        else:
            self.estimator_name = type(estimator).__name__
            self._estimator = estimator
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []
        self.property_columns: list[str] = []
        self.report: TrainingReport | None = None

    # --- training ----------------------------------------------------------

    def train(
        self,
        data: pd.DataFrame,
        feature_columns: list[str] | None = None,
        property_columns: list[str] | None = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> TrainingReport:
        """Fit on ``data`` and return a :class:`TrainingReport`.

        ``feature_columns`` / ``property_columns`` may be omitted; in that
        case any numeric column whose row-wise sum across columns is close
        to 1.0 is treated as a feature, the rest as properties. This makes
        the function "just work" on the common alloy-CSV layout.
        """
        feat, prop = self._infer_columns(data, feature_columns, property_columns)
        self.feature_columns, self.property_columns = feat, prop

        X = data[feat].to_numpy(dtype=float)
        y = data[prop].to_numpy(dtype=float)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("estimator", self._estimator),
        ])
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_val)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
            y_val_2d = y_val[:, None] if y_val.ndim == 1 else y_val
        else:
            y_val_2d = y_val if y_val.ndim > 1 else y_val[:, None]

        val_r2 = {p: float(r2_score(y_val_2d[:, i], y_pred[:, i]))
                  for i, p in enumerate(prop)}
        val_mae = {p: float(mean_absolute_error(y_val_2d[:, i], y_pred[:, i]))
                   for i, p in enumerate(prop)}

        cv_mean, cv_std = self._cross_validate(X, y, cv_folds)

        self.report = TrainingReport(
            feature_columns=feat,
            property_columns=prop,
            n_samples=int(len(data)),
            n_train=int(len(X_train)),
            n_val=int(len(X_val)),
            val_r2=val_r2,
            val_mae=val_mae,
            cv_r2_mean=cv_mean,
            cv_r2_std=cv_std,
            estimator_name=self.estimator_name,
        )
        return self.report

    def _infer_columns(
        self,
        data: pd.DataFrame,
        feature_columns: list[str] | None,
        property_columns: list[str] | None,
    ) -> tuple[list[str], list[str]]:
        if feature_columns and property_columns:
            return list(feature_columns), list(property_columns)
        numeric = data.select_dtypes(include=[float, int]).columns.tolist()
        if feature_columns:
            feats = list(feature_columns)
            props = property_columns or [c for c in numeric if c not in feats]
            return feats, props
        if property_columns:
            props = list(property_columns)
            feats = [c for c in numeric if c not in props]
            return feats, props
        # auto-detect: features are columns whose row-sum is close to 1.0
        feats: list[str] = []
        for col in numeric:
            vals = data[col].to_numpy(dtype=float)
            if (vals >= 0).all() and vals.max() <= 1.0 + 1e-6:
                feats.append(col)
        sums = data[feats].sum(axis=1) if feats else pd.Series([], dtype=float)
        if not feats or not np.allclose(sums, 1.0, atol=1e-2):
            mid = len(numeric) // 2
            feats = numeric[:mid]
            props = numeric[mid:]
        else:
            props = [c for c in numeric if c not in feats]
        if not feats or not props:
            raise ValueError(
                "Could not infer features/properties. Pass feature_columns "
                "and property_columns explicitly."
            )
        return feats, props

    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_splits: int
    ) -> tuple[dict[str, float], dict[str, float]]:
        if n_splits < 2 or len(X) < n_splits + 1:
            zeros = {p: float("nan") for p in self.property_columns}
            return zeros, zeros
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        per_prop_scores: list[list[float]] = [[] for _ in self.property_columns]
        for train_idx, val_idx in kf.split(X):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("estimator", _build_estimator(self.estimator_name, self.random_state)
                 if isinstance(self.estimator_name, str)
                 and self.estimator_name in AVAILABLE_ESTIMATORS
                 else self._estimator),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            preds = pipe.predict(X[val_idx])
            if preds.ndim == 1:
                preds = preds[:, None]
            y_val = y[val_idx] if y.ndim > 1 else y[val_idx][:, None]
            for i in range(len(self.property_columns)):
                per_prop_scores[i].append(float(r2_score(y_val[:, i], preds[:, i])))
        mean = {p: float(np.mean(s)) for p, s in zip(self.property_columns, per_prop_scores)}
        std = {p: float(np.std(s)) for p, s in zip(self.property_columns, per_prop_scores)}
        return mean, std

    # --- prediction --------------------------------------------------------

    def predict(self, composition: dict[str, float]) -> PredictionResult:
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")
        x = self._compose_vector(composition)
        y_pred = self.pipeline.predict(x[None, :])[0]
        if np.ndim(y_pred) == 0:
            y_pred = np.array([y_pred])
        props = {p: float(v) for p, v in zip(self.property_columns, y_pred)}
        uncert = self._tree_uncertainty(x)
        return PredictionResult(
            properties=props,
            uncertainty=uncert,
            model_r2=self.report.val_r2 if self.report else None,
        )

    def predict_batch(self, compositions: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")
        preds = self.pipeline.predict(compositions)
        if preds.ndim == 1:
            preds = preds[:, None]
        return preds

    def _compose_vector(self, composition: dict[str, float]) -> np.ndarray:
        missing = [f for f in self.feature_columns if f not in composition]
        if missing:
            raise ValueError(f"Composition missing elements: {missing}")
        return np.array([composition[f] for f in self.feature_columns], dtype=float)

    def _tree_uncertainty(self, x: np.ndarray) -> dict[str, float] | None:
        est = self.pipeline.named_steps["estimator"]
        if not isinstance(est, RandomForestRegressor):
            return None
        scaled = self.pipeline.named_steps["scaler"].transform(x[None, :])
        tree_preds = np.stack([t.predict(scaled) for t in est.estimators_])
        # tree_preds shape: (n_trees, 1, n_outputs) or (n_trees, 1)
        if tree_preds.ndim == 2:
            tree_preds = tree_preds[:, :, None]
        std = tree_preds.std(axis=0)[0]
        return {p: float(s) for p, s in zip(self.property_columns, std)}

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_columns": self.feature_columns,
                "property_columns": self.property_columns,
                "report": self.report.to_dict() if self.report else None,
                "estimator_name": self.estimator_name,
                "random_state": self.random_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PropertyPredictor":
        state = joblib.load(path)
        inst = cls(estimator=state["estimator_name"],
                   random_state=state.get("random_state", 42))
        inst.pipeline = state["pipeline"]
        inst.feature_columns = state["feature_columns"]
        inst.property_columns = state["property_columns"]
        rep = state.get("report")
        if rep is not None:
            inst.report = TrainingReport(**rep)
        return inst


# ---------------------------------------------------------------------------
# Composition designer (inverse design + verification + feasibility)
# ---------------------------------------------------------------------------

class CompositionDesigner:
    """Inverse design + verification using a trained :class:`PropertyPredictor`.

    Two sampling strategies are exposed:

    * ``strategy="dirichlet"`` — fast Monte Carlo with a Dirichlet prior
      (the baseline from the reference sketch).
    * ``strategy="ga"`` — a small genetic algorithm that mutates the best
      Dirichlet candidates and converges on lower-MSE designs.

    Both strategies honour :class:`DesignConstraints`.
    """

    def __init__(self, predictor: PropertyPredictor) -> None:
        if predictor.pipeline is None:
            raise RuntimeError("Predictor must be trained before design tasks.")
        self.predictor = predictor

    # --- sampling ----------------------------------------------------------

    def _sample_dirichlet(
        self,
        n: int,
        constraints: DesignConstraints | None,
        rng: np.random.Generator,
    ) -> np.ndarray:
        feats = self.predictor.feature_columns
        n_el = len(feats)
        if constraints is None:
            return rng.dirichlet(np.ones(n_el), size=n)

        fixed = constraints.fixed
        lo_map = constraints.min_fraction
        hi_map = constraints.max_fraction

        free_idx = [i for i, f in enumerate(feats) if f not in fixed]
        budget = 1.0 - sum(fixed.values())
        if budget < -1e-9:
            raise ValueError("Fixed fractions exceed 1.0.")

        # Reserve each free element's minimum out of the budget; sample only
        # the *remainder* via Dirichlet so the minimum is exactly respected.
        free_feats = [feats[i] for i in free_idx]
        lo_vec = np.array([lo_map.get(f, 0.0) for f in free_feats])
        hi_vec = np.array([hi_map.get(f, 1.0) for f in free_feats])
        if lo_vec.sum() > budget + 1e-9:
            raise ValueError(
                f"min_fraction sum ({lo_vec.sum():.3f}) exceeds available "
                f"budget after fixed ({budget:.3f})."
            )
        if (hi_vec < lo_vec).any():
            raise ValueError("max_fraction below min_fraction for some element.")

        samples = np.zeros((n, n_el), dtype=float)
        for i, f in enumerate(feats):
            if f in fixed:
                samples[:, i] = fixed[f]

        if free_idx:
            remainder = budget - lo_vec.sum()
            if remainder < 0:
                remainder = 0.0
            extra = (
                rng.dirichlet(np.ones(len(free_idx)), size=n) * remainder
                if remainder > 0 else np.zeros((n, len(free_idx)))
            )
            free_samples = lo_vec[None, :] + extra
            # If max_fraction is binding, water-fill: clip and redistribute
            # excess among elements that still have headroom.
            if (hi_vec < 1.0).any():
                free_samples = _project_to_box_simplex(
                    free_samples, lo_vec, hi_vec, budget
                )
            for j, fi in enumerate(free_idx):
                samples[:, fi] = free_samples[:, j]
        return samples

    def _mutate(
        self,
        parents: np.ndarray,
        constraints: DesignConstraints | None,
        sigma: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        children = parents + rng.normal(0, sigma, size=parents.shape)
        children = np.clip(children, 0.0, 1.0)
        feats = self.predictor.feature_columns
        if constraints is None:
            row_sums = children.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return children / row_sums

        # Pin fixed columns, then project the free sub-vector to satisfy
        # min / max / budget.
        fixed = constraints.fixed
        for i, f in enumerate(feats):
            if f in fixed:
                children[:, i] = fixed[f]
        free_idx = [i for i, f in enumerate(feats) if f not in fixed]
        budget = 1.0 - sum(fixed.values())
        if free_idx and budget > 0:
            free_feats = [feats[i] for i in free_idx]
            lo_vec = np.array([constraints.min_fraction.get(f, 0.0)
                               for f in free_feats])
            hi_vec = np.array([constraints.max_fraction.get(f, 1.0)
                               for f in free_feats])
            block = children[:, free_idx]
            block = _project_to_box_simplex(block, lo_vec, hi_vec, budget)
            children[:, free_idx] = block
        return children

    # --- inverse design ----------------------------------------------------

    def design_inverse(
        self,
        target_properties: dict[str, float],
        weights: dict[str, float] | None = None,
        num_candidates: int = 5000,
        top_k: int = 5,
        constraints: DesignConstraints | None = None,
        strategy: str = "dirichlet",
        ga_generations: int = 5,
        ga_elite_frac: float = 0.1,
        ga_mutation: float = 0.05,
        random_state: int | None = None,
    ) -> list[DesignCandidate]:
        """Return ``top_k`` candidate compositions sorted by weighted MSE.

        The score for each candidate is::

            score = mean(w_p * (pred_p - target_p) ** 2)

        across all target properties ``p``. ``weights`` defaults to all 1.0.
        """
        prop_cols = self.predictor.property_columns
        missing = [p for p in prop_cols if p not in target_properties]
        if missing:
            raise ValueError(f"Missing target values for properties: {missing}")
        if constraints is not None:
            constraints.validate(self.predictor.feature_columns)

        target_vec = np.array([target_properties[p] for p in prop_cols], dtype=float)
        w = np.array([(weights or {}).get(p, 1.0) for p in prop_cols], dtype=float)
        # normalise per-property so targets at very different scales don't dominate
        scale = np.where(np.abs(target_vec) > 1e-9, np.abs(target_vec), 1.0)
        w_eff = w / (scale ** 2)

        rng = np.random.default_rng(random_state)
        samples = self._sample_dirichlet(num_candidates, constraints, rng)
        preds = self.predictor.predict_batch(samples)
        scores = ((preds - target_vec) ** 2 * w_eff).mean(axis=1)

        if strategy == "ga":
            elite_n = max(2, int(num_candidates * ga_elite_frac))
            for _ in range(ga_generations):
                elite_idx = np.argsort(scores)[:elite_n]
                elites = samples[elite_idx]
                # reproduce: repeat elites + mutate to fill population
                repeats = int(np.ceil(num_candidates / elite_n))
                children = self._mutate(
                    np.repeat(elites, repeats, axis=0)[:num_candidates],
                    constraints, ga_mutation, rng,
                )
                child_preds = self.predictor.predict_batch(children)
                child_scores = ((child_preds - target_vec) ** 2 * w_eff).mean(axis=1)
                # merge and keep the best `num_candidates`
                pool = np.vstack([samples, children])
                pool_preds = np.vstack([preds, child_preds])
                pool_scores = np.concatenate([scores, child_scores])
                keep = np.argsort(pool_scores)[:num_candidates]
                samples, preds, scores = pool[keep], pool_preds[keep], pool_scores[keep]
        elif strategy != "dirichlet":
            raise ValueError(f"Unknown strategy: {strategy}")

        best = np.argsort(scores)[:top_k]
        out: list[DesignCandidate] = []
        for idx in best:
            comp_arr = samples[idx]
            comp = {f: float(v) for f, v in zip(self.predictor.feature_columns, comp_arr)}
            pred = {p: float(v) for p, v in zip(prop_cols, preds[idx])}
            rel = {
                p: float(abs(pred[p] - target_properties[p]) /
                         (abs(target_properties[p]) if target_properties[p] != 0 else 1.0))
                for p in prop_cols
            }
            uncert = self._uncertainty_for(comp_arr)
            out.append(DesignCandidate(
                composition=comp, predicted=pred, uncertainty=uncert,
                score=float(scores[idx]), rel_errors=rel,
            ))
        return out

    def _uncertainty_for(self, comp_vec: np.ndarray) -> dict[str, float] | None:
        est = self.predictor.pipeline.named_steps["estimator"]
        if not isinstance(est, RandomForestRegressor):
            return None
        scaled = self.predictor.pipeline.named_steps["scaler"].transform(comp_vec[None, :])
        tree_preds = np.stack([t.predict(scaled) for t in est.estimators_])
        if tree_preds.ndim == 2:
            tree_preds = tree_preds[:, :, None]
        std = tree_preds.std(axis=0)[0]
        return {p: float(s) for p, s in zip(self.predictor.property_columns, std)}

    # --- verification / feasibility ----------------------------------------

    def verify_composition(self, composition: dict[str, float]) -> PredictionResult:
        return self.predictor.predict(composition)

    def analyse_feasibility(
        self,
        composition: dict[str, float],
        target_properties: dict[str, float] | None = None,
        tolerance: float = 0.1,
    ) -> dict[str, Any]:
        """Return predicted properties + (optional) per-target gap analysis.

        ``tolerance`` is the relative error threshold under which a property
        is considered to "meet" the target. The summary includes:

        * ``predicted`` — predicted property dict
        * ``uncertainty`` — per-property std-dev (RF only) or None
        * ``relative_errors`` — |pred - target| / |target|
        * ``meets_target`` — per-property bool against ``tolerance``
        * ``overall_feasible`` — True iff every property meets the target
        * ``recommendation`` — short text suggestion
        """
        pred = self.verify_composition(composition)
        out: dict[str, Any] = {
            "composition": composition,
            "predicted": pred.properties,
            "uncertainty": pred.uncertainty,
        }
        if target_properties is None:
            return out
        rel: dict[str, float] = {}
        meets: dict[str, bool] = {}
        for prop, target in target_properties.items():
            if prop not in pred.properties:
                continue
            denom = abs(target) if abs(target) > 1e-9 else 1.0
            err = abs(pred.properties[prop] - target) / denom
            rel[prop] = float(err)
            meets[prop] = bool(err <= tolerance)
        out["relative_errors"] = rel
        out["meets_target"] = meets
        out["overall_feasible"] = bool(meets and all(meets.values()))
        out["recommendation"] = self._recommend(pred.properties, target_properties, meets)
        return out

    @staticmethod
    def _recommend(
        pred: dict[str, float],
        target: dict[str, float],
        meets: dict[str, bool],
    ) -> str:
        lines = []
        for prop, ok in meets.items():
            if ok:
                continue
            delta = pred[prop] - target[prop]
            direction = "decrease" if delta > 0 else "increase"
            lines.append(
                f"- '{prop}' currently {pred[prop]:.3g} vs target {target[prop]:.3g}; "
                f"composition adjustments should {direction} this property."
            )
        if not lines:
            return "All target properties are within the requested tolerance."
        return "\n".join(["Suggested adjustments:"] + lines)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a CSV. Identifier columns (non-numeric) are kept but ignored by
    the auto-detection in :meth:`PropertyPredictor.train`.
    """
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _project_to_box_simplex(
    samples: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    budget: float,
    max_iter: int = 20,
) -> np.ndarray:
    """Project each row of ``samples`` so that ``lo <= row <= hi`` and the
    row sums to ``budget``. Used to enforce per-element min/max while
    still allocating a fixed total budget across the free elements.
    Iterative water-filling — converges in a few iterations for typical
    constraint sets.
    """
    out = samples.copy()
    for _ in range(max_iter):
        out = np.clip(out, lo, hi)
        row_sums = out.sum(axis=1, keepdims=True)
        diff = budget - row_sums
        if np.all(np.abs(diff) < 1e-9):
            break
        # capacity that can absorb up/down within bounds
        if (diff > 0).any():
            cap_up = (hi - out)
            cap_up_sum = cap_up.sum(axis=1, keepdims=True)
            cap_up_sum[cap_up_sum == 0] = 1.0
            mask = (diff > 0).flatten()
            if mask.any():
                out[mask] += cap_up[mask] * (diff[mask] / cap_up_sum[mask])
        if (diff < 0).any():
            cap_dn = (out - lo)
            cap_dn_sum = cap_dn.sum(axis=1, keepdims=True)
            cap_dn_sum[cap_dn_sum == 0] = 1.0
            mask = (diff < 0).flatten()
            if mask.any():
                out[mask] += cap_dn[mask] * (diff[mask] / cap_dn_sum[mask])
    return np.clip(out, lo, hi)


__all__ = [
    "AVAILABLE_ESTIMATORS",
    "CompositionDesigner",
    "DesignCandidate",
    "DesignConstraints",
    "PredictionResult",
    "PropertyPredictor",
    "TrainingReport",
    "load_dataset",
    "save_dataset",
]
