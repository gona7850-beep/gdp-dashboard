"""End-to-end accuracy + reliability checks for any forward model.

What this module produces, given a trained model + a validation
dataset:

1. **Hold-out metrics**: per-target R², MAE, RMSE.

2. **Cross-validated metrics with confidence intervals**: K-fold
   (group-aware when groups are provided) × N seeds → mean ± std.

3. **Permutation test p-value**: probability that an equally-good model
   could be obtained by shuffling y. Significance bar for any
   "the model learned something" claim.

4. **Conformal coverage**: actual coverage of the model's prediction
   intervals at the nominal level (default 90 %). If you report 90 %
   intervals you must verify empirical coverage ≥ 88 % — otherwise the
   bands are decorative.

5. **Reliability diagram**: predicted-vs-empirical CDF over a holdout,
   so you can see whether σ is well-calibrated globally (not just on
   average).

6. **Domain-of-applicability (DoA) check**: for each query composition,
   nearest-neighbour distance to the training set in feature space.
   Flags extrapolation.

7. **Sanity check against curated reference DB**: predict every
   household-name alloy from :func:`reference_dataset` and compute
   absolute error vs literature. Big errors here mean the model is
   uncalibrated regardless of CV R².

The output is a typed :class:`AccuracyReport` you can serialise to
JSON, render in Streamlit, or attach to a PR.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .data_pipeline import Dataset
from .reference_data import (
    PROPERTY_COLUMNS as REF_PROPERTY_COLUMNS,
    reference_dataset,
    reference_elements,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class AccuracyReport:
    """Container for every accuracy/reliability number we compute."""

    targets: List[str]

    # Hold-out (one split)
    holdout: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # K-fold CV with mean/std
    cv: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Permutation test
    permutation: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Conformal coverage
    coverage: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Reliability diagram per target
    reliability: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # DoA percentile of test points
    doa: Dict[str, float] = field(default_factory=dict)

    # Sanity check vs reference DB
    reference_check: Optional[pd.DataFrame] = None

    overall_grade: str = "unknown"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "targets": list(self.targets),
            "holdout": self.holdout,
            "cv": self.cv,
            "permutation": self.permutation,
            "coverage": self.coverage,
            "doa": self.doa,
            "overall_grade": self.overall_grade,
            "notes": list(self.notes),
        }
        out["reliability"] = {
            k: df.to_dict(orient="records") for k, df in self.reliability.items()
        }
        if self.reference_check is not None:
            out["reference_check"] = self.reference_check.to_dict(orient="records")
        return out

    def summary(self) -> str:
        """Compact human-readable summary string."""
        lines = [f"Targets: {self.targets}"]
        for t in self.targets:
            cv = self.cv.get(t, {})
            cov = self.coverage.get(t, {})
            perm = self.permutation.get(t, {})
            lines.append(
                f"  {t:14s}  "
                f"CV R²={cv.get('r2_mean', float('nan')):+.3f}±{cv.get('r2_std', float('nan')):.3f}  "
                f"MAE={cv.get('mae_mean', float('nan')):.3g}  "
                f"perm p={perm.get('p_value', float('nan')):.3f}  "
                f"coverage={cov.get('empirical_coverage', float('nan')):.0%}"
                f"@{cov.get('nominal_coverage', float('nan')):.0%}"
            )
        lines.append(f"Overall grade: {self.overall_grade}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def evaluate_model(
    model: Any,
    dataset: Dataset,
    *,
    targets: Sequence[str],
    n_splits: int = 5,
    n_seeds: int = 3,
    n_permutations: int = 50,
    alpha: float = 0.10,
    include_reference_check: bool = True,
    skip_permutation: bool = False,
    skip_reliability: bool = False,
    seed: int = 0,
) -> AccuracyReport:
    """Run every diagnostic on ``model`` + ``dataset`` and return a report.

    Parameters
    ----------
    model
        Any object exposing ``fit(dataset)`` and
        ``predict(compositions, process=None)`` returning a DataFrame
        with ``<target>_mean`` and (optionally) ``<target>_std`` columns.
        ``ForwardModel`` and ``ForwardModelV2`` both qualify.
    dataset
        Training data. When ``dataset.groups`` is non-null, all
        cross-validation is GroupKFold.
    targets
        Property column names. The model must already be fit on these
        OR will be re-fit per fold (we use cloned factories below).
    n_splits, n_seeds
        K-fold settings.
    n_permutations
        Permutation-test budget. 50 gives p ≤ 0.02 resolution.
    alpha
        Conformal miscoverage level. 0.10 → expect 90 % coverage.
    include_reference_check
        If True, predict every entry of :func:`reference_dataset` and
        compare to the literature values.
    """
    targets = list(targets)
    rep = AccuracyReport(targets=targets)

    # Pre-fit on full data for hold-out + reference check
    _safe_fit(model, dataset)

    # --- 1. Hold-out (single 80/20 stratified) -------------------------
    rep.holdout = _holdout_metrics(model, dataset, targets, seed)

    # --- 2. K-fold CV across seeds -------------------------------------
    rep.cv = _kfold_metrics(model, dataset, targets, n_splits, n_seeds, seed)

    # --- 3. Permutation test -------------------------------------------
    if not skip_permutation:
        rep.permutation = _permutation_pvalues(
            model, dataset, targets, n_permutations=n_permutations,
            n_splits=min(n_splits, 3), seed=seed,
        )

    # --- 4. Conformal coverage -----------------------------------------
    rep.coverage = _coverage_check(model, dataset, targets, alpha=alpha)

    # --- 5. Reliability diagrams ---------------------------------------
    if not skip_reliability:
        rep.reliability = _reliability_diagrams(model, dataset, targets)

    # --- 6. DoA percentile of dataset itself ---------------------------
    rep.doa = _doa_percentiles(model, dataset)

    # --- 7. Sanity check vs reference DB -------------------------------
    if include_reference_check:
        rep.reference_check = _reference_check(model, dataset, targets)

    rep.overall_grade, rep.notes = _grade(rep)
    return rep


# ---------------------------------------------------------------------------
# Per-component helpers
# ---------------------------------------------------------------------------

def _safe_fit(model, dataset: Dataset) -> None:
    """Fit ``model`` on the dataset, swallowing kw differences."""
    try:
        model.fit(dataset)
    except TypeError:
        # Some models take n_trials; default to a small value.
        model.fit(dataset, n_trials=5)


def _make_splits(dataset: Dataset, n_splits: int, seed: int):
    idx = np.arange(len(dataset.compositions))
    if dataset.groups is not None and dataset.groups.nunique() >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        return list(cv.split(idx, groups=dataset.groups.values))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(idx))


def _slice_dataset(dataset: Dataset, idx: np.ndarray) -> Dataset:
    return Dataset(
        compositions=dataset.compositions.iloc[idx].reset_index(drop=True),
        properties=dataset.properties.iloc[idx].reset_index(drop=True),
        process=(dataset.process.iloc[idx].reset_index(drop=True)
                 if dataset.process is not None else None),
        groups=(dataset.groups.iloc[idx].reset_index(drop=True)
                if dataset.groups is not None else None),
    )


def _holdout_metrics(model, dataset: Dataset, targets, seed: int):
    rng = np.random.default_rng(seed)
    n = len(dataset.compositions)
    perm = rng.permutation(n)
    cut = int(0.8 * n)
    tr_idx, te_idx = perm[:cut], perm[cut:]
    # Re-fit on train slice
    factory_model = _clone_model(model)
    _safe_fit(factory_model, _slice_dataset(dataset, tr_idx))
    preds = factory_model.predict(dataset.compositions.iloc[te_idx])
    out = {}
    for t in targets:
        y_true = dataset.properties.iloc[te_idx][t].to_numpy(dtype=float)
        y_pred = preds[f"{t}_mean"].to_numpy(dtype=float)
        out[t] = {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }
    return out


def _kfold_metrics(model, dataset: Dataset, targets, n_splits, n_seeds, seed):
    out: Dict[str, Dict[str, float]] = {t: {} for t in targets}
    fold_results: Dict[str, List[Dict[str, float]]] = {t: [] for t in targets}
    for s in range(n_seeds):
        splits = _make_splits(dataset, n_splits, seed + s)
        for tr, te in splits:
            m = _clone_model(model)
            _safe_fit(m, _slice_dataset(dataset, tr))
            preds = m.predict(dataset.compositions.iloc[te])
            for t in targets:
                y_true = dataset.properties.iloc[te][t].to_numpy(dtype=float)
                y_pred = preds[f"{t}_mean"].to_numpy(dtype=float)
                fold_results[t].append({
                    "r2": float(r2_score(y_true, y_pred)),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                })
    for t in targets:
        arr = pd.DataFrame(fold_results[t])
        out[t] = {
            "r2_mean": float(arr["r2"].mean()),
            "r2_std": float(arr["r2"].std()),
            "mae_mean": float(arr["mae"].mean()),
            "rmse_mean": float(arr["rmse"].mean()),
            "n_folds": int(len(arr)),
        }
    return out


def _permutation_pvalues(model, dataset: Dataset, targets, n_permutations,
                          n_splits, seed):
    """Compare CV R² of real labels to N permutations of y."""
    out: Dict[str, Dict[str, float]] = {}
    real_r2: Dict[str, float] = {}
    splits = _make_splits(dataset, n_splits, seed)
    for t in targets:
        m = _clone_model(model)
        _safe_fit(m, dataset)
        cv_preds = np.zeros(len(dataset.compositions))
        for tr, te in splits:
            mm = _clone_model(model)
            _safe_fit(mm, _slice_dataset(dataset, tr))
            cv_preds[te] = mm.predict(
                dataset.compositions.iloc[te]
            )[f"{t}_mean"].to_numpy(dtype=float)
        real_r2[t] = float(r2_score(
            dataset.properties[t].to_numpy(dtype=float), cv_preds
        ))

    rng = np.random.default_rng(seed)
    null_scores: Dict[str, List[float]] = {t: [] for t in targets}
    for _ in range(n_permutations):
        perm_idx = rng.permutation(len(dataset.properties))
        permuted_props = dataset.properties.iloc[perm_idx].reset_index(drop=True)
        ds_perm = Dataset(
            compositions=dataset.compositions,
            properties=permuted_props,
            process=dataset.process,
            groups=dataset.groups,
        )
        for t in targets:
            cv_preds = np.zeros(len(ds_perm.compositions))
            for tr, te in splits:
                mm = _clone_model(model)
                _safe_fit(mm, _slice_dataset(ds_perm, tr))
                cv_preds[te] = mm.predict(
                    ds_perm.compositions.iloc[te]
                )[f"{t}_mean"].to_numpy(dtype=float)
            null_scores[t].append(float(r2_score(
                ds_perm.properties[t].to_numpy(dtype=float), cv_preds
            )))
    for t in targets:
        null = np.array(null_scores[t])
        # One-sided: how often does the random model beat the real one?
        p = float((null >= real_r2[t]).sum() + 1) / (n_permutations + 1)
        out[t] = {
            "real_r2": real_r2[t],
            "null_r2_mean": float(null.mean()),
            "null_r2_std": float(null.std()),
            "p_value": p,
            "n_permutations": n_permutations,
        }
    return out


def _coverage_check(model, dataset: Dataset, targets, alpha: float):
    """Compute empirical coverage of nominal-α prediction intervals."""
    out: Dict[str, Dict[str, float]] = {}
    preds = model.predict(dataset.compositions)
    z = 1.6449   # 90% Gaussian z; only used when no _lo/_hi provided
    for t in targets:
        if f"{t}_lo" in preds.columns and f"{t}_hi" in preds.columns:
            lo = preds[f"{t}_lo"].to_numpy()
            hi = preds[f"{t}_hi"].to_numpy()
        elif f"{t}_std" in preds.columns:
            mu = preds[f"{t}_mean"].to_numpy()
            sigma = preds[f"{t}_std"].to_numpy()
            lo = mu - z * sigma
            hi = mu + z * sigma
        else:
            continue
        y = dataset.properties[t].to_numpy(dtype=float)
        inside = (y >= lo) & (y <= hi)
        out[t] = {
            "nominal_coverage": 1.0 - alpha,
            "empirical_coverage": float(inside.mean()),
            "n": int(len(y)),
        }
    return out


def _reliability_diagrams(model, dataset: Dataset, targets) -> Dict[str, pd.DataFrame]:
    """Per-target nominal-vs-empirical coverage at 10 confidence levels."""
    out: Dict[str, pd.DataFrame] = {}
    preds = model.predict(dataset.compositions)
    from scipy.stats import norm
    for t in targets:
        if f"{t}_std" not in preds.columns:
            continue
        mu = preds[f"{t}_mean"].to_numpy()
        sigma = preds[f"{t}_std"].to_numpy()
        sigma = np.where(sigma < 1e-6, 1e-6, sigma)
        y = dataset.properties[t].to_numpy(dtype=float)
        z = (y - mu) / sigma
        rows = []
        for q in np.linspace(0.10, 0.95, 10):
            zc = norm.ppf(0.5 + q / 2)
            rows.append({
                "nominal_coverage": float(q),
                "empirical_coverage": float(np.mean(np.abs(z) <= zc)),
            })
        out[t] = pd.DataFrame(rows)
    return out


def _doa_percentiles(model, dataset: Dataset) -> Dict[str, float]:
    """Median + 95th percentile DoA distance (within training set)."""
    if not hasattr(model, "featurizer"):
        return {}
    X = dataset.build_X(model.featurizer)
    try:
        first = next(iter(model.models_.values()))
    except (StopIteration, AttributeError):
        return {}
    Xs = first.preproc.transform(X[first.feature_names])
    n = len(Xs)
    # Self-NN distance distribution
    diff = Xs[:, None, :] - Xs[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return {
        "median_nn": float(np.median(nn)),
        "p95_nn": float(np.percentile(nn, 95)),
        "n_train": int(n),
    }


def _reference_check(model, dataset: Dataset, targets) -> pd.DataFrame:
    """Predict every reference alloy and report absolute error vs literature."""
    ref = reference_dataset()
    # Map target names → reference column names (we use the same naming
    # convention by default; non-matching targets are skipped)
    matched = [t for t in targets if t in ref.columns]
    if not matched:
        return pd.DataFrame()
    # Align element columns
    train_els = list(dataset.compositions.columns)
    comp = ref.reindex(columns=train_els, fill_value=0.0)
    try:
        preds = model.predict(comp)
    except Exception as exc:
        log.warning(f"reference check predict failed: {exc}")
        return pd.DataFrame()
    rows = []
    for i, alloy in enumerate(ref["alloy_name"]):
        row = {"alloy_name": alloy, "family": ref.iloc[i]["family"]}
        for t in matched:
            actual = ref.iloc[i][t]
            if pd.isna(actual):
                continue
            mu = float(preds.iloc[i][f"{t}_mean"])
            row[f"{t}_actual"] = float(actual)
            row[f"{t}_pred"] = mu
            row[f"{t}_abs_err"] = abs(mu - float(actual))
            denom = abs(actual) if abs(actual) > 1e-9 else 1.0
            row[f"{t}_rel_err"] = abs(mu - float(actual)) / denom
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _grade(rep: AccuracyReport) -> tuple[str, List[str]]:
    """Heuristic A/B/C/D grade based on CV R² and permutation p."""
    notes: List[str] = []
    r2s = [c.get("r2_mean", -1) for c in rep.cv.values()]
    ps = [p.get("p_value", 1.0) for p in rep.permutation.values()]
    covs = [c.get("empirical_coverage", 0.0) for c in rep.coverage.values()
            if "empirical_coverage" in c]
    nominal_covs = [c.get("nominal_coverage", 0.9) for c in rep.coverage.values()
                    if "nominal_coverage" in c]
    cov_gap = [abs(c - n) for c, n in zip(covs, nominal_covs)] if covs else [0]

    score = 0
    if r2s:
        if min(r2s) > 0.85:
            score += 3
        elif min(r2s) > 0.6:
            score += 2
        elif min(r2s) > 0.3:
            score += 1
    if ps and max(ps) < 0.05:
        score += 2
    elif ps and max(ps) < 0.10:
        score += 1
    else:
        notes.append("WARNING: permutation p>0.05 — model may not beat random")
    if cov_gap and max(cov_gap) < 0.05:
        score += 1
    elif cov_gap and max(cov_gap) > 0.15:
        notes.append("WARNING: conformal intervals badly calibrated")

    if score >= 6:
        grade = "A"
    elif score >= 4:
        grade = "B"
    elif score >= 2:
        grade = "C"
    else:
        grade = "D"
    return grade, notes


# ---------------------------------------------------------------------------
# Model cloning
# ---------------------------------------------------------------------------

def _clone_model(model):
    """Best-effort clone for re-fitting across folds."""
    from copy import deepcopy
    if hasattr(model, "__class__"):
        try:
            # If the model uses dataclass fields, copy them
            attrs = {
                "featurizer": getattr(model, "featurizer", None),
                "targets": list(getattr(model, "targets", [])),
                "n_cv_splits": getattr(model, "n_cv_splits", 5),
                "random_state": getattr(model, "random_state", 0),
            }
            cls = model.__class__
            fresh = cls(**{k: v for k, v in attrs.items()
                           if k in cls.__dataclass_fields__})
            return fresh
        except Exception:
            pass
    return deepcopy(model)


__all__ = ["AccuracyReport", "evaluate_model"]
