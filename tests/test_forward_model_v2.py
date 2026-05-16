"""Smoke tests for ForwardModelV2 + extended physics features + benchmark."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost", reason="xgboost not installed")
pytest.importorskip("optuna", reason="optuna not installed")

from core.alloyforge.data_pipeline import CompositionFeaturizer, Dataset
from core.alloyforge.forward_model_v2 import ForwardModelV2
from core.alloyforge.physics_features import ExtendedFeaturizer, make_extended


@pytest.fixture(scope="module")
def tiny_dataset():
    rng = np.random.default_rng(0)
    comp = rng.dirichlet([2, 1, 1, 0.5, 0.3], size=80)
    df = pd.DataFrame(comp, columns=["Fe", "Ni", "Cr", "Mo", "Ti"])
    df["hv"] = (200 + 700 * df["Mo"] + 500 * df["Cr"] + 900 * df["Ti"]
                + rng.normal(0, 10, 80))
    df["uts"] = (400 + 900 * df["Mo"] + 600 * df["Cr"] + 200 * df["Ni"]
                 + rng.normal(0, 20, 80))
    return df


# ---------------------------------------------------------------------------
# Extended featurizer
# ---------------------------------------------------------------------------

def test_extended_featurizer_no_column_collision(tiny_dataset):
    ext = make_extended(["Fe", "Ni", "Cr", "Mo", "Ti"])
    X = ext.transform(tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]])
    # base 33 + ext 10
    assert X.shape == (80, 43)
    assert not X.columns.duplicated().any(), "duplicate columns leaked"


def test_extended_featurizer_omega_clipped(tiny_dataset):
    ext = make_extended(["Fe", "Ni", "Cr", "Mo", "Ti"])
    X = ext.transform(tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]])
    assert (X["Omega_yang"] >= 0).all()
    assert (X["Omega_yang"] <= 50).all(), "Omega should be clipped"


def test_extended_featurizer_vec_buckets_sum_sensible(tiny_dataset):
    ext = make_extended(["Fe", "Ni", "Cr", "Mo", "Ti"])
    X = ext.transform(tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]])
    # Each bucket prob ∈ [0, 1]
    for col in ["vec_bcc_prob", "vec_dual_prob", "vec_mixed_prob", "vec_fcc_prob"]:
        assert (X[col] >= 0).all() and (X[col] <= 1).all()


# ---------------------------------------------------------------------------
# ForwardModelV2 contract
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_v2(tiny_dataset):
    ds = Dataset(
        compositions=tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]],
        properties=tiny_dataset[["hv", "uts"]],
    )
    feat = CompositionFeaturizer(element_columns=["Fe", "Ni", "Cr", "Mo", "Ti"])
    fm = ForwardModelV2(
        featurizer=feat, targets=["hv", "uts"],
        n_seeds=2, n_cv_splits=3, share_targets=False,
        n_trials=2, random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm.fit(ds)
    return fm


def test_v2_predict_returns_mean_and_std(fitted_v2, tiny_dataset):
    preds = fitted_v2.predict(tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]].head(5))
    for t in ["hv", "uts"]:
        assert f"{t}_mean" in preds.columns
        assert f"{t}_std" in preds.columns
        assert (preds[f"{t}_std"] >= 0).all()


def test_v2_decomposed_uncertainty_decomposes(fitted_v2, tiny_dataset):
    preds = fitted_v2.predict(
        tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]].head(5),
        return_decomposed=True,
    )
    for t in ["hv", "uts"]:
        ep = preds[f"{t}_epistemic"].to_numpy()
        al = preds[f"{t}_aleatoric"].to_numpy()
        total = preds[f"{t}_std"].to_numpy()
        # Total ≈ sqrt(epi² + ale²); allow small numerical slack.
        recomputed = np.sqrt(ep ** 2 + al ** 2)
        np.testing.assert_allclose(total, recomputed, atol=1e-6)
        assert (ep >= 0).all() and (al >= 0).all()


def test_v2_report_per_target(fitted_v2):
    rep = fitted_v2.report()
    assert set(rep.index) == {"hv", "uts"}
    for t in ["hv", "uts"]:
        assert np.isfinite(rep.loc[t, "cv_r2"])


def test_v2_multi_task_runs(tiny_dataset):
    ds = Dataset(
        compositions=tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]],
        properties=tiny_dataset[["hv", "uts"]],
    )
    feat = CompositionFeaturizer(element_columns=["Fe", "Ni", "Cr", "Mo", "Ti"])
    fm = ForwardModelV2(
        featurizer=feat, targets=["hv", "uts"],
        n_seeds=2, n_cv_splits=3, share_targets=True,
        n_trials=2, random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm.fit(ds)
    preds = fm.predict(tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]].head(3))
    assert {"hv_mean", "uts_mean"}.issubset(preds.columns)


def test_v2_with_extended_featurizer_runs(tiny_dataset):
    ds = Dataset(
        compositions=tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]],
        properties=tiny_dataset[["hv"]],
    )
    feat = make_extended(["Fe", "Ni", "Cr", "Mo", "Ti"])
    fm = ForwardModelV2(
        featurizer=feat, targets=["hv"],
        n_seeds=2, n_cv_splits=3, share_targets=False,
        n_trials=2, random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm.fit(ds)
    preds = fm.predict(tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]].head(3))
    assert "hv_mean" in preds.columns


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def test_benchmark_compare_runs(tiny_dataset):
    from core.alloyforge.benchmark import benchmark_models
    from core.alloyforge.forward_model import ForwardModel

    ds = Dataset(
        compositions=tiny_dataset[["Fe", "Ni", "Cr", "Mo", "Ti"]],
        properties=tiny_dataset[["hv"]],
    )

    def make_v1():
        m = ForwardModel(
            featurizer=CompositionFeaturizer(
                element_columns=["Fe", "Ni", "Cr", "Mo", "Ti"]
            ),
            targets=["hv"], n_cv_splits=3,
        )
        m.fit = lambda d, **kw: ForwardModel.fit(m, d, n_trials=2)
        return m

    def make_v2():
        return ForwardModelV2(
            featurizer=CompositionFeaturizer(
                element_columns=["Fe", "Ni", "Cr", "Mo", "Ti"]
            ),
            targets=["hv"], n_seeds=2, n_cv_splits=3,
            share_targets=False, n_trials=2, random_state=0,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lb = benchmark_models(
            models={"v1": make_v1, "v2": make_v2},
            dataset=ds,
            element_columns=["Fe", "Ni", "Cr", "Mo", "Ti"],
            targets=["hv"], n_splits=3, seed=0,
        )
    assert set(lb["model"]) == {"v1", "v2"}
    assert set(lb["target"]) == {"hv"}
    for col in ["r2_mean", "r2_std", "mae_mean", "rmse_mean", "fit_seconds"]:
        assert col in lb.columns
