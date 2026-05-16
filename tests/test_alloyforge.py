"""Smoke tests for the AlloyForge advanced ML stack.

These tests verify the pipeline wires together: featurize → fit → predict
→ calibrate → inverse-design → explain. They do NOT verify model quality —
that requires a real dataset.

Skipped automatically if the heavy ML deps (xgboost, optuna, pymoo, shap)
are not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost", reason="xgboost not installed")
pytest.importorskip("optuna", reason="optuna not installed")

from core.alloyforge.data_pipeline import CompositionFeaturizer, Dataset
from core.alloyforge.feasibility import default_checker, hume_rothery_size_mismatch
from core.alloyforge.forward_model import ForwardModel
from core.alloyforge.llm_assistant import LLMAssistant
from core.alloyforge.validation import ConformalCalibrator, DomainOfApplicability


@pytest.fixture(scope="module")
def tiny_dataset():
    rng = np.random.default_rng(0)
    comp = rng.dirichlet([2, 1, 1, 0.5], size=60)
    df = pd.DataFrame(comp, columns=["Fe", "Ni", "Cr", "Mo"])
    df["hardness_hv"] = 200 + 600 * df["Mo"] + 400 * df["Cr"] + rng.normal(0, 10, 60)
    df["tensile_mpa"] = 400 + 800 * df["Mo"] + 500 * df["Cr"] + rng.normal(0, 20, 60)
    df["heat_id"] = [f"H{i // 4}" for i in range(60)]
    return df


def test_featurizer_shape(tiny_dataset):
    feat = CompositionFeaturizer(element_columns=["Fe", "Ni", "Cr", "Mo"])
    X = feat.transform(tiny_dataset[["Fe", "Ni", "Cr", "Mo"]])
    assert X.shape[0] == len(tiny_dataset)
    # 6 properties * 5 aggregates + (n_elements, entropy, delta_r)
    assert X.shape[1] == 6 * 5 + 3


def test_featurizer_rejects_unknown_element():
    with pytest.raises(ValueError):
        CompositionFeaturizer(element_columns=["Fe", "Xx"])


@pytest.fixture(scope="module")
def fitted_model(tiny_dataset):
    df = tiny_dataset
    elements = ["Fe", "Ni", "Cr", "Mo"]
    targets = ["hardness_hv", "tensile_mpa"]
    ds = Dataset(
        compositions=df[elements],
        properties=df[targets],
        groups=df["heat_id"],
    )
    feat = CompositionFeaturizer(element_columns=elements)
    fm = ForwardModel(featurizer=feat, targets=targets, n_cv_splits=3)
    fm.fit(ds, n_trials=3, verbose=False)
    return fm, ds


def test_forward_model_predict_contract(fitted_model):
    fm, ds = fitted_model
    preds = fm.predict(ds.compositions.head(5))
    for t in fm.targets:
        assert f"{t}_mean" in preds.columns
        assert f"{t}_std" in preds.columns
        assert (preds[f"{t}_std"] >= 0).all()


def test_conformal_interval_roundtrip(fitted_model):
    fm, ds = fitted_model
    cal = ConformalCalibrator(alpha=0.2).calibrate(fm, ds)
    preds = fm.predict(ds.compositions.head(5))
    out = cal.intervals(preds)
    for t in fm.targets:
        assert f"{t}_lo" in out.columns and f"{t}_hi" in out.columns
        assert (out[f"{t}_hi"] >= out[f"{t}_lo"]).all()


def test_domain_of_applicability_score_range(fitted_model):
    fm, ds = fitted_model
    doa = DomainOfApplicability().fit(fm, ds)
    X = fm.featurizer.transform(ds.compositions.head(5))
    first = next(iter(fm.models_.values()))
    Xs = first.preproc.transform(X[first.feature_names])
    scores = doa.score(Xs)
    assert scores.shape == (5,)
    assert (scores >= 0).all()


def test_feasibility_checker_runs(fitted_model):
    fm, ds = fitted_model
    checker = default_checker(["Fe", "Ni", "Cr", "Mo"])
    comp = pd.Series({"Fe": 0.6, "Ni": 0.2, "Cr": 0.15, "Mo": 0.05})
    r = checker.check(comp)
    assert isinstance(r.feasible, bool)
    assert "composition_sum=1" in r.scores


def test_hume_rothery_delta_orders_by_mismatch():
    # δ should be much higher for size-mismatched mixtures (H+Hf) than for
    # well-matched 3d-metal mixtures (Fe/Ni/Cr/Mo).
    c_soft = hume_rothery_size_mismatch(threshold_pct=20.0)
    comp_similar = pd.Series({"Fe": 0.25, "Ni": 0.25, "Cr": 0.25, "Mo": 0.25})
    comp_mismatch = pd.Series({"H": 0.5, "Hf": 0.5})
    delta_sim = c_soft.evaluate(comp_similar) + 20.0   # back out raw δ
    delta_mis = c_soft.evaluate(comp_mismatch) + 20.0
    assert delta_mis > delta_sim
    # At threshold=20% even H+Hf shouldn't trigger; at threshold=2% Fe-Ni-Cr-Mo does.
    assert c_soft.evaluate(comp_similar) < 0
    c_tight = hume_rothery_size_mismatch(threshold_pct=2.0)
    assert c_tight.evaluate(comp_similar) > 0


def test_inverse_designer_smoke(fitted_model):
    pytest.importorskip("pymoo", reason="pymoo not installed")
    from core.alloyforge.inverse_design import DesignSpec, InverseDesigner

    fm, _ = fitted_model
    spec = DesignSpec(
        objectives=[("hardness_hv", "max")],
        element_bounds={
            "Fe": (0.5, 0.9),
            "Ni": (0.0, 0.3),
            "Cr": (0.0, 0.3),
            "Mo": (0.0, 0.1),
        },
        risk_lambda=0.5,
        feasibility=default_checker(["Fe", "Ni", "Cr", "Mo"]),
    )
    designer = InverseDesigner(model=fm, spec=spec,
                               element_columns=["Fe", "Ni", "Cr", "Mo"])
    front = designer.run_nsga2(pop_size=12, n_gen=5, seed=0)
    assert len(front) > 0
    assert "agg_score" in front.columns
    for t in fm.targets:
        assert f"{t}_mean" in front.columns


def test_explainer_local_smoke(fitted_model):
    pytest.importorskip("shap", reason="shap not installed")
    from core.alloyforge.explainability import Explainer

    fm, ds = fitted_model
    expl = Explainer(model=fm)
    q = ds.compositions.head(2)
    shap_df = expl.explain(q, target="hardness_hv", background_df=ds.compositions)
    assert set(shap_df.columns) == {"sample_id", "feature", "value", "shap"}
    assert shap_df["sample_id"].nunique() == 2


def test_active_learner_uncertainty_pick(fitted_model):
    from core.alloyforge.active_learning import ActiveLearner

    fm, ds = fitted_model
    learner = ActiveLearner(model=fm)
    picks = learner.sample_uncertainty(
        candidate_pool=ds.compositions.head(20),
        element_columns=["Fe", "Ni", "Cr", "Mo"],
        batch_size=3,
    )
    assert len(picks) == 3
    assert "acq_score" in picks.columns


def test_llm_assistant_offline_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = LLMAssistant()
    assert a.available is False
    out = a.interpret_prediction(
        composition={"Fe": 0.7, "Ni": 0.2, "Cr": 0.1},
        prediction={"hardness_hv_mean": 250.0, "hardness_hv_std": 12.0},
        shap_top=[{"feature": "Mo_mean", "value": 0.05, "shap": 0.4}],
        extrapolation_score=1.2,
    )
    assert "offline" in out.lower()
    assert "DoA" in out or "extrapolation" in out.lower()
