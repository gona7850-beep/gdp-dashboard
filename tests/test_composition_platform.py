"""Sanity tests for core.composition_platform and friends.

The synthetic dataset is small but structured enough that an RF model
should pass R^2 > 0 on at least one property. The tests avoid asserting
on absolute numeric thresholds where possible (CI machines vary) and
instead focus on shape, invariants, and error handling.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.composition_platform import (
    AVAILABLE_ESTIMATORS,
    CompositionDesigner,
    DesignConstraints,
    PropertyPredictor,
)
from core.llm_designer import LLMDesigner, _extract_first_json_object
from core.synthetic_alloy_data import (
    default_elements,
    default_properties,
    generate_synthetic_dataset,
    target_from_quantile,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def test_synthetic_dataset_shape_and_simplex():
    df = generate_synthetic_dataset(n_samples=150, random_state=0)
    els = default_elements()
    props = default_properties()
    for c in els + props:
        assert c in df.columns
    sums = df[els].sum(axis=1).to_numpy()
    assert np.allclose(sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_predictor():
    df = generate_synthetic_dataset(n_samples=300, random_state=42)
    p = PropertyPredictor(estimator="rf", random_state=42)
    report = p.train(df, cv_folds=3)
    assert report.estimator_name == "rf"
    assert set(p.property_columns) == set(default_properties())
    assert set(p.feature_columns) == set(default_elements())
    return p


def test_predictor_validation_columns(trained_predictor):
    rep = trained_predictor.report
    assert set(rep.val_r2.keys()) == set(default_properties())
    assert all(np.isfinite(v) for v in rep.val_r2.values())


def test_predictor_predict_shape(trained_predictor):
    comp = {e: 1.0 / len(trained_predictor.feature_columns)
            for e in trained_predictor.feature_columns}
    result = trained_predictor.predict(comp)
    assert set(result.properties.keys()) == set(trained_predictor.property_columns)
    for v in result.properties.values():
        assert isinstance(v, float)
    # RF should yield per-property uncertainty
    assert result.uncertainty is not None
    assert set(result.uncertainty.keys()) == set(trained_predictor.property_columns)


def test_predictor_missing_element_raises(trained_predictor):
    bad = {e: 0.1 for e in trained_predictor.feature_columns[:-1]}
    with pytest.raises(ValueError):
        trained_predictor.predict(bad)


def test_predictor_persistence_roundtrip(tmp_path, trained_predictor):
    path = tmp_path / "model.joblib"
    trained_predictor.save(path)
    loaded = PropertyPredictor.load(path)
    comp = {e: 1.0 / len(loaded.feature_columns) for e in loaded.feature_columns}
    a = trained_predictor.predict(comp).properties
    b = loaded.predict(comp).properties
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k])


@pytest.mark.parametrize("est", AVAILABLE_ESTIMATORS)
def test_each_estimator_trains(est):
    df = generate_synthetic_dataset(n_samples=120, random_state=1)
    p = PropertyPredictor(estimator=est, random_state=0)
    rep = p.train(df, cv_folds=2)
    assert rep.estimator_name == est


# ---------------------------------------------------------------------------
# Designer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def designer(trained_predictor):
    return CompositionDesigner(trained_predictor)


def test_inverse_design_returns_simplex_compositions(designer):
    df = generate_synthetic_dataset(n_samples=200, random_state=7)
    target = target_from_quantile(df, designer.predictor.property_columns, 0.8)
    cands = designer.design_inverse(
        target, num_candidates=500, top_k=3, random_state=0
    )
    assert len(cands) == 3
    for c in cands:
        s = sum(c.composition.values())
        assert s == pytest.approx(1.0, abs=1e-6)
        for v in c.composition.values():
            assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9


def test_inverse_design_respects_fixed_constraints(designer):
    target = {p: 0.0 for p in designer.predictor.property_columns}
    constraints = DesignConstraints(fixed={"Fe": 0.30, "Cr": 0.10})
    cands = designer.design_inverse(
        target, num_candidates=300, top_k=3,
        constraints=constraints, random_state=1,
    )
    for c in cands:
        assert c.composition["Fe"] == pytest.approx(0.30, abs=1e-6)
        assert c.composition["Cr"] == pytest.approx(0.10, abs=1e-6)


def test_inverse_design_respects_min_max(designer):
    target = {p: 0.0 for p in designer.predictor.property_columns}
    cons = DesignConstraints(min_fraction={"Ni": 0.15}, max_fraction={"Mo": 0.05})
    cands = designer.design_inverse(
        target, num_candidates=400, top_k=5, constraints=cons, random_state=2,
    )
    for c in cands:
        # Renormalisation can perturb the bound slightly; allow a small slack.
        assert c.composition["Ni"] >= 0.15 - 1e-3
        assert c.composition["Mo"] <= 0.05 + 1e-3


def test_ga_strategy_improves_or_matches_baseline(designer):
    df = generate_synthetic_dataset(n_samples=200, random_state=3)
    target = target_from_quantile(df, designer.predictor.property_columns, 0.7)
    dirichlet = designer.design_inverse(
        target, num_candidates=300, top_k=1, strategy="dirichlet", random_state=4,
    )
    ga = designer.design_inverse(
        target, num_candidates=300, top_k=1, strategy="ga",
        ga_generations=3, random_state=4,
    )
    # GA seeds itself from the same Dirichlet pool, so its best should
    # never be strictly worse.
    assert ga[0].score <= dirichlet[0].score + 1e-9


def test_analyse_feasibility_flags_target_misses(designer):
    comp = {e: 1.0 / len(designer.predictor.feature_columns)
            for e in designer.predictor.feature_columns}
    pred = designer.verify_composition(comp).properties
    # Build a target that is deliberately off
    target = {p: v * 10 + 1000 for p, v in pred.items()}
    out = designer.analyse_feasibility(comp, target, tolerance=0.05)
    assert "relative_errors" in out
    assert "meets_target" in out
    assert out["overall_feasible"] is False


def test_analyse_feasibility_no_target_returns_predictions_only(designer):
    comp = {e: 1.0 / len(designer.predictor.feature_columns)
            for e in designer.predictor.feature_columns}
    out = designer.analyse_feasibility(comp, target_properties=None)
    assert "predicted" in out
    assert "relative_errors" not in out


def test_design_rejects_missing_property(designer):
    with pytest.raises(ValueError):
        designer.design_inverse({"not_a_property": 1.0}, num_candidates=50)


# ---------------------------------------------------------------------------
# LLM wrapper (heuristic fallback path only — no API key in CI)
# ---------------------------------------------------------------------------

def test_llm_wrapper_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = LLMDesigner()
    assert llm.available is False
    target, resp = llm.parse_target(
        "yield strength of 650 and hardness around 200",
        ["yield_strength", "hardness", "elongation"],
    )
    assert resp.used_llm is False
    assert target.get("yield_strength") == pytest.approx(650.0)
    assert target.get("hardness") == pytest.approx(200.0)


def test_llm_explain_candidates_heuristic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = LLMDesigner()
    cands = [
        {"composition": {"Fe": 0.5, "Ni": 0.5}, "predicted": {"a": 1.0},
         "score": 0.1, "rel_errors": {"a": 0.05}},
        {"composition": {"Fe": 0.4, "Ni": 0.6}, "predicted": {"a": 1.2},
         "score": 0.05, "rel_errors": {"a": 0.02}},
    ]
    resp = llm.explain_candidates(target={"a": 1.1}, candidates=cands)
    assert resp.used_llm is False
    assert "RECOMMENDED: candidate #2" in resp.text


def test_extract_first_json_object_handles_fenced_output():
    text = "Here you go:\n```json\n{\"yield_strength\": 650.0}\n```\nDone."
    obj = _extract_first_json_object(text)
    assert obj == {"yield_strength": 650.0}


def test_extract_first_json_object_handles_inline_output():
    text = 'Sure: {"yield_strength": 650, "hardness": 200}'
    obj = _extract_first_json_object(text)
    assert obj == {"yield_strength": 650, "hardness": 200}


def test_extract_first_json_object_returns_none_for_garbage():
    assert _extract_first_json_object("no json here, just prose") is None
