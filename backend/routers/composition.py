"""FastAPI router exposing the composition design platform over HTTP.

Endpoints (mounted under ``/api/v1/composition`` by ``backend/main.py``):

* ``POST /train``         — train a model from CSV path or inline JSON rows
* ``POST /predict``       — predict properties for a single composition
* ``POST /design``        — inverse design from target properties
* ``POST /verify``        — alias of /predict, returns the same payload
* ``POST /analyse``       — predicted + relative-error feasibility report
* ``POST /claude/parse``  — convert a free-text request to target dict
* ``POST /claude/explain``— LLM-written rationale for design candidates
* ``GET  /status``        — whether a model is currently loaded
* ``GET  /demo-dataset``  — return a synthetic dataset (JSON rows)

The router keeps a single in-memory ``PropertyPredictor`` per process. This
is fine for the research workbench use case; production deployments should
swap in a persistence layer (joblib disk cache or a small registry).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.composition_platform import (
    AVAILABLE_ESTIMATORS,
    CompositionDesigner,
    DesignConstraints,
    PropertyPredictor,
    load_dataset,
)
from core.llm_designer import LLMDesigner
from core.synthetic_alloy_data import generate_synthetic_dataset

router = APIRouter()

# Process-wide singletons; protected by GIL for these simple writes.
_predictor: PropertyPredictor | None = None
_designer: CompositionDesigner | None = None
_llm: LLMDesigner | None = None


def _ensure_predictor() -> PropertyPredictor:
    if _predictor is None:
        raise HTTPException(400, "Model not trained yet. Call /train first.")
    return _predictor


def _ensure_designer() -> CompositionDesigner:
    if _designer is None:
        raise HTTPException(400, "Model not trained yet. Call /train first.")
    return _designer


def _ensure_llm() -> LLMDesigner:
    global _llm
    if _llm is None:
        _llm = LLMDesigner()
    return _llm


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    dataset_path: str | None = None
    rows: list[dict[str, Any]] | None = None
    feature_columns: list[str] | None = None
    property_columns: list[str] | None = None
    estimator: str = Field(default="rf", description=f"One of {AVAILABLE_ESTIMATORS}")
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42


class PredictRequest(BaseModel):
    composition: dict[str, float]


class DesignRequest(BaseModel):
    target_properties: dict[str, float]
    weights: dict[str, float] | None = None
    num_candidates: int = 5000
    top_k: int = 5
    strategy: str = "dirichlet"
    min_fraction: dict[str, float] | None = None
    max_fraction: dict[str, float] | None = None
    fixed: dict[str, float] | None = None
    ga_generations: int = 5
    ga_mutation: float = 0.05
    random_state: int | None = None


class AnalyseRequest(BaseModel):
    composition: dict[str, float]
    target_properties: dict[str, float] | None = None
    tolerance: float = 0.1


class ClaudeParseRequest(BaseModel):
    user_request: str


class ClaudeExplainRequest(BaseModel):
    target_properties: dict[str, float]
    candidates: list[dict[str, Any]]


class DemoRequest(BaseModel):
    n_samples: int = 200
    noise_scale: float = 0.05
    random_state: int = 42


# ---------------------------------------------------------------------------
# Status / demo
# ---------------------------------------------------------------------------

@router.get("/status")
def status() -> dict[str, Any]:
    llm = _ensure_llm()
    return {
        "model_trained": _predictor is not None,
        "feature_columns": _predictor.feature_columns if _predictor else [],
        "property_columns": _predictor.property_columns if _predictor else [],
        "estimator": _predictor.estimator_name if _predictor else None,
        "report": _predictor.report.to_dict() if (_predictor and _predictor.report) else None,
        "llm_available": llm.available,
    }


@router.post("/demo-dataset")
def demo_dataset(req: DemoRequest) -> dict[str, Any]:
    df = generate_synthetic_dataset(
        n_samples=req.n_samples,
        noise_scale=req.noise_scale,
        random_state=req.random_state,
    )
    return {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Train / predict / design
# ---------------------------------------------------------------------------

@router.post("/train")
def train(req: TrainRequest) -> dict[str, Any]:
    global _predictor, _designer
    if req.dataset_path:
        if not Path(req.dataset_path).exists():
            raise HTTPException(404, f"dataset_path not found: {req.dataset_path}")
        df = load_dataset(req.dataset_path)
    elif req.rows:
        df = pd.DataFrame(req.rows)
    else:
        raise HTTPException(400, "Provide either dataset_path or rows.")
    try:
        predictor = PropertyPredictor(
            estimator=req.estimator, random_state=req.random_state
        )
        report = predictor.train(
            df,
            feature_columns=req.feature_columns,
            property_columns=req.property_columns,
            test_size=req.test_size,
            cv_folds=req.cv_folds,
        )
    except Exception as exc:
        raise HTTPException(400, f"training failed: {exc}") from exc
    _predictor = predictor
    _designer = CompositionDesigner(predictor)
    return {"report": report.to_dict()}


@router.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    pred = _ensure_predictor().predict(req.composition)
    return pred.to_dict()


@router.post("/verify")
def verify(req: PredictRequest) -> dict[str, Any]:
    return predict(req)


@router.post("/design")
def design(req: DesignRequest) -> dict[str, Any]:
    designer = _ensure_designer()
    constraints = None
    if any([req.min_fraction, req.max_fraction, req.fixed]):
        constraints = DesignConstraints(
            min_fraction=req.min_fraction or {},
            max_fraction=req.max_fraction or {},
            fixed=req.fixed or {},
        )
    try:
        cands = designer.design_inverse(
            target_properties=req.target_properties,
            weights=req.weights,
            num_candidates=req.num_candidates,
            top_k=req.top_k,
            constraints=constraints,
            strategy=req.strategy,
            ga_generations=req.ga_generations,
            ga_mutation=req.ga_mutation,
            random_state=req.random_state,
        )
    except Exception as exc:
        raise HTTPException(400, f"design failed: {exc}") from exc
    return {"candidates": [c.to_dict() for c in cands]}


@router.post("/analyse")
def analyse(req: AnalyseRequest) -> dict[str, Any]:
    designer = _ensure_designer()
    try:
        return designer.analyse_feasibility(
            composition=req.composition,
            target_properties=req.target_properties,
            tolerance=req.tolerance,
        )
    except Exception as exc:
        raise HTTPException(400, f"analyse failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Claude (LLM) endpoints
# ---------------------------------------------------------------------------

@router.post("/claude/parse")
def claude_parse(req: ClaudeParseRequest) -> dict[str, Any]:
    predictor = _ensure_predictor()
    llm = _ensure_llm()
    target, resp = llm.parse_target(req.user_request, predictor.property_columns)
    return {
        "target_properties": target,
        "used_llm": resp.used_llm,
        "model": resp.model,
        "raw_text": resp.text,
    }


@router.post("/claude/explain")
def claude_explain(req: ClaudeExplainRequest) -> dict[str, Any]:
    predictor = _ensure_predictor()
    llm = _ensure_llm()
    resp = llm.explain_candidates(
        target=req.target_properties,
        candidates=req.candidates,
        model_r2=predictor.report.val_r2 if predictor.report else None,
    )
    return {"text": resp.text, "used_llm": resp.used_llm, "model": resp.model}
