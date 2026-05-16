"""FastAPI router exposing the AlloyForge advanced ML stack.

Mounted at ``/api/v1/alloyforge`` by ``backend/main.py``.

Endpoints (compared to ``/api/v1/composition`` which uses a simpler
RF/Dirichlet pipeline):

* ``POST /fit``               — train stacked XGB+GP, Optuna-tuned, group-aware CV
* ``POST /predict``           — predict (μ, σ) + conformal 90% intervals + DoA score
* ``POST /feasibility/check`` — run Hume-Rothery / VEC / VED / element-bounds checks
* ``POST /inverse-design``    — NSGA-II with risk-aware ``μ − λσ`` objective
* ``POST /explain``           — SHAP attributions + LLM-mediated metallurgy interpretation
* ``POST /active-learning``   — uncertainty-batch picks with diversity penalty
* ``GET  /sessions``          — list in-memory sessions
* ``GET  /status``            — capability flags + dependency check

The router holds a per-process in-memory session store keyed by UUID. For
production multi-tenant deployments, swap ``_SESSIONS`` for a Redis/SQLite
backend.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("alloyforge.router")

# Heavy ML imports are deferred so the rest of the FastAPI app can boot even
# if XGBoost / SHAP / pymoo aren't installed yet (useful for first-time setup).
_IMPORT_ERROR: Exception | None = None
try:
    from core.alloyforge import (
        ActiveLearner,
        CompositionFeaturizer,
        ConformalCalibrator,
        Dataset,
        DesignSpec,
        DomainOfApplicability,
        Explainer,
        ForwardModel,
        InverseDesigner,
        LLMAssistant,
        default_checker,
        element_bounds,
        ved_window,
        vec_window,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc

router = APIRouter()
_SESSIONS: dict[str, dict[str, Any]] = {}


def _require_alloyforge() -> None:
    if _IMPORT_ERROR is not None:
        raise HTTPException(
            503,
            f"AlloyForge dependencies missing ({_IMPORT_ERROR}). "
            f"Install with: pip install xgboost optuna pymoo shap",
        )


def _session(sid: str) -> dict[str, Any]:
    if sid not in _SESSIONS:
        raise HTTPException(404, f"Unknown session_id: {sid}")
    return _SESSIONS[sid]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FitRequest(BaseModel):
    element_columns: list[str] = Field(..., description="Composition columns")
    target_columns: list[str] = Field(..., description="Property columns to model")
    process_columns: list[str] | None = None
    group_column: str | None = None
    n_trials: int = Field(25, ge=3, le=200)
    n_cv_splits: int = Field(5, ge=2, le=10)
    data: list[dict[str, Any]] = Field(..., description="Training rows (records)")


class FitResponse(BaseModel):
    session_id: str
    metrics: dict[str, dict[str, float]]
    feature_names: list[str]


class PredictRequest(BaseModel):
    session_id: str
    compositions: list[dict[str, float]]
    process: list[dict[str, float]] | None = None


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]
    intervals: list[dict[str, Any]] | None = None
    doa_scores: list[float] | None = None


class FeasibilityRequest(BaseModel):
    session_id: str
    compositions: list[dict[str, float]]
    process: list[dict[str, float]] | None = None
    bounds: dict[str, tuple[float, float]] | None = None
    vec_window: tuple[float, float] | None = None
    ved_window: tuple[float, float] | None = None


class ObjectiveSpec(BaseModel):
    target: str
    direction: str = Field(..., pattern="^(max|min|target)$")
    target_value: float | None = None


class InverseDesignRequest(BaseModel):
    session_id: str
    objectives: list[ObjectiveSpec]
    element_bounds: dict[str, tuple[float, float]]
    process_bounds: dict[str, tuple[float, float]] | None = None
    pop_size: int = Field(80, ge=16, le=400)
    n_gen: int = Field(60, ge=5, le=300)
    risk_lambda: float = Field(1.0, ge=0.0, le=5.0)
    feasibility_bounds: dict[str, tuple[float, float]] | None = None
    top_k: int = Field(10, ge=1, le=100)


class ExplainRequest(BaseModel):
    session_id: str
    composition: dict[str, float]
    target: str
    process: dict[str, float] | None = None


class ActiveLearningRequest(BaseModel):
    session_id: str
    candidate_pool: list[dict[str, float]]
    batch_size: int = Field(5, ge=1, le=50)
    target_weights: dict[str, float] | None = None
    process: list[dict[str, float]] | None = None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status")
def status() -> dict[str, Any]:
    return {
        "alloyforge_available": _IMPORT_ERROR is None,
        "import_error": str(_IMPORT_ERROR) if _IMPORT_ERROR else None,
        "n_sessions": len(_SESSIONS),
    }


@router.get("/sessions")
def sessions() -> dict[str, Any]:
    return {
        "sessions": [
            {
                "session_id": sid,
                "element_columns": s["element_columns"],
                "target_columns": s["target_columns"],
                "created_at": s["created_at"],
            }
            for sid, s in _SESSIONS.items()
        ]
    }


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------

@router.post("/fit", response_model=FitResponse)
def fit(req: FitRequest) -> FitResponse:
    _require_alloyforge()
    t0 = time.time()
    df = pd.DataFrame(req.data)
    missing = [c for c in req.element_columns + req.target_columns if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    ds = Dataset(
        compositions=df[req.element_columns].copy(),
        properties=df[req.target_columns].copy(),
        process=df[req.process_columns].copy() if req.process_columns else None,
        groups=df[req.group_column].copy() if req.group_column else None,
    )
    feat = CompositionFeaturizer(element_columns=req.element_columns)
    model = ForwardModel(
        featurizer=feat,
        targets=req.target_columns,
        n_cv_splits=req.n_cv_splits,
    )
    try:
        model.fit(ds, n_trials=req.n_trials)
        conformal = ConformalCalibrator(alpha=0.1).calibrate(model, ds)
        doa = DomainOfApplicability().fit(model, ds)
    except Exception as exc:
        raise HTTPException(400, f"fit failed: {exc}") from exc

    sid = str(uuid.uuid4())
    _SESSIONS[sid] = {
        "model": model,
        "featurizer": feat,
        "dataset": ds,
        "conformal": conformal,
        "doa": doa,
        "element_columns": req.element_columns,
        "target_columns": req.target_columns,
        "process_columns": req.process_columns or [],
        "created_at": t0,
    }
    log.info(f"alloyforge: fit session {sid} in {time.time() - t0:.1f}s")
    return FitResponse(
        session_id=sid,
        metrics=model.metrics_,
        feature_names=list(feat.feature_names),
    )


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    _require_alloyforge()
    sess = _session(req.session_id)
    model: ForwardModel = sess["model"]
    el_cols = sess["element_columns"]
    proc_cols = sess["process_columns"]

    comp_df = pd.DataFrame(req.compositions).reindex(columns=el_cols, fill_value=0)
    proc_df = None
    if req.process and proc_cols:
        proc_df = pd.DataFrame(req.process).reindex(columns=proc_cols, fill_value=0)

    preds = model.predict(comp_df, process=proc_df)
    intervals = sess["conformal"].intervals(preds)

    X = sess["featurizer"].transform(comp_df)
    if proc_df is not None:
        X = pd.concat([X.reset_index(drop=True), proc_df.reset_index(drop=True)], axis=1)
    first = next(iter(model.models_.values()))
    X_s = first.preproc.transform(X[first.feature_names])
    doa_scores = sess["doa"].score(X_s).tolist()

    return PredictResponse(
        predictions=preds.to_dict(orient="records"),
        intervals=intervals.to_dict(orient="records"),
        doa_scores=doa_scores,
    )


# ---------------------------------------------------------------------------
# Feasibility / inverse design / explain / active learning
# ---------------------------------------------------------------------------

@router.post("/feasibility/check")
def feasibility(req: FeasibilityRequest) -> dict[str, Any]:
    _require_alloyforge()
    sess = _session(req.session_id)
    el_cols = sess["element_columns"]
    comp_df = pd.DataFrame(req.compositions).reindex(columns=el_cols, fill_value=0)

    checker = default_checker(el_cols, bounds=req.bounds)
    if req.vec_window is not None:
        checker.add(vec_window(req.vec_window[0], req.vec_window[1]))
    if req.ved_window is not None:
        checker.add(ved_window(req.ved_window[0], req.ved_window[1]))

    proc_df = pd.DataFrame(req.process) if req.process else None
    results = []
    for i, (_, row) in enumerate(comp_df.iterrows()):
        proc_row = proc_df.iloc[i] if proc_df is not None else None
        results.append(checker.check(row, proc_row).to_dict())
    return {"results": results}


@router.post("/inverse-design")
def inverse_design(req: InverseDesignRequest) -> dict[str, Any]:
    _require_alloyforge()
    sess = _session(req.session_id)
    model: ForwardModel = sess["model"]
    el_cols = sess["element_columns"]

    objs = [(o.target, o.direction) for o in req.objectives]
    target_vals = {
        o.target: o.target_value
        for o in req.objectives
        if o.direction == "target" and o.target_value is not None
    }
    feas = default_checker(el_cols, bounds=req.feasibility_bounds)
    spec = DesignSpec(
        objectives=objs,
        element_bounds=req.element_bounds,
        target_values=target_vals,
        process_bounds=req.process_bounds or {},
        risk_lambda=req.risk_lambda,
        feasibility=feas,
    )
    designer = InverseDesigner(model=model, spec=spec, element_columns=el_cols)
    try:
        df = designer.run_nsga2(pop_size=req.pop_size, n_gen=req.n_gen).head(req.top_k)
    except Exception as exc:
        raise HTTPException(400, f"inverse design failed: {exc}") from exc
    return {"candidates": df.to_dict(orient="records")}


@router.post("/explain")
def explain(req: ExplainRequest) -> dict[str, Any]:
    _require_alloyforge()
    sess = _session(req.session_id)
    model: ForwardModel = sess["model"]
    el_cols = sess["element_columns"]

    q = pd.DataFrame([req.composition]).reindex(columns=el_cols, fill_value=0)
    bg = sess["dataset"].compositions

    expl = Explainer(model=model)
    try:
        shap_df = expl.explain(q, target=req.target, background_df=bg)
        glob = expl.global_importance(req.target, bg).head(15)
    except Exception as exc:
        raise HTTPException(400, f"explain failed: {exc}") from exc

    sample = shap_df[shap_df["sample_id"] == 0]
    top = sample.reindex(sample.shap.abs().sort_values(ascending=False).index).head(8)

    assistant = LLMAssistant()
    proc_df = pd.DataFrame([req.process]) if req.process and sess["process_columns"] else None
    preds = model.predict(q, process=proc_df).iloc[0].to_dict()
    interpretation = assistant.interpret_prediction(
        composition=req.composition,
        prediction=preds,
        shap_top=top[["feature", "value", "shap"]].to_dict(orient="records"),
    )

    return {
        "shap": (
            shap_df[shap_df["sample_id"] == 0]
            .sort_values("shap", key=lambda x: x.abs(), ascending=False)
            .head(15)
            .to_dict(orient="records")
        ),
        "global_importance": glob.to_dict(orient="records"),
        "interpretation": interpretation,
        "used_llm": assistant.available,
    }


@router.post("/active-learning")
def active_learning(req: ActiveLearningRequest) -> dict[str, Any]:
    _require_alloyforge()
    sess = _session(req.session_id)
    model: ForwardModel = sess["model"]
    el_cols = sess["element_columns"]
    proc_cols = sess["process_columns"]

    pool = pd.DataFrame(req.candidate_pool).reindex(columns=el_cols, fill_value=0)
    if req.process and proc_cols:
        proc_df = pd.DataFrame(req.process).reindex(columns=proc_cols, fill_value=0)
        pool = pd.concat([pool.reset_index(drop=True),
                          proc_df.reset_index(drop=True)], axis=1)

    learner = ActiveLearner(model=model)
    try:
        picks = learner.sample_uncertainty(
            candidate_pool=pool,
            element_columns=el_cols,
            process_columns=proc_cols or None,
            batch_size=req.batch_size,
            target_weights=req.target_weights,
        )
    except Exception as exc:
        raise HTTPException(400, f"active learning failed: {exc}") from exc
    return {"picks": picks.to_dict(orient="records")}
