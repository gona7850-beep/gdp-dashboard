"""SHAP router."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.db import materialize_training_set
from core.features import build_feature_matrix
from core.models import available_models
from core.shap_analysis import explain, global_importance, physics_validation, top_interactions

router = APIRouter()


class ShapRequest(BaseModel):
    target: str = "HV"
    condition: str = "RT"
    model: str = "RandomForest"
    use_physics: bool = True
    use_process: bool = True
    compute_interactions: bool = True


@router.post("/explain")
def shap_explain(req: ShapRequest) -> dict:
    df = materialize_training_set(req.target, req.condition)
    if df.empty or req.target not in df.columns:
        raise HTTPException(404, f"No data for {req.target}/{req.condition}")
    X, y = build_feature_matrix(df, use_physics=req.use_physics,
                                use_process=req.use_process, target_col=req.target)
    available = available_models()
    if req.model not in available:
        raise HTTPException(400, f"Model {req.model} not available")
    est = available[req.model]()
    est.fit(X.values, y.values)
    res = explain(est, X, compute_interactions=req.compute_interactions)
    return {
        "n_samples": int(len(y)),
        "global_importance": global_importance(res).to_dict(orient="records"),
        "top_interactions": top_interactions(res, k=15).to_dict(orient="records"),
        "physics_validation": physics_validation(res, req.target).to_dict(orient="records"),
    }
