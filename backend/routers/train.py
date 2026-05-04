"""Training/benchmark router."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.benchmark import benchmark, to_dataframe
from core.db import materialize_training_set
from core.features import build_feature_matrix
from core.models import available_models

router = APIRouter()


class TrainRequest(BaseModel):
    target: str = "HV"
    condition: str = "RT"
    models: list[str] = ["RandomForest"]
    cv: str = "kfold"
    n_splits: int = 5
    n_seeds: int = 3
    use_physics: bool = True
    use_process: bool = True


@router.get("/models")
def list_models() -> list[str]:
    return list(available_models())


@router.post("/run")
def run_train(req: TrainRequest) -> dict:
    df = materialize_training_set(req.target, req.condition)
    if df.empty or req.target not in df.columns:
        raise HTTPException(404, f"No data for target={req.target}, condition={req.condition}")
    X, y = build_feature_matrix(df, use_physics=req.use_physics,
                                use_process=req.use_process, target_col=req.target)
    available = available_models()
    chosen = {k: available[k] for k in req.models if k in available}
    if not chosen:
        raise HTTPException(400, f"No requested models available. Have: {list(available)}")
    results = benchmark(X, y, chosen, cv=req.cv, n_splits=req.n_splits, n_seeds=req.n_seeds)
    return {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "results": to_dataframe(results).to_dict(orient="records"),
    }
