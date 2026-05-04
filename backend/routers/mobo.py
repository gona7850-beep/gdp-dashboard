"""MOBO router."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.db import materialize_training_set
from core.features import detect_element_columns
from core.mobo import BOTORCH_OK, fit_mobo, propose_batch

router = APIRouter()


class MoboRequest(BaseModel):
    obj1: str = "HV"
    obj2: str = "sigma_compressive"
    minimize1: bool = False
    minimize2: bool = False
    condition: str = "RT"
    q: int = 5


@router.get("/backend")
def backend_info() -> dict:
    return {"backend": "botorch" if BOTORCH_OK else "sklearn"}


@router.post("/propose")
def propose(req: MoboRequest) -> dict:
    df1 = materialize_training_set(req.obj1, req.condition)
    df2 = materialize_training_set(req.obj2, req.condition)
    if df1.empty or df2.empty:
        raise HTTPException(404, "Missing data for one of the objectives")
    df = df1.merge(df2[["alloy_id", req.obj2]], on="alloy_id", how="inner")
    elem_cols = detect_element_columns(df)
    if not elem_cols:
        raise HTTPException(400, "No element columns found")
    X = df[elem_cols].fillna(0).astype(float)
    Y = df[[req.obj1, req.obj2]].apply(pd.to_numeric, errors="coerce").dropna()
    X = X.loc[Y.index]
    if len(X) < 5:
        raise HTTPException(400, f"Need ≥5 paired observations, got {len(X)}")
    bounds = np.vstack([np.zeros(X.shape[1]), (X.max(axis=0) * 1.2).clip(lower=1.0).values])
    sg = fit_mobo(X, Y, bounds=bounds, minimize=[req.minimize1, req.minimize2])
    cands = propose_batch(sg, q=req.q)
    return {
        "backend": sg.backend,
        "n_paired": int(len(X)),
        "candidates": [
            {"x": c.x, "y_pred": c.y_pred, "y_std": c.y_std, "acq": c.acq_value}
            for c in cands
        ],
    }
