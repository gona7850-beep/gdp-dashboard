"""Feature engineering router."""

from fastapi import APIRouter
from pydantic import BaseModel

from core.physics import PHYSICS_FEATURE_FNS

router = APIRouter()


class CompositionRequest(BaseModel):
    composition: dict[str, float]
    features: list[str] | None = None


@router.get("/list")
def list_features() -> list[str]:
    return list(PHYSICS_FEATURE_FNS)


@router.post("/compute")
def compute(req: CompositionRequest) -> dict:
    feats = req.features or list(PHYSICS_FEATURE_FNS)
    out = {}
    for name in feats:
        if name not in PHYSICS_FEATURE_FNS:
            continue
        try:
            out[name] = float(PHYSICS_FEATURE_FNS[name](req.composition))
        except Exception as e:
            out[name] = f"error: {e}"
    return {"composition": req.composition, "features": out}
