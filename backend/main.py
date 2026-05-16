"""FastAPI entry — exposes core/* modules as HTTP services.

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
OpenAPI docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import composition, data, features, literature, mobo, shap, train

app = FastAPI(
    title="Nb-Si AM Alloy Design Platform",
    version="0.1.0",
    description="Physics-informed ML + MOBO + XAI services for Nb-Si AM research.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(features.router, prefix="/api/v1/features", tags=["features"])
app.include_router(train.router, prefix="/api/v1/train", tags=["train"])
app.include_router(shap.router, prefix="/api/v1/shap", tags=["shap"])
app.include_router(mobo.router, prefix="/api/v1/mobo", tags=["mobo"])
app.include_router(literature.router, prefix="/api/v1/lit", tags=["literature"])
app.include_router(composition.router, prefix="/api/v1/composition", tags=["composition"])


@app.get("/")
def root() -> dict:
    return {
        "name": "Nb-Si AM Platform API",
        "version": "0.1.0",
        "db_path": os.environ.get("ALLOY_DB_PATH", str(Path("data/alloy.db").resolve())),
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
