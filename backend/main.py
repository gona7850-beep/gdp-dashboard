"""FastAPI entry — exposes the composition + AlloyForge modules as HTTP services.

Run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
OpenAPI docs: http://localhost:8000/docs
Web UI:      http://localhost:8000/

Legacy routers (data, features, train, shap, mobo, literature) target the
older Nb-Si scaffolding whose ``core/db.py`` is corrupted on disk and whose
companion modules (``core/features.py``, ``core/models.py``, ``core/physics.py``,
``core/shap_analysis.py``, ``core/mobo.py``, ``core/literature.py``) are
missing entirely. We deliberately do **not** import them here so the rest of
the platform boots cleanly. To revive that path, restore those modules and
re-enable the imports below.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.routers import alloyforge, composition, data_sources

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(
    title="Composition Design Platform",
    version="0.2.0",
    description=(
        "ML-driven composition / property prediction, inverse design, "
        "validation, and AI-assisted explanation. Lite path = "
        "/api/v1/composition (RF / Dirichlet MC). Advanced path = "
        "/api/v1/alloyforge (XGB + GP + Optuna + NSGA-II + SHAP + AL)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(composition.router, prefix="/api/v1/composition", tags=["composition"])
app.include_router(alloyforge.router, prefix="/api/v1/alloyforge", tags=["alloyforge"])
app.include_router(data_sources.router, prefix="/api/v1/data", tags=["data"])


@app.get("/api", tags=["meta"])
def api_root() -> dict:
    return {
        "name": "Composition Design Platform API",
        "version": "0.2.0",
        "endpoints": {
            "composition_lite": "/api/v1/composition",
            "alloyforge_advanced": "/api/v1/alloyforge",
            "openapi_docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static web UI — served at "/" so the platform is usable from a browser
# without a separate dev server. The Streamlit pages remain the richer
# workbench (run with: streamlit run app/streamlit_app.py).
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        index_path = WEB_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"error": "web/index.html missing"}, status_code=404)
else:
    @app.get("/", include_in_schema=False)
    def index_no_web() -> JSONResponse:
        return JSONResponse(
            {
                "name": "Composition Design Platform API",
                "hint": "Web UI not built. See /api for endpoints, /docs for OpenAPI.",
            }
        )
