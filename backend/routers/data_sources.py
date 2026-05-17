"""FastAPI router for data sources, ingestion, and external APIs.

Mounted under ``/api/v1/data`` by ``backend/main.py``.

Endpoints
---------

* ``GET  /reference-alloys`` — return the 38-alloy curated table.
* ``GET  /reference-alloys/{name}`` — single alloy by case-insensitive name.
* ``POST /ingest`` — accept rows + element/target column hints, run unit
  inference + composition normalisation + outlier flagging + dedup,
  return a clean DataFrame as JSON.
* ``GET  /external/status`` — which external providers are currently usable.
* ``GET  /external/openalex`` — OpenAlex paper search.
* ``GET  /external/arxiv`` — arXiv paper search.
* ``GET  /external/crossref`` — CrossRef paper search.
* ``GET  /external/materials-project`` — MP material summary search.
* ``POST /llm-extract`` — run the Claude table extractor on a chunk of
  paper text, return the structured rows.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.alloyforge.data_ingestion import (
    IngestSummary,
    merge_datasets,
)
from core.alloyforge.external_data import (
    materials_project_summary,
    provider_status,
    search_arxiv,
    search_crossref,
    search_openalex,
)
from core.alloyforge.llm_table_extractor import extract_alloy_table
from core.alloyforge.reference_data import (
    PROPERTY_COLUMNS,
    find_alloy,
    reference_dataset,
    reference_elements,
    reference_families,
)

log = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Reference DB
# ---------------------------------------------------------------------------

@router.get("/reference-alloys")
def get_reference_alloys(
    family: Optional[str] = Query(None, description="Filter by family substring"),
    drop_missing: bool = Query(False,
                                description="Drop rows where any property is null"),
) -> Dict[str, Any]:
    df = reference_dataset(drop_missing_targets=drop_missing)
    if family:
        df = df[df["family"].str.contains(family, case=False)]
    return {
        "n_alloys": int(len(df)),
        "elements": reference_elements(),
        "families": reference_families(),
        "property_columns": PROPERTY_COLUMNS,
        "rows": df.to_dict(orient="records"),
    }


@router.get("/reference-alloys/{name}")
def get_one_reference_alloy(name: str) -> Dict[str, Any]:
    a = find_alloy(name)
    if a is None:
        raise HTTPException(404, f"alloy '{name}' not found in reference DB")
    return {
        "name": a.name,
        "family": a.family,
        "composition_weight_pct": a.composition_wt,
        "composition_atomic_frac": a.as_atomic(),
        "properties": {
            "yield_mpa": a.yield_mpa, "tensile_mpa": a.tensile_mpa,
            "elong_pct": a.elong_pct, "hardness_hv": a.hardness_hv,
            "density_gcc": a.density_gcc, "youngs_gpa": a.youngs_gpa,
            "melting_k": a.melting_k,
        },
        "notes": a.notes,
        "references": a.references,
    }


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    sources: Dict[str, List[Dict[str, Any]]] = Field(
        ..., description="Map source-name → list of row dicts"
    )
    element_columns: List[str]
    target_columns: List[str]
    composition_basis: str = Field(
        "auto",
        description='"auto" / "atomic_pct" / "atomic_frac" / "weight_pct"',
    )
    dedup: bool = True


@router.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    if not req.sources:
        raise HTTPException(400, "no sources supplied")
    dfs = {name: pd.DataFrame(rows) for name, rows in req.sources.items()}
    try:
        merged, summary = merge_datasets(
            sources=dfs,
            element_columns=req.element_columns,
            target_columns=req.target_columns,
            composition_basis=req.composition_basis,
            dedup=req.dedup,
        )
    except Exception as exc:
        raise HTTPException(400, f"merge failed: {exc}") from exc
    return {
        "summary": {
            "n_rows_in": summary.n_rows_in,
            "n_rows_out": summary.n_rows_out,
            "duplicated_dropped": summary.duplicated_dropped,
            "outliers_flagged": summary.outliers_flagged,
            "columns_normalised": summary.columns_normalised,
            "notes": summary.notes,
        },
        "rows": merged.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------

@router.get("/external/status")
def external_status() -> Dict[str, Any]:
    return provider_status()


@router.get("/external/openalex")
def openalex(
    q: str = Query(..., min_length=2),
    per_page: int = Query(25, ge=1, le=200),
    open_access_only: bool = Query(True),
) -> Dict[str, Any]:
    df = search_openalex(q, per_page=per_page, open_access_only=open_access_only)
    return {"n": int(len(df)), "rows": df.to_dict(orient="records")}


@router.get("/external/arxiv")
def arxiv(
    q: str = Query(..., min_length=2),
    max_results: int = Query(25, ge=1, le=200),
) -> Dict[str, Any]:
    df = search_arxiv(q, max_results=max_results)
    return {"n": int(len(df)), "rows": df.to_dict(orient="records")}


@router.get("/external/crossref")
def crossref(
    q: str = Query(..., min_length=2),
    rows: int = Query(25, ge=1, le=100),
) -> Dict[str, Any]:
    df = search_crossref(q, rows=rows)
    return {"n": int(len(df)), "rows": df.to_dict(orient="records")}


@router.get("/external/materials-project")
def mp_summary(
    elements: Optional[str] = Query(
        None, description="Comma-separated element symbols (e.g. 'Fe,Ni,Cr')",
    ),
    formula: Optional[str] = Query(None),
    page_size: int = Query(25, ge=1, le=100),
) -> Dict[str, Any]:
    el_list = [e.strip() for e in elements.split(",")] if elements else None
    df = materials_project_summary(
        elements=el_list, formula=formula, page_size=page_size,
    )
    return {"n": int(len(df)), "rows": df.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# LLM table extractor
# ---------------------------------------------------------------------------

class LLMExtractRequest(BaseModel):
    text: str = Field(..., description="Raw paper text to extract from")
    element_columns: Optional[List[str]] = None
    property_columns: Optional[List[str]] = None
    use_llm: bool = Field(True, description="If False, use heuristic only")
    model: Optional[str] = None


@router.post("/llm-extract")
def llm_extract(req: LLMExtractRequest) -> Dict[str, Any]:
    df, report = extract_alloy_table(
        text=req.text,
        element_columns=req.element_columns,
        property_columns=req.property_columns,
        model=req.model,
        use_llm=req.use_llm,
    )
    return {
        "summary": report.to_dict(),
        "rows": df.to_dict(orient="records"),
    }
