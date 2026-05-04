"""Data router — ingest CSV + materialize training set."""

from io import StringIO

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from core.db import (
    DEFAULT_DB_PATH,
    ingest_dataframe,
    materialize_training_set,
    query_property_breakdown,
    query_stats,
)

router = APIRouter()


@router.get("/stats")
def stats() -> dict:
    return query_stats(DEFAULT_DB_PATH)


@router.get("/breakdown")
def breakdown() -> list[dict]:
    df = query_property_breakdown(DEFAULT_DB_PATH)
    return df.to_dict(orient="records")


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    target_property: str = "HV",
    process_method: str = "unknown",
    condition: str = "RT",
    T_test_K: float = 298.0,
) -> dict:
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        df = pd.read_csv(StringIO(raw))
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")
    try:
        summary = ingest_dataframe(
            df, target_property=target_property,
            process_method=process_method, condition=condition,
            T_test_K=float(T_test_K), db_path=DEFAULT_DB_PATH,
        )
    except Exception as e:
        raise HTTPException(400, f"ingest error: {e}")
    return {
        "rows_in": summary.rows_in,
        "alloys_added": summary.alloys_added,
        "processes_added": summary.processes_added,
        "properties_added": summary.properties_added,
    }


@router.get("/training-set")
def training_set(target: str = "HV", condition: str = "RT", include_process: bool = False) -> dict:
    df = materialize_training_set(target, condition, db_path=DEFAULT_DB_PATH, include_process=include_process)
    return {"rows": len(df), "columns": list(df.columns), "data": df.to_dict(orient="records")}
