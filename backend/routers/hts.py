"""FastAPI router for high-throughput compound screening.

Mounted at ``/api/v1/hts`` by ``backend/main.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.alloyforge.hts_screening import (
    HOSTS,
    NB_HOST_COMPOUNDS,
    ScoreWeights,
    host_plus_precipitate_composition,
    rank_compounds,
    score_compound,
)
from core.alloyforge.oqmd_client import (
    query_oqmd,
    to_known_compounds,
)

router = APIRouter()


@router.get("/hosts")
def list_hosts() -> Dict[str, Any]:
    """Available host-matrix reference phases."""
    return {
        name: {
            "structure": h.structure,
            "lattice_a": h.lattice_a,
            "volume_per_atom": h.volume_per_atom,
        }
        for name, h in HOSTS.items()
    }


@router.get("/compounds")
def list_compounds() -> Dict[str, Any]:
    """The bundled Nb-host compound database."""
    out = []
    for c in NB_HOST_COMPOUNDS:
        out.append({
            "formula": c.formula,
            "elements": list(c.elements),
            "space_group": c.space_group,
            "lattice_a": c.lattice_a,
            "lattice_c": c.lattice_c,
            "volume_per_atom": c.volume_per_atom,
            "delta_h_per_atom_ev": c.formation_energy_per_atom_ev,
            "has_direct_tie_line_with": list(c.has_direct_tie_line_with),
            "notes": c.notes,
            "source": c.source,
        })
    return {"n": len(out), "compounds": out}


class RankRequest(BaseModel):
    host: str = Field("Nb", description="Host element symbol")
    weight_tie_line: float = 1.0
    weight_stability: float = 1.0
    weight_coherency: float = 1.0
    required_elements: Optional[List[str]] = None
    forbidden_elements: Optional[List[str]] = None
    min_tie_line_score: float = 0.0
    top_k: Optional[int] = None


@router.post("/rank")
def rank(req: RankRequest) -> Dict[str, Any]:
    """Rank the bundled compound DB for a given host + scoring weights."""
    if req.host not in HOSTS:
        raise HTTPException(404, f"Unknown host {req.host!r}; "
                                  f"available: {list(HOSTS)}")
    weights = ScoreWeights(
        tie_line=req.weight_tie_line,
        stability=req.weight_stability,
        coherency=req.weight_coherency,
    )
    df = rank_compounds(
        host=req.host,
        weights=weights,
        required_elements=req.required_elements,
        forbidden_elements=req.forbidden_elements,
        min_tie_line_score=req.min_tie_line_score,
        top_k=req.top_k,
    )
    return {"n": int(len(df)), "ranking": df.to_dict(orient="records")}


class CompoundMixRequest(BaseModel):
    host: str = "Nb"
    compound_formula: str
    precipitate_atomic_fraction: float = Field(0.10, ge=0.0, le=1.0)


@router.post("/compound-mix")
def compound_mix(req: CompoundMixRequest) -> Dict[str, Any]:
    """Compose a host-matrix + precipitate composition for the forward model.

    Useful when you want to ask the ML predictor: "What does the host
    alloy look like with X at% of compound Y as a precipitate?"
    """
    if req.host not in HOSTS:
        raise HTTPException(404, f"Unknown host {req.host!r}")
    host = HOSTS[req.host]
    target = next((c for c in NB_HOST_COMPOUNDS
                    if c.formula == req.compound_formula), None)
    if target is None:
        raise HTTPException(404, f"Compound {req.compound_formula!r} "
                                  f"not in bundled DB")
    comp = host_plus_precipitate_composition(
        host, target, req.precipitate_atomic_fraction,
    )
    return {"composition_atomic_fraction": comp}


class OQMDQueryRequest(BaseModel):
    elements: Optional[List[str]] = None
    formula: Optional[str] = None
    n_atoms_max: Optional[int] = None
    stability_max: Optional[float] = None
    page_size: int = Field(25, ge=1, le=100)


@router.post("/oqmd-search")
def oqmd_search(req: OQMDQueryRequest) -> Dict[str, Any]:
    df = query_oqmd(
        elements=req.elements,
        formula=req.formula,
        n_atoms_max=req.n_atoms_max,
        stability_max=req.stability_max,
        page_size=req.page_size,
    )
    return {"n": int(len(df)), "rows": df.to_dict(orient="records")}


@router.post("/oqmd-rank")
def oqmd_rank(req: OQMDQueryRequest) -> Dict[str, Any]:
    """Query OQMD then rank the returned compounds with default weights."""
    df = query_oqmd(
        elements=req.elements, formula=req.formula,
        n_atoms_max=req.n_atoms_max, stability_max=req.stability_max,
        page_size=req.page_size,
    )
    if df.empty:
        return {"n": 0, "ranking": []}
    compounds = to_known_compounds(df)
    out = rank_compounds(host="Nb", compounds=compounds)
    return {"n": int(len(out)), "ranking": out.to_dict(orient="records")}
