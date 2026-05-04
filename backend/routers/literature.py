"""Literature router."""

from fastapi import APIRouter, Query

from core.literature import (
    NB_SI_PRESET_WITH,
    NB_SI_PRESET_WITHOUT,
    build_query,
    compound_search,
    unpaywall_lookup,
)

router = APIRouter()


@router.get("/preset")
def preset() -> dict:
    return {"with_words": NB_SI_PRESET_WITH, "without_words": NB_SI_PRESET_WITHOUT}


@router.get("/search")
def search(
    query: str = Query(..., description="Free-text query (use /preset for default Nb-Si bundle)"),
    sources: str = "crossref,openalex",
    rows: int = 25,
) -> dict:
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    papers = compound_search(query, sources=src_list, rows=rows)
    return {
        "query": query,
        "sources": src_list,
        "n_results": len(papers),
        "papers": [
            {
                "doi": p.doi, "title": p.title, "authors": p.authors,
                "year": p.year, "venue": p.venue, "abstract": p.abstract,
                "oa_url": p.oa_url, "citations": p.citations, "source": p.source,
            }
            for p in papers
        ],
    }


@router.get("/preset-search")
def preset_search(rows: int = 25) -> dict:
    q = build_query(NB_SI_PRESET_WITH, NB_SI_PRESET_WITHOUT)
    return search(query=q, sources="crossref,openalex", rows=rows)


@router.get("/unpaywall")
def unpaywall(doi: str) -> dict:
    return {"doi": doi, "oa_url": unpaywall_lookup(doi)}
