"""External data-source clients for materials-science research.

Four clients are exposed, each behind a single function call:

* :func:`search_openalex` — open-source bibliographic graph (no auth,
  recommended to set ``OPENALEX_MAILTO`` env var per the rate-limit
  guidelines at https://docs.openalex.org/api/get-started).

* :func:`search_arxiv` — arXiv full-text papers (no auth, XML API).

* :func:`search_crossref` — CrossRef DOI/metadata lookup
  (https://api.crossref.org/works; no auth).

* :func:`materials_project_summary` — Materials Project structured
  materials database (https://api.materialsproject.org). Requires
  ``MP_API_KEY`` env var; falls back to a clear empty result + warning
  when missing.

Each function returns a tidy ``pandas.DataFrame`` with a fixed schema so
they can be merged via :func:`core.alloyforge.data_ingestion.merge_datasets`.
All clients have a hard timeout (default 15 s) and return an empty
DataFrame on any HTTP / parsing error — the platform never crashes
because of an external API outage.

Common columns
--------------

OpenAlex / arXiv / CrossRef return paper metadata:
    ``title``, ``authors``, ``year``, ``venue``, ``doi``, ``url``,
    ``abstract``, ``source``.

Materials Project returns materials records:
    ``mp_id``, ``formula``, ``elements``, ``space_group``, ``density``,
    ``volume``, ``energy_per_atom``, ``band_gap``, ``source``.

These are intentionally NOT in our ML-training schema (which expects
element columns + property columns). Treat the output as a *literature
index* the user can use to find papers and then run the LLM table
extractor (:mod:`core.alloyforge.llm_table_extractor`) on the relevant
PDFs to produce training rows.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _HAS_HTTPX = False

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0
_USER_AGENT = (
    "AlloyForge/0.1 (composition-design-platform; mailto:{email})"
)


def _ua() -> str:
    email = (os.environ.get("OPENALEX_MAILTO")
             or os.environ.get("CROSSREF_MAILTO")
             or "research@example.invalid")
    return _USER_AGENT.format(email=email)


def _empty(columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# OpenAlex (https://api.openalex.org)
# ---------------------------------------------------------------------------

OPENALEX_BASE = "https://api.openalex.org"
_PAPER_COLUMNS = ["title", "authors", "year", "venue", "doi", "url",
                  "abstract", "source"]


def search_openalex(
    query: str,
    per_page: int = 25,
    open_access_only: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Search OpenAlex for papers matching ``query``.

    OpenAlex stores abstracts as inverted indexes; we reconstruct a plain
    text string. Filters to open-access by default to stay within the
    repository's ethics policy.
    """
    if not _HAS_HTTPX:
        log.warning("httpx not available; cannot query OpenAlex")
        return _empty(_PAPER_COLUMNS)
    params = {
        "search": query,
        "per_page": min(per_page, 200),
        "select": "id,title,authorships,publication_year,host_venue,"
                  "primary_location,doi,abstract_inverted_index,open_access",
    }
    if open_access_only:
        params["filter"] = "open_access.is_oa:true"
    url = f"{OPENALEX_BASE}/works?" + urllib.parse.urlencode(params)
    try:
        r = httpx.get(url, headers={"User-Agent": _ua()}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning(f"OpenAlex query failed: {exc}")
        return _empty(_PAPER_COLUMNS)
    rows = []
    for w in data.get("results", []):
        rows.append({
            "title": w.get("title") or "",
            "authors": ", ".join(
                a.get("author", {}).get("display_name", "")
                for a in w.get("authorships", [])[:5]
            ),
            "year": w.get("publication_year"),
            "venue": ((w.get("host_venue") or {}).get("display_name") or
                      (w.get("primary_location") or {}).get("source", {}).get("display_name", "")),
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "url": (w.get("primary_location") or {}).get("landing_page_url", ""),
            "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
            "source": "openalex",
        })
    return pd.DataFrame(rows, columns=_PAPER_COLUMNS)


def _reconstruct_abstract(inv_idx: Optional[Dict[str, List[int]]]) -> str:
    if not inv_idx:
        return ""
    positions: List[tuple] = []
    for word, idxs in inv_idx.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


# ---------------------------------------------------------------------------
# arXiv (http://export.arxiv.org/api/query)
# ---------------------------------------------------------------------------

ARXIV_BASE = "http://export.arxiv.org/api/query"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(
    query: str,
    max_results: int = 25,
    timeout: float = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Search arXiv for papers. Returns metadata + abstract (no full text)."""
    if not _HAS_HTTPX:
        log.warning("httpx not available; cannot query arXiv")
        return _empty(_PAPER_COLUMNS)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max(1, min(max_results, 200)),
    }
    url = ARXIV_BASE + "?" + urllib.parse.urlencode(params)
    try:
        r = httpx.get(url, headers={"User-Agent": _ua()}, timeout=timeout)
        r.raise_for_status()
        tree = ET.fromstring(r.text)
    except Exception as exc:
        log.warning(f"arXiv query failed: {exc}")
        return _empty(_PAPER_COLUMNS)
    rows = []
    for entry in tree.findall("atom:entry", _ARXIV_NS):
        title = (entry.findtext("atom:title", default="", namespaces=_ARXIV_NS)
                 or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", default="",
                                    namespaces=_ARXIV_NS) or "").strip().replace("\n", " ")
        published = entry.findtext("atom:published", default="",
                                    namespaces=_ARXIV_NS)
        year = None
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                pass
        authors = [a.findtext("atom:name", default="",
                              namespaces=_ARXIV_NS)
                   for a in entry.findall("atom:author", _ARXIV_NS)]
        link_pdf = ""
        link_abs = ""
        for ln in entry.findall("atom:link", _ARXIV_NS):
            if ln.get("title") == "pdf":
                link_pdf = ln.get("href", "")
            elif ln.get("rel") == "alternate":
                link_abs = ln.get("href", "")
        rows.append({
            "title": title,
            "authors": ", ".join(authors[:5]),
            "year": year,
            "venue": "arXiv",
            "doi": "",
            "url": link_pdf or link_abs,
            "abstract": abstract,
            "source": "arxiv",
        })
    return pd.DataFrame(rows, columns=_PAPER_COLUMNS)


# ---------------------------------------------------------------------------
# CrossRef (https://api.crossref.org/works)
# ---------------------------------------------------------------------------

CROSSREF_BASE = "https://api.crossref.org/works"


def search_crossref(
    query: str,
    rows: int = 25,
    timeout: float = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Search CrossRef for DOI-backed publications."""
    if not _HAS_HTTPX:
        log.warning("httpx not available; cannot query CrossRef")
        return _empty(_PAPER_COLUMNS)
    params = {"query": query, "rows": min(rows, 100)}
    url = CROSSREF_BASE + "?" + urllib.parse.urlencode(params)
    try:
        r = httpx.get(url, headers={"User-Agent": _ua()}, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("message", {}).get("items", [])
    except Exception as exc:
        log.warning(f"CrossRef query failed: {exc}")
        return _empty(_PAPER_COLUMNS)
    rows_out = []
    for it in data:
        title = (it.get("title") or [""])[0]
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in it.get("author", [])[:5]
        )
        year = (it.get("issued", {}).get("date-parts", [[None]])[0][0])
        venue = ((it.get("container-title") or [""])[0]
                 or it.get("publisher", ""))
        rows_out.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "doi": it.get("DOI", ""),
            "url": it.get("URL", ""),
            "abstract": it.get("abstract", ""),
            "source": "crossref",
        })
    return pd.DataFrame(rows_out, columns=_PAPER_COLUMNS)


# ---------------------------------------------------------------------------
# Materials Project (https://api.materialsproject.org)
# ---------------------------------------------------------------------------

MP_BASE = "https://api.materialsproject.org"
_MP_COLUMNS = ["mp_id", "formula", "elements", "space_group", "density",
               "volume", "energy_per_atom", "band_gap", "source"]


@dataclass
class MPConfig:
    api_key: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    base_url: str = MP_BASE

    @classmethod
    def from_env(cls) -> "MPConfig":
        return cls(api_key=os.environ.get("MP_API_KEY"))

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def materials_project_summary(
    elements: Optional[List[str]] = None,
    formula: Optional[str] = None,
    num_elements_range: Optional[tuple] = None,
    page_size: int = 25,
    config: Optional[MPConfig] = None,
) -> pd.DataFrame:
    """Query Materials Project's summary endpoint.

    Requires ``MP_API_KEY``. Returns an empty DataFrame with a warning
    when the key is missing — never raises.

    Parameters
    ----------
    elements
        e.g. ``["Fe", "Ni", "Cr"]`` — only return materials containing
        all of these elements.
    formula
        e.g. ``"Ti3Al"`` — anonymous formula or chemical formula.
    num_elements_range
        ``(min, max)`` tuple to bound element count.
    """
    cfg = config or MPConfig.from_env()
    if not _HAS_HTTPX:
        log.warning("httpx not available; cannot query Materials Project")
        return _empty(_MP_COLUMNS)
    if not cfg.has_key:
        log.warning("MP_API_KEY not set; Materials Project query skipped")
        return _empty(_MP_COLUMNS)
    headers = {"X-API-KEY": cfg.api_key, "User-Agent": _ua()}
    params: Dict[str, Any] = {
        "_per_page": min(page_size, 100),
        "_fields": "material_id,formula_pretty,elements,symmetry,"
                   "density,volume,energy_per_atom,band_gap",
    }
    if elements:
        params["elements"] = ",".join(elements)
    if formula:
        params["formula"] = formula
    if num_elements_range:
        lo, hi = num_elements_range
        params["nelements_min"] = int(lo)
        params["nelements_max"] = int(hi)
    url = cfg.base_url.rstrip("/") + "/materials/summary/"
    try:
        r = httpx.get(url, headers=headers, params=params, timeout=cfg.timeout)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as exc:
        log.warning(f"Materials Project query failed: {exc}")
        return _empty(_MP_COLUMNS)
    rows = []
    for it in data:
        sym = it.get("symmetry") or {}
        rows.append({
            "mp_id": it.get("material_id", ""),
            "formula": it.get("formula_pretty", ""),
            "elements": ",".join(it.get("elements") or []),
            "space_group": sym.get("symbol", ""),
            "density": it.get("density"),
            "volume": it.get("volume"),
            "energy_per_atom": it.get("energy_per_atom"),
            "band_gap": it.get("band_gap"),
            "source": "materials_project",
        })
    return pd.DataFrame(rows, columns=_MP_COLUMNS)


# ---------------------------------------------------------------------------
# Provider status — used by the FastAPI router and verify.py
# ---------------------------------------------------------------------------

def provider_status() -> Dict[str, Any]:
    """Snapshot of which external providers are usable right now."""
    return {
        "httpx_available": _HAS_HTTPX,
        "openalex": {"requires_auth": False, "mailto_set":
                     bool(os.environ.get("OPENALEX_MAILTO"))},
        "arxiv": {"requires_auth": False},
        "crossref": {"requires_auth": False, "mailto_set":
                     bool(os.environ.get("CROSSREF_MAILTO"))},
        "materials_project": {
            "requires_auth": True,
            "api_key_set": bool(os.environ.get("MP_API_KEY")),
        },
    }


__all__ = [
    "MPConfig",
    "materials_project_summary",
    "provider_status",
    "search_arxiv",
    "search_crossref",
    "search_openalex",
]
