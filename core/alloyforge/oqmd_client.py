"""OQMD (Open Quantum Materials Database) REST client.

Wraps https://oqmd.org/oqmdapi/ so users can pull DFT compound data
straight into a :class:`KnownCompound`-shaped DataFrame for HTS
screening.

Graceful behaviour
------------------

* Returns an empty DataFrame on any network / parsing error.
* Logs a warning when the API rejects a query.
* The default OQMD instance allows anonymous read-only access; no API
  key needed.

The OQMD API is rate-limited; for large queries (>1000 compounds), set
``page_size`` and iterate over multiple pages or download a local
snapshot from https://oqmd.org/download/ and feed it via
``parse_oqmd_csv``.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .hts_screening import KnownCompound

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _HAS_HTTPX = False

log = logging.getLogger(__name__)

OQMD_BASE = "https://oqmd.org/oqmdapi/formationenergy"


# ---------------------------------------------------------------------------
def query_oqmd(
    elements: Optional[Sequence[str]] = None,
    formula: Optional[str] = None,
    n_atoms_max: Optional[int] = None,
    stability_max: Optional[float] = None,
    page_size: int = 50,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch compound records matching the filters from OQMD.

    Parameters
    ----------
    elements
        e.g. ``["Nb", "Si", "Ti"]`` — only compounds whose constituents
        are a subset of this list. Use this to bound a ternary search.
    formula
        Specific formula or anonymous formula (e.g. ``"AB2"``).
    n_atoms_max
        Drop compounds with more than this many atoms per unit cell.
    stability_max
        Drop compounds with stability above this (eV/atom; smaller = more
        stable). Default ``None`` keeps everything.
    page_size
        Records per API page. OQMD caps at ~100.

    Returns
    -------
    pandas.DataFrame
        Columns: ``formula, elements, n_atoms, space_group, delta_h_per_atom,
        stability, volume_per_atom, lattice_a, source``.
        Empty on any failure.
    """
    cols = ["formula", "elements", "n_atoms", "space_group",
            "delta_h_per_atom_ev", "stability", "volume_per_atom",
            "lattice_a", "source"]
    if not _HAS_HTTPX:
        log.warning("httpx not available; OQMD query skipped")
        return pd.DataFrame(columns=cols)

    params: Dict[str, Any] = {"limit": min(page_size, 100), "format": "json"}
    filters: List[str] = []
    if elements:
        # OQMD uses element_set= for "exact element-set membership"
        filters.append("element_set=" + ",".join(f"~{e}" for e in elements))
    if formula:
        filters.append(f"composition={urllib.parse.quote(formula)}")
    if n_atoms_max is not None:
        filters.append(f"natoms<{int(n_atoms_max) + 1}")
    if stability_max is not None:
        filters.append(f"stability<{float(stability_max)}")
    if filters:
        params["filter"] = " AND ".join(filters)

    try:
        r = httpx.get(OQMD_BASE, params=params, timeout=timeout,
                      headers={"User-Agent": "AlloyForge/0.1 (+research)"})
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as exc:
        log.warning(f"OQMD query failed: {exc}")
        return pd.DataFrame(columns=cols)

    rows = []
    for it in data:
        rows.append({
            "formula": it.get("name") or it.get("composition") or "",
            "elements": ",".join(it.get("element_set") or []),
            "n_atoms": it.get("natoms"),
            "space_group": (it.get("spacegroup") or {}).get("symbol", ""),
            "delta_h_per_atom_ev": it.get("delta_e"),
            "stability": it.get("stability"),
            "volume_per_atom": (it.get("unit_cell", {})
                                .get("volume_per_atom")),
            "lattice_a": (it.get("unit_cell", {}).get("lattice_a")),
            "source": "oqmd",
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
def to_known_compounds(df: pd.DataFrame) -> List[KnownCompound]:
    """Convert an OQMD query DataFrame to :class:`KnownCompound` records.

    The OQMD endpoint doesn't always include the lattice_c value or
    explicit stoichiometry counts; we fill those with sensible defaults
    so downstream code doesn't crash. Treat OQMD-derived compounds as
    *candidates for screening*, not curated ground truth — verify the
    space group and lattice parameters before using.
    """
    out: List[KnownCompound] = []
    for _, row in df.iterrows():
        elements = [e for e in str(row.get("elements") or "").split(",") if e]
        # Crude formula → stoichiometry: parse alternating element / count
        formula = row.get("formula") or ""
        stoich = _parse_formula_stoich(formula)
        try:
            out.append(KnownCompound(
                formula=str(formula),
                elements=tuple(elements),
                stoichiometry=stoich or {e: 1 for e in elements},
                space_group=str(row.get("space_group") or ""),
                lattice_a=float(row.get("lattice_a") or 0.0),
                lattice_c=None,
                volume_per_atom=float(row.get("volume_per_atom") or 0.0),
                formation_energy_per_atom_ev=float(
                    row.get("delta_h_per_atom_ev") or 0.0
                ),
                has_direct_tie_line_with=(),
                notes="from OQMD; verify lattice + tie line before trusting",
                source="oqmd",
            ))
        except (TypeError, ValueError):
            continue
    return out


_FORMULA_TOKEN = __import__("re").compile(r"([A-Z][a-z]?)(\d*)")


def _parse_formula_stoich(formula: str) -> Dict[str, int]:
    """Crude formula parser. Handles ``Nb5Si3``, ``NbAl3``, ``NbCr2``, etc.

    Does NOT handle parenthetical groups or fractional subscripts.
    """
    out: Dict[str, int] = {}
    for el, count in _FORMULA_TOKEN.findall(formula):
        if not el:
            continue
        out[el] = out.get(el, 0) + (int(count) if count else 1)
    return out


def parse_oqmd_csv(path: str) -> pd.DataFrame:
    """Load a locally-saved OQMD snapshot CSV; same schema as
    :func:`query_oqmd` returns."""
    df = pd.read_csv(path)
    return df


__all__ = ["query_oqmd", "to_known_compounds", "parse_oqmd_csv"]
