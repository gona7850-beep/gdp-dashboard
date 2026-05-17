"""LLM-mediated composition / property extraction from paper text.

Workflow
--------

1. User pastes (or fetches via the literature API) the text of a
   materials-science paper.

2. :func:`extract_alloy_table` chunks the text, asks Claude to return a
   structured JSON list of (composition + property + unit) tuples per
   row, validates the JSON shape, normalises units via
   :mod:`core.alloyforge.data_ingestion`, and returns a clean
   ``pandas.DataFrame`` ready for :func:`merge_datasets`.

3. Every extracted row carries a confidence flag (``"high"`` /
   ``"medium"`` / ``"low"``) and a ``warning`` field that flags
   sentences the LLM was unsure about — these are exactly the rows a
   human should spot-check before training.

Offline fallback
----------------

If ``anthropic`` is missing or ``ANTHROPIC_API_KEY`` is unset, a
regex-based heuristic captures the simplest tabular patterns
(``Fe-20Cr-10Ni: UTS = 500 MPa``). Coverage is far worse than the LLM
path; the platform always reports which mode was used so users don't
mistake heuristic noise for verified data.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .data_ingestion import convert_value, infer_units
from .data_pipeline import ELEMENT_PROPERTIES

log = logging.getLogger(__name__)

# Heavy ML imports stay deferred so this module can be imported in
# offline contexts.
try:
    from anthropic import Anthropic
    _HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover
    _HAS_ANTHROPIC = False


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You extract structured composition + mechanical-property
data from materials-science paper text. Return strict JSON only — no
prose, no Markdown fences. The JSON shape is:

{
  "rows": [
    {
      "composition": {"Fe": 70.0, "Cr": 18.0, "Ni": 9.0, "Mn": 2.0, "Si": 1.0},
      "composition_basis": "weight_pct" | "atomic_pct" | "atomic_frac",
      "properties": {"yield_mpa": 215.0, "tensile_mpa": 505.0,
                     "elong_pct": 70.0, "hardness_hv": 200.0},
      "alloy_name": "AISI 304",
      "process": "annealed",
      "confidence": "high" | "medium" | "low",
      "warning": ""  // sentence quoted from the text if you are uncertain
    },
    ...
  ]
}

Rules:
- Composition values must be numbers, not ranges. If the paper gives a
  range "17-21% Cr", use the midpoint and write the range in warning.
- Property keys must come from this list, in canonical units:
  yield_mpa, tensile_mpa, elong_pct, hardness_hv, density_gcc,
  youngs_gpa, melting_k. Convert ksi/GPa/HRC/°C yourself.
- Only include rows where you can pin both composition AND at least one
  property. Otherwise skip — do not invent values.
- If the paper reports as-cast and aged conditions separately, emit two
  rows with the process field distinguished ("as-cast" vs "aged").
- Confidence: "high" only when composition + every property is stated
  explicitly. "low" when you inferred either."""


@dataclass
class ExtractionReport:
    n_rows: int
    used_llm: bool
    model: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    confidence_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "used_llm": self.used_llm,
            "model": self.model,
            "warnings": self.warnings,
            "confidence_counts": self.confidence_counts,
        }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def extract_alloy_table(
    text: str,
    *,
    element_columns: Optional[Sequence[str]] = None,
    property_columns: Optional[Sequence[str]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 4000,
    use_llm: bool = True,
) -> tuple[pd.DataFrame, ExtractionReport]:
    """Return (DataFrame, ExtractionReport) for ``text``.

    The DataFrame's columns are: every element listed in
    ``element_columns`` (fractions, sum=1), every property in
    ``property_columns``, plus ``alloy_name``, ``process``,
    ``confidence``, ``warning``, ``source`` (always ``"llm_extracted"``).
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(), ExtractionReport(n_rows=0, used_llm=False)

    rows: List[Dict[str, Any]] = []
    used_llm = False
    used_model: Optional[str] = None
    warnings: List[str] = []

    if use_llm:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if _HAS_ANTHROPIC and key:
            try:
                rows, used_model = _llm_extract(text, key, model, max_tokens)
                used_llm = True
            except Exception as exc:
                warnings.append(f"LLM extraction failed: {exc}; "
                                 f"falling back to heuristic")
                rows = _heuristic_extract(text)
        else:
            warnings.append("No ANTHROPIC_API_KEY — using heuristic extractor")
            rows = _heuristic_extract(text)
    else:
        rows = _heuristic_extract(text)

    # Normalise to wide DataFrame
    if not rows:
        return pd.DataFrame(), ExtractionReport(
            n_rows=0, used_llm=used_llm, model=used_model, warnings=warnings,
        )
    return _rows_to_dataframe(
        rows, element_columns, property_columns, used_llm, used_model, warnings,
    )


def _llm_extract(text: str, api_key: str, model: Optional[str],
                 max_tokens: int) -> tuple[List[Dict[str, Any]], str]:
    client = Anthropic(api_key=api_key)
    chosen_model = model or os.environ.get(
        "CLAUDE_TABLE_EXTRACTOR_MODEL", "claude-sonnet-4-6"
    )
    # Keep input bounded so we don't blow the context window on huge papers.
    snippet = text[:50000]
    resp = client.messages.create(
        model=chosen_model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": snippet}],
    )
    raw = "".join(getattr(b, "text", "") for b in resp.content).strip()
    parsed = _parse_json(raw)
    if not parsed or "rows" not in parsed:
        return [], chosen_model
    return parsed["rows"], chosen_model


def _parse_json(s: str) -> Optional[Dict[str, Any]]:
    """Robust JSON extraction — strips Markdown fences if present."""
    s = s.strip()
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.+?)\s*```", s, re.DOTALL)
        if m:
            s = m.group(1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_ELEMENT_RE = re.compile(
    r"(?P<el>[A-Z][a-z]?)\s*[-:=]?\s*(?P<val>\d+(?:\.\d+)?)\s*(?:wt%?|at%?)?",
)
# Allow up to 30 non-digit characters between the property keyword and
# the value, so phrasings like "Yield strength = 215 MPa" or "hardness
# of about 200 HV" still match. We avoid `.*?` because it can cross
# sentence boundaries.
_PROP_RE = re.compile(
    r"(?P<prop>yield|tensile|ultimate|UTS|elongation|hardness|density)"
    r"[^\d\n.]{0,30}"
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>MPa|GPa|ksi|HV|HRC|HB|%|g/cm|kg/m)",
    re.IGNORECASE,
)


def _heuristic_extract(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Naive: only handle one alloy per text chunk.
    comp: Dict[str, float] = {}
    seen = set()
    for m in _ELEMENT_RE.finditer(text):
        el = m.group("el")
        if el not in ELEMENT_PROPERTIES or el in seen:
            continue
        try:
            comp[el] = float(m.group("val"))
            seen.add(el)
        except ValueError:
            pass

    props: Dict[str, float] = {}
    for m in _PROP_RE.finditer(text):
        prop = m.group("prop").lower()
        val = float(m.group("val"))
        unit = m.group("unit").lower()
        if prop in ("yield",):
            props["yield_mpa"] = _to_mpa(val, unit)
        elif prop in ("tensile", "ultimate", "uts"):
            props["tensile_mpa"] = _to_mpa(val, unit)
        elif prop == "hardness":
            props["hardness_hv"] = _to_hv(val, unit)
        elif prop == "elongation":
            props["elong_pct"] = val
        elif prop == "density":
            props["density_gcc"] = val
    if not comp or not props:
        return []
    return [{
        "composition": comp,
        "composition_basis": "weight_pct",   # safe default for free text
        "properties": props,
        "alloy_name": "",
        "process": "",
        "confidence": "low",
        "warning": "heuristic extraction; verify manually",
    }]


def _to_mpa(v: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "mpa":
        return v
    if unit == "gpa":
        return v * 1000.0
    if unit == "ksi":
        return v * 6.89476
    return v


def _to_hv(v: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "hv":
        return v
    try:
        return convert_value(v, unit, "hv")
    except ValueError:
        return v


# ---------------------------------------------------------------------------
# Row → DataFrame
# ---------------------------------------------------------------------------

def _rows_to_dataframe(
    rows: List[Dict[str, Any]],
    element_columns: Optional[Sequence[str]],
    property_columns: Optional[Sequence[str]],
    used_llm: bool, used_model: Optional[str],
    warnings: List[str],
) -> tuple[pd.DataFrame, ExtractionReport]:
    # Discover element + property columns if caller didn't provide them.
    found_els: List[str] = []
    found_props: List[str] = []
    for r in rows:
        for el in (r.get("composition") or {}):
            if el not in found_els:
                found_els.append(el)
        for p in (r.get("properties") or {}):
            if p not in found_props:
                found_props.append(p)
    els = list(element_columns) if element_columns else found_els
    props = list(property_columns) if property_columns else found_props

    out_rows: List[Dict[str, Any]] = []
    conf_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in rows:
        comp = r.get("composition") or {}
        basis = (r.get("composition_basis") or "weight_pct").lower()
        # Convert composition to atomic fractions
        if basis == "weight_pct":
            try:
                from .reference_data import weight_to_atomic_pct
                atomic = weight_to_atomic_pct({k: float(v) for k, v in comp.items()
                                                if k in ELEMENT_PROPERTIES})
            except Exception:
                atomic = {k: float(v) / 100.0 for k, v in comp.items()}
        elif basis == "atomic_pct":
            total = sum(float(v) for v in comp.values()) or 1.0
            atomic = {k: float(v) / total for k, v in comp.items()}
        else:  # atomic_frac
            total = sum(float(v) for v in comp.values()) or 1.0
            atomic = {k: float(v) / total for k, v in comp.items()}

        row_out: Dict[str, Any] = {}
        for el in els:
            row_out[el] = float(atomic.get(el, 0.0))
        for p in props:
            v = (r.get("properties") or {}).get(p)
            row_out[p] = float(v) if v is not None else float("nan")
        row_out["alloy_name"] = r.get("alloy_name", "")
        row_out["process"] = r.get("process", "")
        conf = (r.get("confidence") or "low").lower()
        if conf not in conf_counts:
            conf = "low"
        conf_counts[conf] += 1
        row_out["confidence"] = conf
        row_out["warning"] = r.get("warning", "")
        row_out["source"] = "llm_extracted" if used_llm else "heuristic_extracted"
        out_rows.append(row_out)

    df = pd.DataFrame(out_rows)
    report = ExtractionReport(
        n_rows=len(df), used_llm=used_llm, model=used_model,
        warnings=warnings, confidence_counts=conf_counts,
    )
    return df, report


__all__ = ["ExtractionReport", "extract_alloy_table"]
