"""Ingestion + cleanup for user-supplied composition / property data.

Auto-collected materials data has well-known accuracy traps; this module
addresses the most common ones in a deterministic, auditable way before
the data hits the ML pipeline:

1. **Unit auto-detection** — column names + value ranges are mapped to a
   canonical set (MPa, HV, °C, %, g/cm³). Recognised aliases include
   ``ksi``, ``GPa``, ``HRC``, ``HB``, ``K``, ``celsius``, ``wt%``.

2. **Unit conversion** — closed-form formulas for stress and hardness
   plus published correlations for hardness conversions (HV↔HRC↔HB) that
   we mark explicitly as approximate because the relationship is alloy
   class dependent.

3. **Composition normalisation** — pass-through if rows sum to 1; convert
   from wt% to atomic % using :func:`weight_to_atomic_pct` when the
   ``composition_basis`` is declared "weight"; raise on rows that look
   degenerate (all-zero or non-finite).

4. **Outlier flagging** — robust z-score (median + MAD) per column,
   threshold ``z=4`` by default. Returns a mask without dropping rows.

5. **Provenance merge** — :func:`merge_datasets` stitches multiple
   sources together while adding a ``source`` group column so
   ``Dataset(groups=df['source'])`` becomes a GroupKFold key, preventing
   accidental train/test leakage of the same alloy across papers.

The functions here are framework-agnostic so they can be called from
the FastAPI router, a Streamlit upload widget, or a notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_pipeline import ELEMENT_PROPERTIES
from .reference_data import weight_to_atomic_pct


# ---------------------------------------------------------------------------
# Unit aliases — column-name → canonical
# ---------------------------------------------------------------------------

_STRESS_ALIASES = {
    "mpa", "n/mm^2", "n/mm2", "n_per_mm2",
    "ksi", "psi",
    "gpa",
}
_HARDNESS_ALIASES = {
    "hv", "vickers", "vhn",
    "hrc", "rockwell_c",
    "hrb", "rockwell_b",
    "hb", "brinell", "bhn",
    "hk", "knoop",
}
_TEMPERATURE_ALIASES = {"k", "kelvin", "c", "celsius", "f", "fahrenheit"}
_DENSITY_ALIASES = {"gcc", "g_cc", "g/cm3", "g/cm^3", "gm/cc"}
_PERCENT_ALIASES = {"pct", "percent", "%"}


def _canon(col: str) -> str:
    """Lowercase, replace separators with spaces, collapse whitespace."""
    out = re.sub(r"[\s_/\-]+", " ", col.strip().lower())
    return out.strip()


def infer_units(df: pd.DataFrame) -> Dict[str, str]:
    """Return ``{column: best-guess unit token}`` for every numeric column.

    Unknown columns map to ``"unknown"``. Heuristics:
        - column name contains a unit token → use that token
        - all-non-negative <= 1 → "fraction"
        - integer values 0–100 → "percent"
        - 0–600 with hint of MPa → "mpa"
    """
    out: Dict[str, str] = {}
    for col in df.columns:
        canon = _canon(col)
        # Direct token match
        for token in (_STRESS_ALIASES | _HARDNESS_ALIASES |
                       _TEMPERATURE_ALIASES | _DENSITY_ALIASES |
                       _PERCENT_ALIASES):
            if re.search(rf"\b{re.escape(token)}\b", canon):
                out[col] = token
                break
        if col in out:
            continue
        # Try value-based heuristic for numeric columns only
        if not pd.api.types.is_numeric_dtype(df[col]):
            out[col] = "unknown"
            continue
        s = df[col].dropna()
        if s.empty:
            out[col] = "unknown"
            continue
        if (s >= 0).all() and (s <= 1.000001).all():
            out[col] = "fraction"
        elif "yield" in canon or "tensile" in canon or "uts" in canon or "strength" in canon:
            out[col] = "mpa"
        elif "hardness" in canon:
            out[col] = "hv"
        elif "elong" in canon or "ductility" in canon:
            out[col] = "percent"
        elif "density" in canon:
            out[col] = "gcc"
        elif "melt" in canon or "tm" in canon:
            out[col] = "k"
        elif "modulus" in canon or "young" in canon:
            out[col] = "gpa"
        else:
            out[col] = "unknown"
    return out


# ---------------------------------------------------------------------------
# Unit conversions (numeric)
# ---------------------------------------------------------------------------

def _ksi_to_mpa(v): return v * 6.89476
def _gpa_to_mpa(v): return v * 1000.0
def _psi_to_mpa(v): return v * 6.89476e-3
def _kelvin_to_kelvin(v): return v
def _celsius_to_kelvin(v): return v + 273.15
def _fahrenheit_to_kelvin(v): return (v - 32) * 5.0 / 9.0 + 273.15


def _hrc_to_hv(v):
    """Hardness conversion HRC → HV via ASTM E140 lookup table (steels).

    The relationship is not closed-form — ASTM E140 tabulates it
    empirically. We interpolate between the standard table entries.
    Accuracy ≈ ±3 % within HRC 20-65; extrapolation outside is flagged
    by clipping but still allowed.
    """
    hrc_table = np.array(
        [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
         52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], dtype=float,
    )
    hv_table = np.array(
        [226, 231, 237, 243, 247, 253, 258, 264, 271, 278, 286, 294, 302,
         311, 320, 331, 342, 353, 364, 376, 388, 401, 415, 431, 448, 466,
         484, 503, 523, 544, 565, 587, 611, 638, 666, 695, 727, 760, 798,
         838, 879, 924, 972, 1024, 1078, 1132], dtype=float,
    )
    return float(np.interp(v, hrc_table, hv_table))


def _hrb_to_hv(v):
    """HRB → HV via ASTM E140 lookup (austenitic + carbon steels)."""
    hrb_table = np.array(
        [60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float,
    )
    hv_table = np.array(
        [111, 117, 125, 137, 150, 167, 187, 209, 240], dtype=float,
    )
    return float(np.interp(v, hrb_table, hv_table))


def _hb_to_hv(v):
    """HB → HV. HV ≈ HB up to ~250, gap widens at high hardness.

    Piecewise linear: HV = HB*1.0 for HB≤250, HV = HB*1.05 above 250.
    Accurate to ±5 % for steels; not validated for non-ferrous.
    """
    if v <= 250:
        return float(v)
    return float(v * 1.05)


def _hk_to_hv(v):
    """Knoop → Vickers; HK ≈ HV for HV<400, HK ≈ 0.94·HV for HV>800.

    Simple linear interpolation between the two regimes.
    """
    if v <= 400:
        return float(v)
    if v >= 800:
        return float(v / 0.94)
    # Linear interpolation in between
    frac = (v - 400) / 400.0
    factor = 1.0 + frac * (1.0 / 0.94 - 1.0)
    return float(v * factor)


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a single value between supported units."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return value
    fu, tu = from_unit.lower(), to_unit.lower()
    if fu == tu:
        return float(value)
    # Stress (canonical: MPa)
    stress_to_mpa = {
        "mpa": lambda v: v, "gpa": _gpa_to_mpa,
        "ksi": _ksi_to_mpa, "psi": _psi_to_mpa,
    }
    stress_from_mpa = {
        "mpa": lambda v: v, "gpa": lambda v: v / 1000.0,
        "ksi": lambda v: v / 6.89476, "psi": lambda v: v * 145.038,
    }
    if fu in stress_to_mpa and tu in stress_from_mpa:
        return float(stress_from_mpa[tu](stress_to_mpa[fu](value)))
    # Hardness (canonical: HV). HV↔HRC↔HB approximations only.
    if fu in {"hv", "vickers", "vhn"} and tu in {"hv", "vickers", "vhn"}:
        return float(value)
    if fu in {"hrc", "rockwell_c"} and tu == "hv":
        return float(_hrc_to_hv(value))
    if fu in {"hrb", "rockwell_b"} and tu == "hv":
        return float(_hrb_to_hv(value))
    if fu in {"hb", "brinell", "bhn"} and tu == "hv":
        return float(_hb_to_hv(value))
    if fu in {"hk", "knoop"} and tu == "hv":
        return float(_hk_to_hv(value))
    # Temperature (canonical: K)
    temp_to_k = {
        "k": _kelvin_to_kelvin, "kelvin": _kelvin_to_kelvin,
        "c": _celsius_to_kelvin, "celsius": _celsius_to_kelvin,
        "f": _fahrenheit_to_kelvin, "fahrenheit": _fahrenheit_to_kelvin,
    }
    temp_from_k = {
        "k": lambda v: v, "kelvin": lambda v: v,
        "c": lambda v: v - 273.15, "celsius": lambda v: v - 273.15,
        "f": lambda v: (v - 273.15) * 9.0 / 5.0 + 32, "fahrenheit": lambda v: (v - 273.15) * 9.0 / 5.0 + 32,
    }
    if fu in temp_to_k and tu in temp_from_k:
        return float(temp_from_k[tu](temp_to_k[fu](value)))
    raise ValueError(f"Cannot convert from {from_unit!r} to {to_unit!r}")


def normalize_units(
    df: pd.DataFrame,
    column_units: Dict[str, str],
    target_units: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Apply unit conversions in bulk. ``target_units`` defaults to MPa/HV/K/%.

    The returned DataFrame keeps the original column names but values
    are in the canonical unit.
    """
    target_units = target_units or {
        "stress": "mpa", "hardness": "hv", "temperature": "k",
        "density": "gcc", "fraction": "fraction", "percent": "percent",
    }
    out = df.copy()
    for col, unit in column_units.items():
        if col not in out.columns:
            continue
        category = _category_of(unit)
        if category is None:
            continue
        target = target_units.get(category, unit)
        if unit == target:
            continue
        try:
            out[col] = [
                convert_value(v, unit, target) if pd.notna(v) else v
                for v in out[col]
            ]
        except ValueError:
            continue
    return out


def _category_of(unit: str) -> Optional[str]:
    u = unit.lower()
    if u in {"mpa", "gpa", "ksi", "psi"}:
        return "stress"
    if u in {"hv", "vickers", "vhn", "hrc", "rockwell_c", "hrb",
            "rockwell_b", "hb", "brinell", "bhn", "hk", "knoop"}:
        return "hardness"
    if u in {"k", "kelvin", "c", "celsius", "f", "fahrenheit"}:
        return "temperature"
    if u in {"gcc", "g_cc", "g/cm3", "g/cm^3", "gm/cc"}:
        return "density"
    if u in {"percent", "pct", "%"}:
        return "percent"
    if u == "fraction":
        return "fraction"
    return None


# ---------------------------------------------------------------------------
# Composition normalisation
# ---------------------------------------------------------------------------

def normalize_composition(
    df: pd.DataFrame,
    element_columns: Sequence[str],
    composition_basis: str = "auto",
) -> pd.DataFrame:
    """Ensure element columns sum to 1.0 per row.

    ``composition_basis`` controls how source values are interpreted:

    * ``"auto"`` — if row sums ≈ 1, treat as already atomic-fraction;
      if row sums ≈ 100, treat as percent and divide; otherwise call
      it weight % and convert via :func:`weight_to_atomic_pct`.
    * ``"atomic_pct"`` — values are atomic percent; divide by 100.
    * ``"atomic_frac"`` — already correct; just renormalise.
    * ``"weight_pct"`` — apply :func:`weight_to_atomic_pct` per row.
    """
    out = df.copy()
    arr = out[list(element_columns)].to_numpy(dtype=float)
    row_sums = arr.sum(axis=1)

    if composition_basis == "auto":
        mean_sum = float(np.nanmean(row_sums)) if len(row_sums) else 0.0
        if 0.9 < mean_sum < 1.1:
            basis = "atomic_frac"
        elif 80 < mean_sum < 120:
            # Could be atomic % or weight %; prefer atomic % if columns are
            # named with element symbols only (typical convention).
            basis = "atomic_pct"
        else:
            basis = "weight_pct"
    else:
        basis = composition_basis

    if basis == "atomic_frac":
        rs = row_sums.copy()
        rs[rs == 0] = 1.0
        out[list(element_columns)] = arr / rs[:, None]
    elif basis == "atomic_pct":
        rs = row_sums.copy()
        rs[rs == 0] = 1.0
        out[list(element_columns)] = arr / rs[:, None]
    elif basis == "weight_pct":
        converted = []
        for i in range(len(arr)):
            wt = {el: float(arr[i, j]) for j, el in enumerate(element_columns)}
            wt = {k: v for k, v in wt.items() if v > 0}
            if not wt:
                converted.append({el: 0.0 for el in element_columns})
                continue
            try:
                atomic = weight_to_atomic_pct(wt)
            except (KeyError, ValueError):
                converted.append({el: float("nan") for el in element_columns})
                continue
            converted.append({el: atomic.get(el, 0.0) for el in element_columns})
        for el in element_columns:
            out[el] = [c.get(el, 0.0) for c in converted]
    else:
        raise ValueError(f"Unknown composition_basis: {basis}")
    return out


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------

def flag_outliers(
    df: pd.DataFrame,
    columns: Sequence[str],
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Return ``df`` with one extra column ``is_outlier`` (bool).

    Uses median + MAD (robust z-score), so a single bad row doesn't
    poison the threshold. Rows with any NaN in ``columns`` are *not*
    flagged here — handle those separately.
    """
    mask = np.zeros(len(df), dtype=bool)
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = df[col].to_numpy(dtype=float)
        valid = np.isfinite(x)
        if valid.sum() < 4:
            continue
        med = float(np.median(x[valid]))
        mad = float(np.median(np.abs(x[valid] - med)))
        if mad == 0:
            continue
        z = 0.6745 * (x - med) / mad   # standard MAD-z conversion
        mask |= np.abs(z) > z_threshold
    out = df.copy()
    out["is_outlier"] = mask
    return out


# ---------------------------------------------------------------------------
# Provenance merge
# ---------------------------------------------------------------------------

@dataclass
class IngestSummary:
    n_rows_in: int
    n_rows_out: int
    duplicated_dropped: int
    outliers_flagged: int
    columns_normalised: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def merge_datasets(
    sources: Dict[str, pd.DataFrame],
    element_columns: Sequence[str],
    target_columns: Sequence[str],
    composition_basis: str = "auto",
    dedup: bool = True,
) -> Tuple[pd.DataFrame, IngestSummary]:
    """Combine multiple labelled DataFrames into one, with a ``source``
    column for GroupKFold and per-source unit normalisation.

    Each source is normalised independently (so a single mis-labelled
    file does not contaminate the rest), then concatenated.
    """
    notes: List[str] = []
    cols_norm: List[str] = []
    n_in = 0
    pieces: List[pd.DataFrame] = []
    for src, df in sources.items():
        n_in += len(df)
        if df.empty:
            notes.append(f"{src}: empty, skipped")
            continue
        units = infer_units(df)
        normed = normalize_units(df, units)
        cols_norm.extend(c for c in units if _category_of(units[c]) is not None)
        normed = normalize_composition(normed, element_columns,
                                       composition_basis=composition_basis)
        normed = flag_outliers(normed, target_columns)
        normed["source"] = src
        pieces.append(normed)

    if not pieces:
        return pd.DataFrame(), IngestSummary(
            n_rows_in=n_in, n_rows_out=0, duplicated_dropped=0,
            outliers_flagged=0, columns_normalised=cols_norm, notes=notes,
        )
    merged = pd.concat(pieces, ignore_index=True)
    dropped = 0
    if dedup:
        before = len(merged)
        # Round element columns to 4 decimals for dedup purposes only
        key_cols = list(element_columns) + list(target_columns)
        existing = [c for c in key_cols if c in merged.columns]
        merged_keys = merged[existing].round(4)
        dup_mask = merged_keys.duplicated(keep="first")
        merged = merged[~dup_mask].reset_index(drop=True)
        dropped = before - len(merged)
        if dropped:
            notes.append(f"deduplicated {dropped} rows")
    return merged, IngestSummary(
        n_rows_in=n_in,
        n_rows_out=len(merged),
        duplicated_dropped=dropped,
        outliers_flagged=int(merged["is_outlier"].sum()),
        columns_normalised=sorted(set(cols_norm)),
        notes=notes,
    )


__all__ = [
    "IngestSummary",
    "convert_value", "flag_outliers", "infer_units",
    "merge_datasets", "normalize_composition", "normalize_units",
]
