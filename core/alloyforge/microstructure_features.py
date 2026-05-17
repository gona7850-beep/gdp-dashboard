"""Microstructure (phase-fraction) features for the Nb-Si cleaned dataset.

The cleaned database (``data/nb_si/nb_silicide_cleaned.csv``) ships 64
phase-presence columns (binary 0/1 indicators per phase per row). Raw
columns are too granular for ML — many phases appear in only 1-3 rows.
This module aggregates them into ~8 metallurgically meaningful family
flags + count features that downstream models can use directly.

Output per row (returned by :meth:`PhaseFractionFeaturizer.transform`):

  has_Nbss                — Nb solid solution (any variant)
  has_Nb3Si_family        — Nb3Si or (Nb,X)3Si type
  has_Nb5Si3_family       — Nb5Si3 or (Nb,X)5Si3 (α/β/γ all collapsed)
  has_Ti_rich_silicide    — Ti-rich silicide variants
  has_Laves_phase         — NbCr2, Cr2Nb, Cr2(Nb,X), Cr14_Laves
  has_boride              — NbB2/ZrB2/TaB2/B4C
  has_carbide             — NbC, ZrC, TiC
  has_intermetallic_other — Nb3Sn, FeNb5Si, FeNb4Si, π_phase, etc.
  n_phases                — total number of distinct phases present
  silicide_diversity      — count of distinct silicide phase variants

These features can be concatenated with composition-based features
(``CompositionFeaturizer`` / ``ExtendedFeaturizer``) by passing them
into ``Dataset.process`` — the model will treat them as process /
microstructure descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Phase classification — each known phase name maps to one or more families.
# Tested against the columns in nb_silicide_cleaned.csv.
# ---------------------------------------------------------------------------

PHASE_FAMILIES: Dict[str, List[str]] = {
    "Nbss": [
        "Nbss", "Nb.1", "(Nb,Ti)ss", "Ti_rich_(Nb,X)ss", "Ti_rich_Nbss",
        "Nbss_no_Si", "lean_in_W_and_rich_in_Hf_Nbss", "Hf_rich_Nbss",
    ],
    "Nb3Si_family": [
        "Nb3Si", "(Nb,Zr)3Si", "γ_Nb3Si", "(Nb,Ti)3Si",
        "(primary_eutectic Nb3Si)",
    ],
    "Nb5Si3_family": [
        "Nb5Si3", "(Nb,Zr)5Si3", "(Nb,W)5Si3", "primary_α_Nb5Si3",
        "Eutectoid_α_Nb5Si3", "α_Nb5Si3", "β_Nb5Si3", "γ_Nb5Si3",
        "(Nb,Ti)5Si3", "α_(Nb,Ti)5Si3", "β_(Nb,Ti)5Si3", "γ_(Nb,Ti)5Si3",
        "Hf_rich_Nb5Si3", "Si_lean_(Nb,Ti)5Si3",
    ],
    "Ti_rich_silicide": [
        "Ti_rich_(Nb,Ti)5Si3", "Ti_rich_Nb5Si3", "Ti_rich_α_Nb5Si3",
        "Ti_rich_β_Nb5Si3", "Ti_rich_γ_Nb5Si3",
        "Ti_rich_α_(Nb,Ti)5Si3", "Ti_rich_β_(Nb,Ti)5Si3",
        "Ti_rich_γ_(Nb,Ti)5Si3",
    ],
    "Laves_phase": [
        "Cr14_Laves", "C15-Cr2Nb", "Cr2Nb", "Cr2(Nb,X)", "NbCr2",
    ],
    "boride": ["ZrB2", "TaB2", "B4C"],   # B4C is a boron-carbide; close enough
    "carbide": ["ZrC", "NbC", "TiC.1"],
    "intermetallic_other": [
        "Nb3Sn", "HfO2", "Hf_rich_Nb3Sn", "Sn_rich_Nb3Sn",
        "FeNb5Si", "FeNb4Si", "(Nb,Ti)3Sn", "π_phase",
        "Si_rich_(Nb,Ti)3Sn", "Nb4FeSi", "Tiss", "Zr.1",
        "Ti5Si3", "Ti2Ni", "Ti2Co", "Ti_Cr_V_rich_Nbss",
    ],
}

# Eutectic columns are flagged separately because they represent
# microstructure morphology, not chemistry alone.
EUTECTIC_COLUMNS = [
    "(Nb+Nb5Si3)_eut", "(Nbss+Nb5Si3)_eut",
    "(Nbss+Nb5Si3+NbCr2)_eut", "(Nb3Sn+Nb5Si3)_eut",
    "Nbss_β_Nb5Si3_eutectic", "Nbss_CrNb2_eutectic",
]


@dataclass
class PhaseFractionFeaturizer:
    """Group raw 0/1 phase columns into family-level features.

    The input DataFrame must contain the binary phase columns from
    ``nb_silicide_cleaned.csv``. Missing columns are treated as zero so
    the featurizer survives partial datasets.
    """

    families: Dict[str, List[str]] = None

    def __post_init__(self):
        if self.families is None:
            self.families = dict(PHASE_FAMILIES)

    @property
    def feature_names(self) -> List[str]:
        names = [f"has_{fam}" for fam in self.families]
        names += [
            "has_eutectic",
            "n_phases",
            "silicide_diversity",
            "is_Nbss_plus_silicide",
        ]
        return names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-row family flags + summary counts.

        Returns a DataFrame with one row per input row and the feature
        columns described in :attr:`feature_names`. Numeric columns
        outside the known phase list are ignored.
        """
        out = {}
        # Family-level presence flags
        for fam, members in self.families.items():
            present = [c for c in members if c in df.columns]
            if present:
                flag = (df[present].fillna(0).astype(float) > 0).any(axis=1)
            else:
                flag = pd.Series([False] * len(df), index=df.index)
            out[f"has_{fam}"] = flag.astype(int)

        # Eutectic columns
        eut_present = [c for c in EUTECTIC_COLUMNS if c in df.columns]
        if eut_present:
            out["has_eutectic"] = (
                df[eut_present].fillna(0).astype(float) > 0
            ).any(axis=1).astype(int)
        else:
            out["has_eutectic"] = 0

        # Total phase count: sum across every known phase column
        all_phases = set()
        for ms in self.families.values():
            all_phases.update(ms)
        all_phases.update(EUTECTIC_COLUMNS)
        cols = [c for c in all_phases if c in df.columns]
        if cols:
            out["n_phases"] = (
                df[cols].fillna(0).astype(float) > 0
            ).sum(axis=1).astype(int)
        else:
            out["n_phases"] = 0

        # Silicide-variant diversity: count of distinct Nb3Si + Nb5Si3 phases
        silicide_cols = [
            c for c in (self.families["Nb3Si_family"]
                         + self.families["Nb5Si3_family"]
                         + self.families["Ti_rich_silicide"])
            if c in df.columns
        ]
        if silicide_cols:
            out["silicide_diversity"] = (
                df[silicide_cols].fillna(0).astype(float) > 0
            ).sum(axis=1).astype(int)
        else:
            out["silicide_diversity"] = 0

        # "Classic" Nb-Si in-situ composite indicator
        out["is_Nbss_plus_silicide"] = (
            (out["has_Nbss"] == 1)
            & ((out["has_Nb5Si3_family"] == 1)
                | (out["has_Nb3Si_family"] == 1))
        ).astype(int)

        result = pd.DataFrame(out, index=df.index)
        # Ensure column order matches feature_names
        return result[self.feature_names]


def load_cleaned_nb_si(path: str = "data/nb_si/nb_silicide_cleaned.csv"
                       ) -> pd.DataFrame:
    """Load the cleaned Nb-Si dataset; strip whitespace from string columns."""
    df = pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def split_columns(df: pd.DataFrame) -> tuple[List[str], List[str], List[str]]:
    """Return (element_columns, phase_columns, property_columns) from the
    cleaned-dataset schema."""
    cols = df.columns.tolist()
    # Element columns: from "Nb" to before "Nbss"
    try:
        i_nb = cols.index("Nb")
        i_phase = cols.index("Nbss")
    except ValueError:
        return [], [], []
    element_columns = [c for c in cols[i_nb:i_phase]
                        if c and not c.startswith("HT")
                        and c != "Withdrawl_Rate(mm/min^-1)"]
    # Phase columns: from Nbss until first property column
    prop_start_keywords = ("Fracture_toughness_temp", "Fracture_toughness")
    i_prop = next(
        (i for i, c in enumerate(cols) if c in prop_start_keywords), len(cols)
    )
    phase_columns = cols[i_phase:i_prop]
    property_columns = cols[i_prop:]
    return element_columns, phase_columns, property_columns


__all__ = [
    "EUTECTIC_COLUMNS",
    "PHASE_FAMILIES",
    "PhaseFractionFeaturizer",
    "load_cleaned_nb_si",
    "split_columns",
]
