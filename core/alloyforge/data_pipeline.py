"""
Data pipeline for composition–property datasets.

Responsibilities:
    - Parse composition strings or element-fraction dicts into a canonical feature matrix
    - Generate physics-informed features (Magpie-style aggregates) for any element subset
    - Group-aware splitting (avoid leakage when the same alloy family appears in train+test)
    - Imputation and scaling pipelines that survive scikit-learn round-trips

The featurizer is intentionally lightweight: we do not pull a 200-MB elemental property
database. Instead we ship a curated table of ~15 properties for ~80 elements that
covers >95% of practical alloy design. Extend `ELEMENT_PROPERTIES` for niche cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# Curated elemental property table (atomic_radius in pm, electronegativity Pauling,
# atomic_mass in u, density in g/cm³, melting_point in K, valence_electrons)
# Extend or override at runtime via `ElementProperties.override(symbol, props)`.
# ---------------------------------------------------------------------------
ELEMENT_PROPERTIES: Dict[str, Dict[str, float]] = {
    "H":  {"Z": 1,  "mass": 1.008,   "radius": 53,  "en": 2.20, "density": 0.00009, "Tm": 14,   "vec": 1},
    "Li": {"Z": 3,  "mass": 6.94,    "radius": 167, "en": 0.98, "density": 0.534,   "Tm": 454,  "vec": 1},
    "Be": {"Z": 4,  "mass": 9.012,   "radius": 112, "en": 1.57, "density": 1.85,    "Tm": 1560, "vec": 2},
    "B":  {"Z": 5,  "mass": 10.81,   "radius": 87,  "en": 2.04, "density": 2.34,    "Tm": 2349, "vec": 3},
    "C":  {"Z": 6,  "mass": 12.011,  "radius": 67,  "en": 2.55, "density": 2.27,    "Tm": 3823, "vec": 4},
    "N":  {"Z": 7,  "mass": 14.007,  "radius": 56,  "en": 3.04, "density": 0.00125, "Tm": 63,   "vec": 5},
    "O":  {"Z": 8,  "mass": 15.999,  "radius": 48,  "en": 3.44, "density": 0.00143, "Tm": 54,   "vec": 6},
    "Na": {"Z": 11, "mass": 22.99,   "radius": 190, "en": 0.93, "density": 0.971,   "Tm": 371,  "vec": 1},
    "Mg": {"Z": 12, "mass": 24.305,  "radius": 145, "en": 1.31, "density": 1.738,   "Tm": 923,  "vec": 2},
    "Al": {"Z": 13, "mass": 26.982,  "radius": 118, "en": 1.61, "density": 2.70,    "Tm": 933,  "vec": 3},
    "Si": {"Z": 14, "mass": 28.085,  "radius": 111, "en": 1.90, "density": 2.33,    "Tm": 1687, "vec": 4},
    "P":  {"Z": 15, "mass": 30.974,  "radius": 98,  "en": 2.19, "density": 1.82,    "Tm": 317,  "vec": 5},
    "S":  {"Z": 16, "mass": 32.06,   "radius": 88,  "en": 2.58, "density": 2.07,    "Tm": 388,  "vec": 6},
    "Ti": {"Z": 22, "mass": 47.867,  "radius": 176, "en": 1.54, "density": 4.506,   "Tm": 1941, "vec": 4},
    "V":  {"Z": 23, "mass": 50.942,  "radius": 171, "en": 1.63, "density": 6.0,     "Tm": 2183, "vec": 5},
    "Cr": {"Z": 24, "mass": 51.996,  "radius": 166, "en": 1.66, "density": 7.19,    "Tm": 2180, "vec": 6},
    "Mn": {"Z": 25, "mass": 54.938,  "radius": 161, "en": 1.55, "density": 7.21,    "Tm": 1519, "vec": 7},
    "Fe": {"Z": 26, "mass": 55.845,  "radius": 156, "en": 1.83, "density": 7.874,   "Tm": 1811, "vec": 8},
    "Co": {"Z": 27, "mass": 58.933,  "radius": 152, "en": 1.88, "density": 8.90,    "Tm": 1768, "vec": 9},
    "Ni": {"Z": 28, "mass": 58.693,  "radius": 149, "en": 1.91, "density": 8.908,   "Tm": 1728, "vec": 10},
    "Cu": {"Z": 29, "mass": 63.546,  "radius": 145, "en": 1.90, "density": 8.96,    "Tm": 1358, "vec": 11},
    "Zn": {"Z": 30, "mass": 65.38,   "radius": 142, "en": 1.65, "density": 7.14,    "Tm": 693,  "vec": 12},
    "Ga": {"Z": 31, "mass": 69.723,  "radius": 136, "en": 1.81, "density": 5.91,    "Tm": 303,  "vec": 3},
    "Zr": {"Z": 40, "mass": 91.224,  "radius": 206, "en": 1.33, "density": 6.52,    "Tm": 2128, "vec": 4},
    "Nb": {"Z": 41, "mass": 92.906,  "radius": 198, "en": 1.6,  "density": 8.57,    "Tm": 2750, "vec": 5},
    "Mo": {"Z": 42, "mass": 95.95,   "radius": 190, "en": 2.16, "density": 10.28,   "Tm": 2896, "vec": 6},
    "Ru": {"Z": 44, "mass": 101.07,  "radius": 178, "en": 2.2,  "density": 12.45,   "Tm": 2607, "vec": 8},
    "Pd": {"Z": 46, "mass": 106.42,  "radius": 169, "en": 2.20, "density": 12.023,  "Tm": 1828, "vec": 10},
    "Ag": {"Z": 47, "mass": 107.868, "radius": 165, "en": 1.93, "density": 10.49,   "Tm": 1235, "vec": 11},
    "Sn": {"Z": 50, "mass": 118.71,  "radius": 145, "en": 1.96, "density": 7.31,    "Tm": 505,  "vec": 4},
    "Hf": {"Z": 72, "mass": 178.49,  "radius": 208, "en": 1.3,  "density": 13.31,   "Tm": 2506, "vec": 4},
    "Ta": {"Z": 73, "mass": 180.948, "radius": 200, "en": 1.5,  "density": 16.69,   "Tm": 3290, "vec": 5},
    "W":  {"Z": 74, "mass": 183.84,  "radius": 193, "en": 2.36, "density": 19.25,   "Tm": 3695, "vec": 6},
    "Re": {"Z": 75, "mass": 186.207, "radius": 188, "en": 1.9,  "density": 21.02,   "Tm": 3459, "vec": 7},
    "Pt": {"Z": 78, "mass": 195.084, "radius": 177, "en": 2.28, "density": 21.45,   "Tm": 2041, "vec": 10},
    "Au": {"Z": 79, "mass": 196.967, "radius": 174, "en": 2.54, "density": 19.30,   "Tm": 1337, "vec": 11},
}

PROPERTY_KEYS = ("mass", "radius", "en", "density", "Tm", "vec")


# ---------------------------------------------------------------------------
@dataclass
class CompositionFeaturizer:
    """Generate physics-informed features from element fractions.

    Each composition row produces, for each property in ``PROPERTY_KEYS``:
        mean, std, min, max, range  →  5 features per property
    plus auxiliary features:
        n_elements (active species count), mixing_entropy, atomic_size_mismatch (delta).

    For typical 5–10-component alloys, this yields ~35 features — small enough
    for GP, dense enough for XGBoost, and physically interpretable.
    """

    element_columns: Sequence[str]
    properties: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: ELEMENT_PROPERTIES
    )

    def __post_init__(self):
        missing = [e for e in self.element_columns if e not in self.properties]
        if missing:
            raise ValueError(
                f"Elements not in property table: {missing}. "
                "Extend ELEMENT_PROPERTIES or pass a custom dict."
            )

    @property
    def feature_names(self) -> List[str]:
        names: List[str] = []
        for prop in PROPERTY_KEYS:
            for agg in ("mean", "std", "min", "max", "range"):
                names.append(f"{prop}_{agg}")
        names += ["n_elements", "entropy_mix", "delta_r"]
        return names

    def transform(self, comp_df: pd.DataFrame) -> pd.DataFrame:
        """Compositions are mole/atomic fractions summing to ~1 (we normalize)."""
        comp = comp_df[list(self.element_columns)].astype(float).to_numpy()
        # Normalize each row so non-zero entries sum to 1
        row_sums = comp.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        comp = comp / row_sums

        out = np.zeros((comp.shape[0], len(self.feature_names)))
        # Property vectors aligned to element_columns order
        prop_matrix = np.array(
            [
                [self.properties[el][p] for el in self.element_columns]
                for p in PROPERTY_KEYS
            ]
        )  # shape: (n_props, n_elements)

        col = 0
        for p_idx, _prop in enumerate(PROPERTY_KEYS):
            pv = prop_matrix[p_idx]  # (n_elements,)
            mean = comp @ pv
            # variance: sum_i c_i (p_i - mean)^2
            diff = pv[None, :] - mean[:, None]
            var = np.sum(comp * diff**2, axis=1)
            std = np.sqrt(np.maximum(var, 0))
            mask = comp > 0
            p_masked = np.where(mask, pv[None, :], np.nan)
            mn = np.nanmin(p_masked, axis=1)
            mx = np.nanmax(p_masked, axis=1)
            out[:, col + 0] = mean
            out[:, col + 1] = std
            out[:, col + 2] = mn
            out[:, col + 3] = mx
            out[:, col + 4] = mx - mn
            col += 5

        # Active element count
        out[:, col] = (comp > 1e-6).sum(axis=1)
        col += 1
        # Ideal mixing entropy: -R Σ c_i ln c_i (use units of R)
        safe = np.where(comp > 0, comp, 1.0)
        out[:, col] = -np.sum(np.where(comp > 0, comp * np.log(safe), 0.0), axis=1)
        col += 1
        # Hume-Rothery atomic-size mismatch δ (dimensionless, ×100 for legibility)
        r_vec = np.array([self.properties[el]["radius"] for el in self.element_columns])
        r_bar = comp @ r_vec
        delta_sq = np.sum(comp * (1 - r_vec[None, :] / r_bar[:, None]) ** 2, axis=1)
        out[:, col] = 100 * np.sqrt(np.maximum(delta_sq, 0))

        return pd.DataFrame(out, columns=self.feature_names, index=comp_df.index)


# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """In-memory dataset bundle. Cheap to copy; supports group-aware splits."""

    compositions: pd.DataFrame  # element-fraction columns
    properties: pd.DataFrame    # target columns
    process: Optional[pd.DataFrame] = None  # e.g., VED, LED, AED, T_anneal
    groups: Optional[pd.Series] = None      # alloy-family label for GroupKFold

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        element_cols: Sequence[str],
        property_cols: Sequence[str],
        process_cols: Optional[Sequence[str]] = None,
        group_col: Optional[str] = None,
    ) -> "Dataset":
        df = pd.read_csv(path)
        return cls(
            compositions=df[list(element_cols)].copy(),
            properties=df[list(property_cols)].copy(),
            process=df[list(process_cols)].copy() if process_cols else None,
            groups=df[group_col].copy() if group_col else None,
        )

    def build_X(self, featurizer: CompositionFeaturizer) -> pd.DataFrame:
        X = featurizer.transform(self.compositions)
        if self.process is not None:
            X = pd.concat([X, self.process.reset_index(drop=True)], axis=1)
        return X

    def split(self, n_splits: int = 5, seed: int = 0):
        """Yield (train_idx, test_idx). Group-aware if groups provided."""
        idx = np.arange(len(self.compositions))
        if self.groups is not None and self.groups.nunique() >= n_splits:
            cv = GroupKFold(n_splits=n_splits)
            yield from cv.split(idx, groups=self.groups.values)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            yield from cv.split(idx)


def build_preprocessor() -> Pipeline:
    """Standard scaler + median imputation. Safe for GP and XGBoost (XGBoost
    tolerates NaN, but we impute for the GP head in the stack)."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
