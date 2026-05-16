"""Extended physics-informed features for alloy compositions.

The base ``CompositionFeaturizer`` in :mod:`core.alloyforge.data_pipeline`
produces 33 features (mean/std/min/max/range × 6 element properties +
n_elements + entropy_mix + delta_r). This module adds the heavier
metallurgical features that distinguish modern HEA / superalloy models
from naive composition aggregates:

* **Mixing enthalpy (Miedema-style approximation)** — pairwise
  electronegativity-difference and atomic-size-difference penalty,
  weighted by composition products. Captures intermetallic-formation
  tendency that linear aggregates miss.

* **Ω parameter (Yang 2012)** — Ω = T_m · ΔS_mix / |ΔH_mix|. Ω > 1
  predicts solid-solution stability over intermetallics. Single most
  predictive feature for HEA phase formation in published benchmarks.

* **Electronegativity standard deviation (Δχ)** — Pauling-EN spread.
  Independent of mean and complementary to Miedema enthalpy.

* **Valence-electron concentration windows** — sigmoid-bucketed VEC into
  4 windows (BCC < 6.87, dual 6.87-7.55, mixed 7.55-8.0, FCC > 8.0).
  Equivalent to a learned non-linear transform but explicit.

* **Stiffness proxy (k·T_m)** — sum of (c_i · k_i · Tm_i) with k_i
  literature-tabulated bond-strength constants. Strongly correlated with
  modulus and creep resistance.

These features are pure functions of composition, so they integrate
cleanly with both v1 and v2 forward models. Pass an
``ExtendedFeaturizer`` instance to ``ForwardModelV2`` to enable them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .data_pipeline import ELEMENT_PROPERTIES, CompositionFeaturizer

# ---------------------------------------------------------------------------
# Miedema-style binary interaction parameters
# ---------------------------------------------------------------------------
# Δh_AB ≈ -P · (Δϕ*)² + Q · (Δn_ws^{1/3})²   (per Miedema 1980)
# We use a simplified two-term approximation: electronegativity (en)
# difference dominates the negative term, atomic-radius difference
# dominates the positive term. Coefficients are tuned to give realistic
# kJ/mol magnitudes for the elements in ``ELEMENT_PROPERTIES``.
_MIEDEMA_P = 14.2   # kJ/mol per (Pauling EN)²
_MIEDEMA_Q = 0.31   # kJ/mol per (radius in pm)²·(10⁻³)


def _miedema_pair_enthalpy(en_a: float, en_b: float,
                           r_a: float, r_b: float) -> float:
    """Pairwise interaction enthalpy in kJ/mol."""
    d_en = en_a - en_b
    d_r = (r_a - r_b) / 100.0  # scale to similar magnitude
    return -_MIEDEMA_P * d_en ** 2 + _MIEDEMA_Q * 1000 * d_r ** 2 * 1e-3


# ---------------------------------------------------------------------------
# VEC bucket centers and slope for sigmoid bucketing
# ---------------------------------------------------------------------------
_VEC_BUCKETS = [
    ("bcc",  6.87,  6.0),  # below 6.87 → BCC predominant
    ("dual", 7.20,  6.0),
    ("mixed", 7.75, 8.0),
    ("fcc",  8.20,  6.0),
]


def _sigmoid(x: float, center: float, slope: float) -> float:
    return 1.0 / (1.0 + np.exp(-slope * (x - center)))


# ---------------------------------------------------------------------------
# Element stiffness proxies (rough k constants × 10⁻²)
# ---------------------------------------------------------------------------
# These are dimensionless bond-strength surrogates fitted so that
# Σ c_i · k_i · Tm_i / 10⁴ falls in the 1–4 range for common alloys.
_STIFFNESS_K: Dict[str, float] = {
    "H": 0.1, "Li": 0.5, "Be": 1.4, "B": 1.6, "C": 1.8, "N": 1.0, "O": 0.8,
    "Na": 0.4, "Mg": 0.7, "Al": 0.9, "Si": 1.5, "P": 1.2, "S": 1.0,
    "Ti": 1.6, "V": 1.7, "Cr": 1.6, "Mn": 1.4, "Fe": 1.7,
    "Co": 1.7, "Ni": 1.5, "Cu": 1.2, "Zn": 0.9, "Ga": 0.8,
    "Zr": 1.4, "Nb": 1.7, "Mo": 1.9, "Ru": 1.9, "Pd": 1.4, "Ag": 1.1,
    "Sn": 0.7, "Hf": 1.5, "Ta": 1.8, "W": 2.0, "Re": 1.9,
    "Pt": 1.6, "Au": 1.2,
}


# ---------------------------------------------------------------------------
@dataclass
class ExtendedFeaturizer:
    """Wrap a base ``CompositionFeaturizer`` and add ~10 metallurgical features.

    The output is the base 33 features **plus**:

    1. ``H_mix_kj``           — Σ_{i<j} 4·c_i·c_j · Δh_ij(Miedema)
    2. ``H_mix_abs_kj``       — Σ_{i<j} 4·c_i·c_j · |Δh_ij|
    3. ``Omega_yang``         — T_m̄ · ΔS_mix / |ΔH_mix|
    4. ``en_std``             — composition-weighted Pauling-EN std
    5. ``en_range``           — max EN − min EN over active elements
    6. ``vec_bcc_prob``       — sigmoid(VEC < 6.87)
    7. ``vec_dual_prob``      — sigmoid bump 6.87–7.55
    8. ``vec_mixed_prob``     — sigmoid bump 7.55–8.0
    9. ``vec_fcc_prob``       — sigmoid(VEC > 8.0)
    10. ``stiffness_proxy``   — Σ c_i · k_i · Tm_i / 10⁴
    """

    base: CompositionFeaturizer
    properties: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: ELEMENT_PROPERTIES
    )

    @property
    def element_columns(self) -> List[str]:
        return list(self.base.element_columns)

    @property
    def feature_names(self) -> List[str]:
        return list(self.base.feature_names) + [
            "H_mix_kj", "H_mix_abs_kj", "Omega_yang",
            "en_active_std", "en_active_range",
            "vec_bcc_prob", "vec_dual_prob", "vec_mixed_prob", "vec_fcc_prob",
            "stiffness_proxy",
        ]

    # ------------------------------------------------------------------ core
    def transform(self, comp_df: pd.DataFrame) -> pd.DataFrame:
        base_X = self.base.transform(comp_df)

        els = list(self.base.element_columns)
        c = comp_df[els].to_numpy(dtype=float)
        row_sums = c.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        c = c / row_sums

        n_samples = c.shape[0]
        out = np.zeros((n_samples, 10), dtype=float)

        en = np.array([self.properties[e]["en"] for e in els])
        radius = np.array([self.properties[e]["radius"] for e in els])
        tm = np.array([self.properties[e]["Tm"] for e in els])
        vec = np.array([self.properties[e]["vec"] for e in els])
        k = np.array([_STIFFNESS_K.get(e, 1.0) for e in els])

        # Pairwise Miedema enthalpy matrix
        h_pair = np.zeros((len(els), len(els)))
        for i in range(len(els)):
            for j in range(i + 1, len(els)):
                h = _miedema_pair_enthalpy(en[i], en[j], radius[i], radius[j])
                h_pair[i, j] = h
                h_pair[j, i] = h

        for s in range(n_samples):
            ci = c[s]
            # H_mix = Σ_{i<j} 4 c_i c_j Δh_ij  (kJ/mol)
            h_sum = 0.0
            h_abs_sum = 0.0
            for i in range(len(els)):
                for j in range(i + 1, len(els)):
                    term = 4.0 * ci[i] * ci[j] * h_pair[i, j]
                    h_sum += term
                    h_abs_sum += abs(term)
            out[s, 0] = h_sum
            out[s, 1] = h_abs_sum

            # ΔS_mix = -R Σ c_i ln c_i (we drop R for unitless)
            safe = np.where(ci > 0, ci, 1.0)
            s_mix = -np.sum(np.where(ci > 0, ci * np.log(safe), 0.0))
            # Mean melting point weighted by composition
            tm_bar = float(ci @ tm)
            # Ω = T_m · ΔS_mix / |ΔH_mix|; clip outliers so small-denominator
            # rows don't blow up StandardScaler's variance estimate.
            denom = max(abs(h_sum), 0.5)
            out[s, 2] = float(np.clip(tm_bar * s_mix / denom, 0.0, 50.0))

            # EN std + range (over active elements only)
            mask = ci > 1e-6
            if mask.sum() >= 2:
                en_bar = float(ci[mask] @ en[mask] / ci[mask].sum())
                en_var = float(np.sum(ci[mask] * (en[mask] - en_bar) ** 2))
                out[s, 3] = float(np.sqrt(max(en_var, 0)))
                out[s, 4] = float(en[mask].max() - en[mask].min())

            # VEC and bucket probabilities
            vec_bar = float(ci @ vec)
            out[s, 5] = _sigmoid(-vec_bar, -_VEC_BUCKETS[0][1], _VEC_BUCKETS[0][2])
            # dual + mixed: bump function (peak at center)
            out[s, 6] = _sigmoid(vec_bar, _VEC_BUCKETS[1][1], _VEC_BUCKETS[1][2]) * \
                         _sigmoid(-vec_bar, -_VEC_BUCKETS[2][1], _VEC_BUCKETS[1][2])
            out[s, 7] = _sigmoid(vec_bar, _VEC_BUCKETS[2][1], _VEC_BUCKETS[2][2]) * \
                         _sigmoid(-vec_bar, -_VEC_BUCKETS[3][1], _VEC_BUCKETS[2][2])
            out[s, 8] = _sigmoid(vec_bar, _VEC_BUCKETS[3][1], _VEC_BUCKETS[3][2])

            # Stiffness proxy
            out[s, 9] = float(np.sum(ci * k * tm)) / 1e4

        ext = pd.DataFrame(
            out,
            columns=[
                "H_mix_kj", "H_mix_abs_kj", "Omega_yang",
                "en_active_std", "en_active_range",
                "vec_bcc_prob", "vec_dual_prob", "vec_mixed_prob", "vec_fcc_prob",
                "stiffness_proxy",
            ],
            index=comp_df.index,
        )
        return pd.concat([base_X.reset_index(drop=True),
                          ext.reset_index(drop=True)], axis=1)


def make_extended(element_columns: Sequence[str]) -> ExtendedFeaturizer:
    """Factory: build an ``ExtendedFeaturizer`` from element column list."""
    return ExtendedFeaturizer(
        base=CompositionFeaturizer(element_columns=list(element_columns))
    )


__all__ = ["ExtendedFeaturizer", "make_extended"]
