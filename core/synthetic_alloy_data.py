"""Synthetic alloy dataset generator for demos and tests.

When users land on the composition-design platform without their own CSV
we want a "Try it now" button to still work. This module fabricates a
plausible-looking dataset (compositions on the simplex + a handful of
mechanical / functional properties) using simple physically-motivated rules
of mixing plus Gaussian noise. The numbers are not meant to be predictive
of real alloy behaviour — they only need enough structure for the ML model
to learn from and for the inverse-design loop to demonstrate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Element-wise base contributions used by the synthetic rules of mixing.
# Roughly inspired by published alloy datasets but tuned to be didactic.
_ELEMENT_PROFILES: dict[str, dict[str, float]] = {
    "Fe": {"yield_strength": 320, "hardness": 180, "elongation": 22, "density": 7.87},
    "Ni": {"yield_strength": 280, "hardness": 160, "elongation": 30, "density": 8.90},
    "Cr": {"yield_strength": 250, "hardness": 220, "elongation": 14, "density": 7.19},
    "Co": {"yield_strength": 340, "hardness": 210, "elongation": 18, "density": 8.90},
    "Al": {"yield_strength": 110, "hardness":  60, "elongation": 35, "density": 2.70},
    "Ti": {"yield_strength": 290, "hardness": 200, "elongation": 25, "density": 4.51},
    "Cu": {"yield_strength": 200, "hardness": 100, "elongation": 28, "density": 8.96},
    "Mo": {"yield_strength": 400, "hardness": 250, "elongation": 12, "density": 10.28},
    "V":  {"yield_strength": 360, "hardness": 230, "elongation": 16, "density": 6.11},
    "Mn": {"yield_strength": 240, "hardness": 170, "elongation": 20, "density": 7.21},
}


def default_elements() -> list[str]:
    return list(_ELEMENT_PROFILES.keys())


def default_properties() -> list[str]:
    return ["yield_strength", "hardness", "elongation", "density"]


def generate_synthetic_dataset(
    n_samples: int = 400,
    elements: list[str] | None = None,
    properties: list[str] | None = None,
    noise_scale: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate ``n_samples`` synthetic alloy rows.

    Compositions are drawn from a Dirichlet with concentration ``alpha=2``
    so the simplex is sampled away from corners (most real datasets cluster
    around the interior). Properties are computed by a rule-of-mixtures
    plus pairwise-interaction term plus Gaussian noise proportional to
    ``noise_scale``.
    """
    rng = np.random.default_rng(random_state)
    elements = elements or default_elements()
    properties = properties or default_properties()

    bad = [e for e in elements if e not in _ELEMENT_PROFILES]
    if bad:
        raise ValueError(
            f"Unknown elements: {bad}. Known: {list(_ELEMENT_PROFILES)}"
        )

    n_el = len(elements)
    comp = rng.dirichlet(np.full(n_el, 2.0), size=n_samples)
    # baseline rule-of-mixtures
    rows: list[dict[str, float]] = []
    # symmetric interaction matrix per property — small random terms so the
    # model has to learn non-linear behaviour, not just averaging.
    interactions = {
        p: rng.normal(0, 30, size=(n_el, n_el)) * 0
            + rng.normal(0, 1, size=(n_el, n_el))
        for p in properties
    }
    for p in properties:
        interactions[p] = (interactions[p] + interactions[p].T) / 2
        np.fill_diagonal(interactions[p], 0.0)

    for s in range(n_samples):
        row: dict[str, float] = {e: float(comp[s, i]) for i, e in enumerate(elements)}
        for p in properties:
            base = sum(comp[s, i] * _ELEMENT_PROFILES[e][p]
                       for i, e in enumerate(elements))
            inter = float(comp[s] @ interactions[p] @ comp[s])
            value = base + 50 * inter
            value *= 1.0 + rng.normal(0, noise_scale)
            row[p] = float(max(value, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def target_from_quantile(
    df: pd.DataFrame,
    properties: list[str],
    quantile: float = 0.9,
) -> dict[str, float]:
    """Pick a target = ``quantile`` of each property from ``df``.

    Useful for demos: "design an alloy in the top 10% on every property".
    """
    return {p: float(df[p].quantile(quantile)) for p in properties}
