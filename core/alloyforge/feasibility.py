"""
Feasibility analysis for designed compositions.

We separate *feasibility* from *property prediction*. A composition can score well
on the ML model yet be infeasible because:
    1. It violates basic Hume-Rothery rules (size mismatch, electronegativity diff).
    2. It implies an undesired phase per CALPHAD/empirical rule (e.g., σ-phase tendency
       in Ni superalloys, BCC↔HCP transus in Ti alloys outside operating window).
    3. The composition falls outside the processability window of the chosen
       manufacturing route (e.g., VED outside the keyhole/lack-of-fusion limits for L-PBF).

The default constraint set is intentionally generic; teams should subclass
``FeasibilityChecker`` to inject domain knowledge (CALPHAD calls, processability
maps, cost ceilings, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_pipeline import ELEMENT_PROPERTIES, CompositionFeaturizer


# ---------------------------------------------------------------------------
@dataclass
class Constraint:
    """A named, scalar constraint expressed as g(comp, proc) ≤ 0.

    ``severity`` ∈ {"hard", "soft"}. Hard violations mark the candidate infeasible;
    soft violations are penalized but reported as warnings.
    """

    name: str
    fn: Callable[[pd.Series, Optional[pd.Series]], float]
    severity: str = "hard"
    description: str = ""

    def evaluate(self, comp: pd.Series, proc: Optional[pd.Series] = None) -> float:
        return float(self.fn(comp, proc))


# ---------------------------------------------------------------------------
def hume_rothery_size_mismatch(threshold_pct: float = 6.5) -> Constraint:
    """δ = 100·√(Σ c_i (1 - r_i/r̄)²) should be below ~6.5% for solid-solution stability
    (Zhang et al., Adv. Eng. Mater. 2008)."""
    def _fn(comp: pd.Series, proc):
        comp = comp / max(comp.sum(), 1e-9)
        elements = [e for e in comp.index if e in ELEMENT_PROPERTIES]
        r = np.array([ELEMENT_PROPERTIES[e]["radius"] for e in elements])
        c = comp[elements].to_numpy()
        r_bar = float(c @ r)
        if r_bar <= 0:
            return 1e6
        delta = 100 * float(np.sqrt(np.sum(c * (1 - r / r_bar) ** 2)))
        return delta - threshold_pct
    return Constraint(
        name=f"hume_rothery_delta<={threshold_pct}%",
        fn=_fn,
        severity="soft",
        description="Atomic-size mismatch (δ). High δ destabilizes solid solutions.",
    )


def vec_window(low: float, high: float) -> Constraint:
    """Valence electron concentration window. Useful for HEA phase prediction
    (VEC > 8 → FCC, VEC < 6.87 → BCC; the user picks the relevant window)."""
    def _fn(comp: pd.Series, proc):
        comp = comp / max(comp.sum(), 1e-9)
        elements = [e for e in comp.index if e in ELEMENT_PROPERTIES]
        v = np.array([ELEMENT_PROPERTIES[e]["vec"] for e in elements])
        c = comp[elements].to_numpy()
        vec = float(c @ v)
        # Two-sided: violation if outside [low, high]
        return max(low - vec, vec - high)
    return Constraint(
        name=f"VEC in [{low},{high}]",
        fn=_fn,
        severity="soft",
        description="Valence electron concentration window for desired phase.",
    )


def composition_sum_equals_one(tol: float = 1e-3) -> Constraint:
    """Hard constraint: input must be a proper composition."""
    def _fn(comp: pd.Series, proc):
        return abs(float(comp.sum()) - 1.0) - tol
    return Constraint(
        name="composition_sum=1",
        fn=_fn, severity="hard",
        description="Mole/atomic fractions must sum to 1 within tolerance.",
    )


def element_bounds(bounds: Dict[str, tuple]) -> List[Constraint]:
    """Per-element [min, max] bounds. Returns a list of constraints (one per side)."""
    cs: List[Constraint] = []
    for el, (lo, hi) in bounds.items():
        cs.append(Constraint(
            name=f"{el}>={lo}",
            fn=lambda comp, proc, e=el, b=lo: b - float(comp.get(e, 0.0)),
            severity="hard",
            description=f"Lower bound on {el}.",
        ))
        cs.append(Constraint(
            name=f"{el}<={hi}",
            fn=lambda comp, proc, e=el, b=hi: float(comp.get(e, 0.0)) - b,
            severity="hard",
            description=f"Upper bound on {el}.",
        ))
    return cs


def ved_window(low: float, high: float, ved_col: str = "VED") -> Constraint:
    """L-PBF / PBF-EB volumetric energy density window. Outside → keyhole or LoF."""
    def _fn(comp, proc):
        if proc is None or ved_col not in proc.index:
            return -1.0  # skip silently if not provided
        v = float(proc[ved_col])
        return max(low - v, v - high)
    return Constraint(
        name=f"VED in [{low},{high}] J/mm³",
        fn=_fn, severity="hard",
        description="Volumetric energy density processability window.",
    )


# ---------------------------------------------------------------------------
@dataclass
class FeasibilityResult:
    feasible: bool
    hard_violations: List[str] = field(default_factory=list)
    soft_violations: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "feasible": self.feasible,
            "hard_violations": self.hard_violations,
            "soft_violations": self.soft_violations,
            "scores": self.scores,
        }


@dataclass
class FeasibilityChecker:
    """Aggregate constraint checker. Subclass to add CALPHAD or DFT screens."""

    constraints: List[Constraint] = field(default_factory=list)

    def add(self, *cs: Constraint) -> "FeasibilityChecker":
        self.constraints.extend(cs)
        return self

    def check(self, composition: pd.Series,
              process: Optional[pd.Series] = None) -> FeasibilityResult:
        result = FeasibilityResult(feasible=True)
        for c in self.constraints:
            g = c.evaluate(composition, process)
            result.scores[c.name] = g
            if g > 0:
                if c.severity == "hard":
                    result.feasible = False
                    result.hard_violations.append(c.name)
                else:
                    result.soft_violations.append(c.name)
        return result

    def check_batch(self, compositions: pd.DataFrame,
                    processes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        rows = []
        for i, (_, comp_row) in enumerate(compositions.iterrows()):
            proc_row = processes.iloc[i] if processes is not None else None
            r = self.check(comp_row, proc_row)
            rows.append({
                "feasible": r.feasible,
                "n_hard": len(r.hard_violations),
                "n_soft": len(r.soft_violations),
                **{f"g[{k}]": v for k, v in r.scores.items()},
            })
        return pd.DataFrame(rows, index=compositions.index)


def default_checker(element_columns: Sequence[str],
                    bounds: Optional[Dict[str, tuple]] = None) -> FeasibilityChecker:
    """Sensible defaults for first-pass screening."""
    fc = FeasibilityChecker()
    fc.add(composition_sum_equals_one())
    fc.add(hume_rothery_size_mismatch())
    if bounds:
        fc.add(*element_bounds(bounds))
    return fc
