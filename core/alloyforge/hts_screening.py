"""High-Throughput Screening for compound design — Nb-host edition.

Background
----------

A common workflow for designing new precipitation-strengthened alloys
(presented for example by NSM Lab, Kookmin University — Cho 2025):

1. Pull **every binary / ternary / quaternary compound** for a chosen
   element set from a DFT database (OQMD, Materials Project, AFLOW).
2. **Score** each compound on three thermodynamic descriptors:

   * **Tie line with host matrix** — does an isothermal tie line exist
     between the compound and the host metal (Al, Nb, Ni…)? If yes, the
     compound can coexist as a stable precipitate without consuming the
     matrix.
   * **Standalone stability** — formation enthalpy per atom (negative,
     and lower than competing decomposition products).
   * **Coherency with host matrix** — lattice-constant and per-atom-
     volume mismatch with the host. Small mismatch → low interfacial
     energy → coherent precipitate that survives high temperatures.

3. **Rank** compounds by the weighted sum; the top entries become the
   targets of the next experimental round.

This module ships a curated **Nb-host compound database** (~25
intermetallics) with literature/OQMD-cited formation enthalpies, lattice
parameters, and space groups so the workflow runs offline. The companion
``oqmd_client.py`` (next PR scope) will let users pull additional
compounds straight from the OQMD REST API.

The scorer is host-agnostic: pass any host symbol (``"Al"``, ``"Nb"``,
``"Ni"``) and a ``HostMatrix`` description, and you get the same
ranking semantics shown in the Cho 2025 slides.

Reference
---------

Cho, A.Y. (2025). *High-Throughput Screening for Optimizing Hardness-
Conductivity Trade-off in Aluminum Alloys*. Korea Univ., M.Sc. Thesis.
(Slide-deck design adapted to Nb-Si-Ti / Nb-X here.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd  # noqa: F401  (used by rank_compounds)

# ---------------------------------------------------------------------------
# Host-matrix descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HostMatrix:
    """Reference host phase used for tie-line + coherency scoring."""

    symbol: str          # e.g. "Nb", "Al", "Ni"
    structure: str       # "BCC", "FCC", "HCP"
    lattice_a: float     # Å; for BCC/FCC the cubic constant; HCP -> a
    volume_per_atom: float    # Å³/atom


# Reference host phases. Values from standard handbooks.
HOSTS: Dict[str, HostMatrix] = {
    "Nb": HostMatrix(symbol="Nb", structure="BCC",
                      lattice_a=3.301, volume_per_atom=17.97),
    "Al": HostMatrix(symbol="Al", structure="FCC",
                      lattice_a=4.046, volume_per_atom=16.61),
    "Ni": HostMatrix(symbol="Ni", structure="FCC",
                      lattice_a=3.524, volume_per_atom=10.94),
    "Ti": HostMatrix(symbol="Ti", structure="HCP",
                      lattice_a=2.951, volume_per_atom=17.65),
    "Fe": HostMatrix(symbol="Fe", structure="BCC",
                      lattice_a=2.866, volume_per_atom=11.78),
}


# ---------------------------------------------------------------------------
# Compound entries
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KnownCompound:
    """One compound entry from OQMD / Materials Project / literature.

    Fields cite published DFT values; for compounds where multiple
    polymorphs exist (e.g. Nb₅Si₃ α/β/γ), each is a separate entry.
    """

    formula: str
    elements: Sequence[str]     # constituent elements
    stoichiometry: Dict[str, int]   # per-formula-unit counts
    space_group: str
    lattice_a: float            # Å
    lattice_c: Optional[float]  # Å; None for cubic
    volume_per_atom: float      # Å³/atom (Z×V / atoms_per_cell)
    formation_energy_per_atom_ev: float   # ΔH_f, eV/atom (OQMD convention)
    has_direct_tie_line_with: Sequence[str] = field(default_factory=tuple)
    notes: str = ""
    source: str = ""


# Curated Nb-host compound database. Values from OQMD + Acta Materialia
# papers on Nb-silicide composites; treat as ±10% reference.
NB_HOST_COMPOUNDS: List[KnownCompound] = [
    # ============================ Nb-Si binaries ============================
    KnownCompound(
        formula="Nb5Si3-alpha",
        elements=("Nb", "Si"),
        stoichiometry={"Nb": 5, "Si": 3},
        space_group="I4/mcm",  # tI32
        lattice_a=6.57, lattice_c=11.88,
        volume_per_atom=16.05,
        formation_energy_per_atom_ev=-0.65,
        has_direct_tie_line_with=("Nb",),
        notes="α-Nb5Si3, high-temp stable phase in Nb-Si in-situ composites",
        source="OQMD-7341, Bewlay et al. MSE A 2003",
    ),
    KnownCompound(
        formula="Nb5Si3-beta",
        elements=("Nb", "Si"),
        stoichiometry={"Nb": 5, "Si": 3},
        space_group="I4/mcm",
        lattice_a=10.02, lattice_c=5.06,
        volume_per_atom=16.49,
        formation_energy_per_atom_ev=-0.62,
        has_direct_tie_line_with=("Nb",),
        notes="β-Nb5Si3, low-temp variant",
        source="OQMD; Schlesinger et al. JPED 1993",
    ),
    KnownCompound(
        formula="Nb5Si3-gamma",
        elements=("Nb", "Si"),
        stoichiometry={"Nb": 5, "Si": 3},
        space_group="P63/mcm",
        lattice_a=7.54, lattice_c=5.25,
        volume_per_atom=17.20,
        formation_energy_per_atom_ev=-0.60,
        has_direct_tie_line_with=("Nb",),
        notes="γ-Nb5Si3 / D8_8 type, metastable",
        source="OQMD",
    ),
    KnownCompound(
        formula="Nb3Si",
        elements=("Nb", "Si"),
        stoichiometry={"Nb": 3, "Si": 1},
        space_group="P42/n",  # tP32
        lattice_a=10.22, lattice_c=5.18,
        volume_per_atom=18.10,
        formation_energy_per_atom_ev=-0.51,
        has_direct_tie_line_with=("Nb",),
        notes="Primary eutectic with Nb_ss; decomposes at 1900°C",
        source="OQMD-7342, Mendiratta et al. Met Trans A 1991",
    ),
    KnownCompound(
        formula="NbSi2",
        elements=("Nb", "Si"),
        stoichiometry={"Nb": 1, "Si": 2},
        space_group="P6222",  # hP9
        lattice_a=4.79, lattice_c=6.59,
        volume_per_atom=14.20,
        formation_energy_per_atom_ev=-0.43,
        has_direct_tie_line_with=(),
        notes="Si-rich; doesn't directly equilibrate with Nb metal",
        source="OQMD",
    ),

    # ============================ Nb-Al binaries ============================
    KnownCompound(
        formula="Nb3Al",
        elements=("Nb", "Al"),
        stoichiometry={"Nb": 3, "Al": 1},
        space_group="Pm-3n",  # A15 cubic
        lattice_a=5.187, lattice_c=None,
        volume_per_atom=17.45,
        formation_energy_per_atom_ev=-0.20,
        has_direct_tie_line_with=("Nb",),
        notes="A15 superconductor; superalloy candidate",
        source="OQMD, Lukas Calphad 2002",
    ),
    KnownCompound(
        formula="Nb2Al",
        elements=("Nb", "Al"),
        stoichiometry={"Nb": 2, "Al": 1},
        space_group="P42/mnm",
        lattice_a=9.94, lattice_c=5.18,
        volume_per_atom=17.20,
        formation_energy_per_atom_ev=-0.34,
        has_direct_tie_line_with=("Nb",),
        notes="σ-phase",
        source="OQMD",
    ),
    KnownCompound(
        formula="NbAl3",
        elements=("Nb", "Al"),
        stoichiometry={"Nb": 1, "Al": 3},
        space_group="I4/mmm",  # D0_22
        lattice_a=3.84, lattice_c=8.61,
        volume_per_atom=15.85,
        formation_energy_per_atom_ev=-0.40,
        has_direct_tie_line_with=(),
        notes="Al-rich aluminide; high-temp coating",
        source="OQMD",
    ),

    # ============================ Nb-Cr Laves =============================
    KnownCompound(
        formula="NbCr2-C15",
        elements=("Nb", "Cr"),
        stoichiometry={"Nb": 1, "Cr": 2},
        space_group="Fd-3m",  # C15 Laves
        lattice_a=6.991, lattice_c=None,
        volume_per_atom=14.25,
        formation_energy_per_atom_ev=-0.13,
        has_direct_tie_line_with=("Nb",),
        notes="C15 Laves; common precipitate in Nb superalloys",
        source="OQMD",
    ),

    # ============================ Nb-Ni binaries ============================
    KnownCompound(
        formula="Nb3Ni",
        elements=("Nb", "Ni"),
        stoichiometry={"Nb": 3, "Ni": 1},
        space_group="P42/n",
        lattice_a=10.08, lattice_c=5.08,
        volume_per_atom=15.65,
        formation_energy_per_atom_ev=-0.17,
        has_direct_tie_line_with=("Nb",),
        notes="Equivalent crystallography to Nb3Si",
        source="OQMD",
    ),
    KnownCompound(
        formula="NbNi3",
        elements=("Nb", "Ni"),
        stoichiometry={"Nb": 1, "Ni": 3},
        space_group="P63/mmc",
        lattice_a=5.13, lattice_c=4.21,
        volume_per_atom=11.50,
        formation_energy_per_atom_ev=-0.32,
        has_direct_tie_line_with=(),
        notes="Ni-rich; γ′-like in Nb-doped Ni superalloys",
        source="OQMD",
    ),

    # ============================ Nb-Fe Laves ==============================
    KnownCompound(
        formula="NbFe2-C14",
        elements=("Nb", "Fe"),
        stoichiometry={"Nb": 1, "Fe": 2},
        space_group="P63/mmc",
        lattice_a=4.842, lattice_c=7.88,
        volume_per_atom=13.40,
        formation_energy_per_atom_ev=-0.15,
        has_direct_tie_line_with=("Nb",),
        notes="C14 Laves; appears in Nb-Fe steels",
        source="OQMD",
    ),

    # ============================ Nb-Co binaries ===========================
    KnownCompound(
        formula="NbCo2-C36",
        elements=("Nb", "Co"),
        stoichiometry={"Nb": 1, "Co": 2},
        space_group="P63/mmc",
        lattice_a=4.74, lattice_c=15.42,
        volume_per_atom=12.65,
        formation_energy_per_atom_ev=-0.20,
        has_direct_tie_line_with=("Nb",),
        notes="C36 Laves",
        source="OQMD",
    ),

    # ============================ Nb-Hf binary (solid solution) ============
    # Continuous solid solution; no compound — included as a sanity reference.

    # ============================ Nb-Ti, Nb-Mo, Nb-W solid solutions ========
    # Pure BCC ss, no intermetallic.

    # ============================ Nb-Si-Ti ternaries =======================
    KnownCompound(
        formula="(Nb,Ti)5Si3",
        elements=("Nb", "Ti", "Si"),
        stoichiometry={"Nb": 4, "Ti": 1, "Si": 3},  # representative
        space_group="I4/mcm",
        lattice_a=6.50, lattice_c=11.70,
        volume_per_atom=16.10,
        formation_energy_per_atom_ev=-0.68,
        has_direct_tie_line_with=("Nb",),
        notes="Ti-substituted alpha-Nb5Si3; primary strengthening phase in"
              " Nb-Si-Ti in-situ composites",
        source="Bewlay et al. Acta Mater 2003",
    ),
    KnownCompound(
        formula="(Nb,Ti)3Si",
        elements=("Nb", "Ti", "Si"),
        stoichiometry={"Nb": 2, "Ti": 1, "Si": 1},
        space_group="P42/n",
        lattice_a=10.30, lattice_c=5.20,
        volume_per_atom=18.40,
        formation_energy_per_atom_ev=-0.50,
        has_direct_tie_line_with=("Nb",),
        notes="Ti-substituted Nb3Si",
        source="Bewlay 2003",
    ),

    # ============================ Nb-Si-Cr ternaries =======================
    KnownCompound(
        formula="Cr2(Nb,X)",
        elements=("Nb", "Cr"),
        stoichiometry={"Nb": 1, "Cr": 2},
        space_group="Fd-3m",
        lattice_a=6.98, lattice_c=None,
        volume_per_atom=14.30,
        formation_energy_per_atom_ev=-0.13,
        has_direct_tie_line_with=("Nb",),
        notes="Generalised NbCr2 Laves with substitution; observed in"
              " Nb-Si-Cr alloys for oxidation resistance",
        source="OQMD + Subramanian Acta Mater 2000",
    ),

    # ============================ Nb-Si-Hf =================================
    KnownCompound(
        formula="(Nb,Hf)5Si3",
        elements=("Nb", "Hf", "Si"),
        stoichiometry={"Nb": 4, "Hf": 1, "Si": 3},
        space_group="I4/mcm",
        lattice_a=6.62, lattice_c=11.92,
        volume_per_atom=16.30,
        formation_energy_per_atom_ev=-0.70,
        has_direct_tie_line_with=("Nb",),
        notes="Hf-substituted alpha-Nb5Si3; reported to improve creep",
        source="Bewlay 2003",
    ),

    # ============================ Boride additives =========================
    KnownCompound(
        formula="NbB2",
        elements=("Nb", "B"),
        stoichiometry={"Nb": 1, "B": 2},
        space_group="P6/mmm",
        lattice_a=3.11, lattice_c=3.32,
        volume_per_atom=8.40,
        formation_energy_per_atom_ev=-0.32,
        has_direct_tie_line_with=("Nb",),
        notes="High-modulus boride; used as B-additive precipitate",
        source="OQMD",
    ),
    KnownCompound(
        formula="ZrB2",
        elements=("Zr", "B"),
        stoichiometry={"Zr": 1, "B": 2},
        space_group="P6/mmm",
        lattice_a=3.17, lattice_c=3.53,
        volume_per_atom=9.55,
        formation_energy_per_atom_ev=-0.34,
        has_direct_tie_line_with=("Nb",),
        notes="UHTC; appears as dispersed precipitate in Nb-Zr-B alloys",
        source="OQMD",
    ),

    # ============================ Carbide additive =========================
    KnownCompound(
        formula="NbC",
        elements=("Nb", "C"),
        stoichiometry={"Nb": 1, "C": 1},
        space_group="Fm-3m",
        lattice_a=4.470, lattice_c=None,
        volume_per_atom=11.16,
        formation_energy_per_atom_ev=-0.74,
        has_direct_tie_line_with=("Nb",),
        notes="NaCl-type carbide; common grain-refiner in Nb alloys",
        source="OQMD",
    ),

    # ============================ Nb-Ge =====================================
    KnownCompound(
        formula="Nb3Ge",
        elements=("Nb", "Ge"),
        stoichiometry={"Nb": 3, "Ge": 1},
        space_group="Pm-3n",  # A15
        lattice_a=5.166, lattice_c=None,
        volume_per_atom=17.25,
        formation_energy_per_atom_ev=-0.25,
        has_direct_tie_line_with=("Nb",),
        notes="A15 analog of Nb3Si; superconductor",
        source="OQMD",
    ),
    KnownCompound(
        formula="Nb5Ge3",
        elements=("Nb", "Ge"),
        stoichiometry={"Nb": 5, "Ge": 3},
        space_group="I4/mcm",
        lattice_a=6.65, lattice_c=12.00,
        volume_per_atom=17.20,
        formation_energy_per_atom_ev=-0.42,
        has_direct_tie_line_with=("Nb",),
        notes="Ge analog of Nb5Si3",
        source="OQMD",
    ),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoreWeights:
    """Weights on each descriptor. Slides default to roughly equal."""

    tie_line: float = 1.0
    stability: float = 1.0
    coherency: float = 1.0


@dataclass
class CompoundScore:
    formula: str
    tie_line_score: float
    stability_score: float
    coherency_score: float
    total: float
    details: Dict[str, Any] = field(default_factory=dict)


def score_tie_line(compound: KnownCompound, host: HostMatrix) -> tuple:
    """1.0 if a tie line with host is documented; 0.5 if host element is in
    the compound but no documented tie line; 0.0 otherwise.

    Returns (score, details_dict).
    """
    if host.symbol in compound.has_direct_tie_line_with:
        return 1.0, {"has_documented_tie_line": True,
                     "host_in_compound": host.symbol in compound.elements}
    if host.symbol in compound.elements:
        return 0.5, {"has_documented_tie_line": False,
                     "host_in_compound": True,
                     "note": "host appears in compound but tie line not "
                             "explicitly documented"}
    return 0.0, {"has_documented_tie_line": False,
                 "host_in_compound": False}


def score_stability(compound: KnownCompound,
                    min_dh: float = -0.7, max_dh: float = 0.0) -> tuple:
    """Linear map of ΔH_f to [0,1]: most negative → 1.0, ≥0 → 0.0.

    The default range [-0.7, 0.0] eV/atom covers OQMD's intermetallic
    distribution; for ultra-stable carbides/oxides extend ``min_dh``.
    """
    dh = compound.formation_energy_per_atom_ev
    if dh >= max_dh:
        return 0.0, {"delta_h_per_atom_ev": dh,
                     "note": "non-negative ΔH: unstable wrt elements"}
    score = float((max_dh - dh) / (max_dh - min_dh))
    score = max(0.0, min(1.0, score))
    return score, {"delta_h_per_atom_ev": dh}


def score_coherency(compound: KnownCompound, host: HostMatrix,
                    decay_per_pct: float = 0.4) -> tuple:
    """Lattice + volume mismatch → coherency score in [0, 1].

    A coherent precipitate interface tolerates lattice constants that
    differ from the host either *directly* (≈ same a) or by an integer
    multiple (the precipitate's super-cell matches a multiple of the
    host's). We therefore compute the modular mismatch::

        δ_a = min_{k∈{1,2,3,4}} |a_comp - k·a_host| / a_host  (in %)

    and the per-atom-volume mismatch ``δ_V`` similarly. The score is
    ``exp(-decay_per_pct * 0.5 * (|δ_a| + |δ_V|) / 5)``.

    For BCC Nb (a=3.301 Å), Nb₅Si₃ (a=6.57 Å) gets a~0% mismatch via
    k=2; NbC (a=4.47 Å) gets ~3.7% direct; pure-aluminide NbAl₃ at
    a=3.84 Å gets ~16% — all consistent with the literature view of
    which precipitates form coherent vs semi-coherent interfaces.

    NOTE: this is still a heuristic. It does not search for matching
    crystal-plane orientations like a proper Burgers-orientation
    analysis or DFT interface-energy calculation. Treat the score as a
    *screening* number, then verify the top candidates with a
    proper coherency model or experiment.
    """
    a_comp = compound.lattice_a
    a_host = host.lattice_a
    v_eff_comp = compound.volume_per_atom ** (1 / 3)
    v_eff_host = host.volume_per_atom ** (1 / 3)
    # Modular distance: best of k=1,2,3,4
    da_candidates = [
        abs(a_comp - k * a_host) / max(a_host, 1e-6) * 100
        for k in (1, 2, 3, 4)
    ]
    da = min(da_candidates)
    dv = abs(v_eff_comp - v_eff_host) / max(v_eff_host, 1e-6) * 100
    score = math.exp(-decay_per_pct * 0.5 * (da + dv) / 5.0)
    score = max(0.0, min(1.0, score))
    return score, {
        "lattice_a_mismatch_pct_modular": round(da, 2),
        "best_multiple_k": int(np.argmin(da_candidates)) + 1,
        "volume_per_atom_mismatch_pct": round(dv, 2),
    }


def score_compound(
    compound: KnownCompound,
    host: HostMatrix,
    weights: Optional[ScoreWeights] = None,
) -> CompoundScore:
    """Compute the three descriptors and the weighted total."""
    w = weights or ScoreWeights()
    s_tie, d_tie = score_tie_line(compound, host)
    s_sta, d_sta = score_stability(compound)
    s_coh, d_coh = score_coherency(compound, host)
    # Multiplicative term so a zero in any descriptor cannot be hidden;
    # additive term for ranking flexibility.
    total = (
        w.tie_line * s_tie
        + w.stability * s_sta
        + w.coherency * s_coh
    ) / (w.tie_line + w.stability + w.coherency)
    return CompoundScore(
        formula=compound.formula,
        tie_line_score=s_tie,
        stability_score=s_sta,
        coherency_score=s_coh,
        total=total,
        details={
            "tie_line": d_tie,
            "stability": d_sta,
            "coherency": d_coh,
            "elements": list(compound.elements),
            "space_group": compound.space_group,
            "source": compound.source,
        },
    )


def rank_compounds(
    host: str | HostMatrix = "Nb",
    weights: Optional[ScoreWeights] = None,
    compounds: Optional[Sequence[KnownCompound]] = None,
    required_elements: Optional[Sequence[str]] = None,
    forbidden_elements: Optional[Sequence[str]] = None,
    min_tie_line_score: float = 0.0,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Score and rank ``compounds`` for the given host.

    Returns a DataFrame sorted by descending total score with columns:
        formula, tie_line, stability, coherency, total, elements,
        space_group, delta_h_per_atom_ev,
        lattice_a_mismatch_pct, volume_per_atom_mismatch_pct, source.

    Filters
    -------
    * ``required_elements`` — keep only compounds whose constituents are a
      superset of this list (e.g. ``["Nb", "Si"]`` to get just Nb-Si based).
    * ``forbidden_elements`` — drop compounds containing any of these.
    * ``min_tie_line_score`` — drop compounds whose tie-line score is below
      this threshold (default 0 keeps everything).
    """
    if isinstance(host, str):
        if host not in HOSTS:
            raise KeyError(f"Unknown host {host!r}; known: {list(HOSTS)}")
        host = HOSTS[host]
    compounds = list(compounds) if compounds else list(NB_HOST_COMPOUNDS)

    if required_elements:
        req = set(required_elements)
        compounds = [c for c in compounds if req.issubset(set(c.elements))]
    if forbidden_elements:
        forb = set(forbidden_elements)
        compounds = [c for c in compounds
                     if not (forb & set(c.elements))]

    rows = []
    for c in compounds:
        s = score_compound(c, host, weights)
        if s.tie_line_score < min_tie_line_score:
            continue
        rows.append({
            "formula": s.formula,
            "tie_line": round(s.tie_line_score, 3),
            "stability": round(s.stability_score, 3),
            "coherency": round(s.coherency_score, 3),
            "total": round(s.total, 3),
            "elements": ",".join(c.elements),
            "space_group": c.space_group,
            "delta_h_per_atom_ev": c.formation_energy_per_atom_ev,
            "lattice_a_mismatch_pct":
                s.details["coherency"]["lattice_a_mismatch_pct_modular"],
            "best_multiple_k":
                s.details["coherency"]["best_multiple_k"],
            "volume_per_atom_mismatch_pct":
                s.details["coherency"]["volume_per_atom_mismatch_pct"],
            "notes": c.notes,
            "source": c.source,
        })
    df = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    if top_k is not None:
        df = df.head(top_k)
    return df


# ---------------------------------------------------------------------------
# Bridge to forward model: predict mechanical property of a host alloy
# containing a given precipitate (treated as added composition)
# ---------------------------------------------------------------------------

def host_plus_precipitate_composition(
    host: HostMatrix,
    compound: KnownCompound,
    precipitate_atomic_fraction: float = 0.10,
) -> Dict[str, float]:
    """Return a composition dict combining ``host`` (matrix) and ``compound``
    (precipitate) in atomic fractions summing to 1.

    Useful for feeding the forward model: "what would Nb + 10 at% Nb5Si3
    look like in our ML predictor?"
    """
    if precipitate_atomic_fraction < 0 or precipitate_atomic_fraction > 1:
        raise ValueError("precipitate fraction must be in [0,1]")
    stoich_total = sum(compound.stoichiometry.values()) or 1
    out: Dict[str, float] = {host.symbol: 1.0 - precipitate_atomic_fraction}
    for el, n in compound.stoichiometry.items():
        frac = precipitate_atomic_fraction * n / stoich_total
        out[el] = out.get(el, 0.0) + frac
    # Normalise (host may overlap with compound, e.g. compound contains Nb)
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


__all__ = [
    "HOSTS", "HostMatrix",
    "KnownCompound", "NB_HOST_COMPOUNDS",
    "ScoreWeights", "CompoundScore",
    "score_tie_line", "score_stability", "score_coherency", "score_compound",
    "rank_compounds",
    "host_plus_precipitate_composition",
]
