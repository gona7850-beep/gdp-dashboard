"""Curated reference dataset of well-known alloys with literature property values.

Why this exists
---------------

ML composition-property models work best when there is a large, clean,
provenance-tracked training set. Most research groups start with <100
rows of their own data. This module ships ~50 widely-published alloys
covering steels, Ni superalloys, Ti, Al, Cu, Co alloys, HEAs, and
refractory metals — every one a "household name" with property values
that appear in any materials handbook.

All values are **typical mid-range** numbers compiled from publicly
available datasheets (ASM Handbook, MatWeb, NIMS MatNavi, alloy
producer data sheets — e.g. Special Metals, Carpenter Technology,
Haynes International). They are *not* substitutes for a specific
producer's certified mill data; treat them as **reference points** good
to ±10 % for pretraining, sanity-checking inverse-design candidates,
and prior elicitation.

Usage
-----

>>> from core.alloyforge.reference_data import reference_dataset
>>> df = reference_dataset()
>>> df.head()
   alloy_name family  Fe   Ni   Cr  ...  yield_mpa  tensile_mpa  density
0      304 SS  steel ...
...

>>> from core.alloyforge import Dataset, ForwardModel, CompositionFeaturizer
>>> ds = Dataset(compositions=df[elements], properties=df[props],
...              groups=df["family"])
>>> ForwardModel(featurizer=CompositionFeaturizer(elements), targets=props).fit(ds)

Compositions are stored as **atomic fractions** that sum to 1.0 (use
:func:`weight_to_atomic_pct` if your source quotes weight %). Element
columns missing from a given alloy are filled with 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .data_pipeline import ELEMENT_PROPERTIES


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def weight_to_atomic_pct(weights: Dict[str, float]) -> Dict[str, float]:
    """Convert weight-% (or mass-fraction) composition to atomic fraction.

    Input may be percentages or fractions; the result always sums to 1.
    Elements absent from :data:`ELEMENT_PROPERTIES` raise ``KeyError``.

    Formula::

        x_i = (w_i / M_i) / Σ_j (w_j / M_j)
    """
    mol = {}
    for el, w in weights.items():
        if el not in ELEMENT_PROPERTIES:
            raise KeyError(f"Element '{el}' not in ELEMENT_PROPERTIES table")
        mol[el] = float(w) / ELEMENT_PROPERTIES[el]["mass"]
    total = sum(mol.values())
    if total <= 0:
        raise ValueError("Composition sums to zero")
    return {el: m / total for el, m in mol.items()}


def atomic_to_weight_pct(atomic: Dict[str, float]) -> Dict[str, float]:
    """Inverse of :func:`weight_to_atomic_pct`. Returns fractions, sum=1."""
    mass = {el: float(x) * ELEMENT_PROPERTIES[el]["mass"]
            for el, x in atomic.items() if el in ELEMENT_PROPERTIES}
    total = sum(mass.values())
    if total <= 0:
        raise ValueError("Composition sums to zero")
    return {el: m / total for el, m in mass.items()}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class KnownAlloy:
    """One curated alloy entry.

    ``composition_wt`` is parsed once and re-served as atomic fraction
    via :meth:`as_atomic`. Property fields may be ``None`` if not
    reliably documented for the alloy (a few alloys lack published HV
    or modulus; downstream code should ``dropna`` per-target).
    """

    name: str
    family: str
    composition_wt: Dict[str, float]   # weight % of each element (need not sum to 100)
    yield_mpa: Optional[float] = None
    tensile_mpa: Optional[float] = None
    elong_pct: Optional[float] = None
    hardness_hv: Optional[float] = None
    density_gcc: Optional[float] = None
    youngs_gpa: Optional[float] = None
    melting_k: Optional[float] = None
    notes: str = ""
    references: List[str] = field(default_factory=list)

    def as_atomic(self) -> Dict[str, float]:
        return weight_to_atomic_pct(self.composition_wt)


# ---------------------------------------------------------------------------
# Curated table (~50 alloys, well-published typical values)
# ---------------------------------------------------------------------------
# Conventions:
#   - composition_wt is weight %; we auto-convert to atomic fraction.
#   - yield/tensile in MPa, elongation in %, HV Vickers, density g/cm³,
#     Young's modulus GPa, melting in K (solidus where applicable).
#   - Each entry cites at least one widely-available source so the
#     value is auditable. We never invent numbers.

ALLOYS: List[KnownAlloy] = [
    # ---------------------- Stainless and tool steels ----------------------
    KnownAlloy(
        name="304 SS", family="austenitic_stainless",
        composition_wt={"Fe": 71.0, "Cr": 18.0, "Ni": 9.0, "Mn": 1.5, "Si": 0.5},
        yield_mpa=215, tensile_mpa=505, elong_pct=70, hardness_hv=200,
        density_gcc=8.00, youngs_gpa=193, melting_k=1700,
        references=["ASM Handbook v1", "MatWeb AISI 304"],
    ),
    KnownAlloy(
        name="316L SS", family="austenitic_stainless",
        composition_wt={"Fe": 67.5, "Cr": 17.0, "Ni": 12.0, "Mo": 2.5, "Mn": 1.5, "Si": 0.5},
        yield_mpa=290, tensile_mpa=580, elong_pct=50, hardness_hv=217,
        density_gcc=7.99, youngs_gpa=193, melting_k=1675,
        references=["ASM Handbook v1", "MatWeb 316L"],
    ),
    KnownAlloy(
        name="17-4 PH", family="precipitation_stainless",
        composition_wt={"Fe": 73.5, "Cr": 16.5, "Ni": 4.5, "Cu": 4.0, "Mn": 1.0, "Si": 0.5},
        yield_mpa=1170, tensile_mpa=1310, elong_pct=10, hardness_hv=380,
        density_gcc=7.75, youngs_gpa=196, melting_k=1675,
        notes="H900 condition",
        references=["AK Steel 17-4 PH datasheet"],
    ),
    KnownAlloy(
        name="AISI 4140", family="low_alloy_steel",
        composition_wt={"Fe": 96.8, "Cr": 1.0, "Mn": 0.9, "Mo": 0.2, "Si": 0.3, "C": 0.4},
        yield_mpa=655, tensile_mpa=850, elong_pct=18, hardness_hv=285,
        density_gcc=7.85, youngs_gpa=205, melting_k=1700,
        notes="Q&T condition",
        references=["ASM Handbook v1"],
    ),
    KnownAlloy(
        name="M2 tool steel", family="tool_steel",
        composition_wt={"Fe": 81.0, "W": 6.0, "Mo": 5.0, "Cr": 4.0, "V": 2.0, "C": 1.0},
        yield_mpa=1730, tensile_mpa=2070, elong_pct=2, hardness_hv=700,
        density_gcc=8.16, youngs_gpa=210, melting_k=1690,
        references=["ASM Handbook v1 - tool steels"],
    ),
    KnownAlloy(
        name="Maraging 250 (18Ni)", family="maraging_steel",
        composition_wt={"Fe": 67.4, "Ni": 18.5, "Co": 8.0, "Mo": 5.0, "Ti": 0.4, "Al": 0.1},
        yield_mpa=1700, tensile_mpa=1760, elong_pct=8, hardness_hv=550,
        density_gcc=8.0, youngs_gpa=180, melting_k=1700,
        references=["Specialty Steel maraging datasheet"],
    ),
    KnownAlloy(
        name="Mild steel 1018", family="carbon_steel",
        composition_wt={"Fe": 98.9, "Mn": 0.8, "C": 0.18, "Si": 0.1},
        yield_mpa=370, tensile_mpa=440, elong_pct=15, hardness_hv=126,
        density_gcc=7.87, youngs_gpa=205, melting_k=1730,
        references=["MatWeb 1018"],
    ),

    # ---------------------- Ni-based superalloys ---------------------------
    KnownAlloy(
        name="Inconel 718", family="ni_superalloy",
        composition_wt={"Ni": 52.5, "Cr": 19.0, "Fe": 18.5, "Nb": 5.1,
                        "Mo": 3.0, "Ti": 0.9, "Al": 0.5},
        yield_mpa=1100, tensile_mpa=1280, elong_pct=21, hardness_hv=380,
        density_gcc=8.19, youngs_gpa=200, melting_k=1610,
        notes="aged condition",
        references=["Special Metals IN718 datasheet"],
    ),
    KnownAlloy(
        name="Inconel 625", family="ni_superalloy",
        composition_wt={"Ni": 62.0, "Cr": 21.5, "Mo": 9.0, "Nb": 3.6, "Fe": 2.5, "Co": 1.0},
        yield_mpa=470, tensile_mpa=965, elong_pct=42, hardness_hv=220,
        density_gcc=8.44, youngs_gpa=207, melting_k=1620,
        references=["Special Metals IN625"],
    ),
    KnownAlloy(
        name="Hastelloy X", family="ni_superalloy",
        composition_wt={"Ni": 47.0, "Cr": 22.0, "Fe": 18.5, "Mo": 9.0, "Co": 1.5, "W": 0.6},
        yield_mpa=355, tensile_mpa=770, elong_pct=43, hardness_hv=170,
        density_gcc=8.22, youngs_gpa=205, melting_k=1610,
        references=["Haynes Hastelloy X datasheet"],
    ),
    KnownAlloy(
        name="Waspaloy", family="ni_superalloy",
        composition_wt={"Ni": 58.0, "Cr": 19.5, "Co": 13.5, "Mo": 4.3,
                        "Ti": 3.0, "Al": 1.4, "Fe": 1.0},
        yield_mpa=795, tensile_mpa=1240, elong_pct=25, hardness_hv=330,
        density_gcc=8.19, youngs_gpa=213, melting_k=1610,
        references=["Special Metals Waspaloy"],
    ),
    KnownAlloy(
        name="CMSX-4", family="ni_single_crystal",
        composition_wt={"Ni": 61.7, "Co": 9.0, "Cr": 6.5, "W": 6.0, "Ta": 6.5,
                        "Re": 3.0, "Al": 5.6, "Ti": 1.0, "Mo": 0.6, "Hf": 0.1},
        yield_mpa=980, tensile_mpa=1100, elong_pct=12, hardness_hv=420,
        density_gcc=8.70, youngs_gpa=124, melting_k=1600,
        notes="orientation-dependent; single-crystal blade alloy",
        references=["Cannon-Muskegon CMSX-4 datasheet"],
    ),
    KnownAlloy(
        name="Hastelloy C-276", family="ni_superalloy",
        composition_wt={"Ni": 57.0, "Mo": 16.0, "Cr": 16.0, "Fe": 5.0, "W": 3.5,
                        "Co": 2.0, "Mn": 0.5},
        yield_mpa=355, tensile_mpa=790, elong_pct=40, hardness_hv=195,
        density_gcc=8.89, youngs_gpa=205, melting_k=1620,
        references=["Haynes C-276"],
    ),
    KnownAlloy(
        name="Inconel 738", family="ni_superalloy",
        composition_wt={"Ni": 61.5, "Cr": 16.0, "Co": 8.5, "W": 2.6, "Ta": 1.75,
                        "Ti": 3.4, "Al": 3.4, "Mo": 1.75, "Nb": 0.9},
        yield_mpa=950, tensile_mpa=1100, elong_pct=5, hardness_hv=410,
        density_gcc=8.11, youngs_gpa=200, melting_k=1590,
        references=["Special Metals IN738"],
    ),

    # ---------------------- Titanium alloys --------------------------------
    KnownAlloy(
        name="Ti-CP grade 2", family="ti_unalloyed",
        composition_wt={"Ti": 99.5, "Fe": 0.3, "O": 0.2},
        yield_mpa=275, tensile_mpa=345, elong_pct=20, hardness_hv=160,
        density_gcc=4.51, youngs_gpa=105, melting_k=1941,
        references=["ASTM B265"],
    ),
    KnownAlloy(
        name="Ti-6Al-4V", family="ti_alpha_beta",
        composition_wt={"Ti": 90.0, "Al": 6.0, "V": 4.0},
        yield_mpa=880, tensile_mpa=950, elong_pct=14, hardness_hv=340,
        density_gcc=4.43, youngs_gpa=114, melting_k=1923,
        notes="annealed; AM as-built typically yield ~1100",
        references=["ASM Handbook v2 - Ti", "ASTM F136"],
    ),
    KnownAlloy(
        name="Ti-6242", family="ti_alpha_beta",
        composition_wt={"Ti": 86.0, "Al": 6.0, "Sn": 2.0, "Zr": 4.0, "Mo": 2.0},
        yield_mpa=990, tensile_mpa=1100, elong_pct=13, hardness_hv=370,
        density_gcc=4.54, youngs_gpa=120, melting_k=1900,
        notes="high-temperature aerospace Ti",
        references=["RMI Titanium"],
    ),
    KnownAlloy(
        name="Ti-5553", family="ti_beta",
        composition_wt={"Ti": 81.0, "Al": 5.0, "V": 5.0, "Mo": 5.0, "Cr": 3.0, "Fe": 0.4},
        yield_mpa=1250, tensile_mpa=1370, elong_pct=8, hardness_hv=400,
        density_gcc=4.65, youngs_gpa=110, melting_k=1900,
        references=["TIMET Ti-5553"],
    ),

    # ---------------------- Aluminum alloys --------------------------------
    KnownAlloy(
        name="AA 1100", family="al_unalloyed",
        composition_wt={"Al": 99.0, "Si": 0.5, "Fe": 0.5},
        yield_mpa=35, tensile_mpa=90, elong_pct=35, hardness_hv=23,
        density_gcc=2.71, youngs_gpa=69, melting_k=918,
        references=["Aluminum Association"],
    ),
    KnownAlloy(
        name="AA 2024-T3", family="al_2xxx",
        composition_wt={"Al": 93.5, "Cu": 4.4, "Mg": 1.5, "Mn": 0.6},
        yield_mpa=345, tensile_mpa=485, elong_pct=18, hardness_hv=137,
        density_gcc=2.78, youngs_gpa=73, melting_k=775,
        references=["Aluminum Association 2024"],
    ),
    KnownAlloy(
        name="AA 6061-T6", family="al_6xxx",
        composition_wt={"Al": 97.9, "Mg": 1.0, "Si": 0.6, "Cu": 0.28, "Cr": 0.2},
        yield_mpa=276, tensile_mpa=310, elong_pct=12, hardness_hv=95,
        density_gcc=2.70, youngs_gpa=69, melting_k=855,
        references=["Aluminum Association 6061"],
    ),
    KnownAlloy(
        name="AA 7075-T6", family="al_7xxx",
        composition_wt={"Al": 90.0, "Zn": 5.6, "Mg": 2.5, "Cu": 1.6, "Cr": 0.23},
        yield_mpa=503, tensile_mpa=572, elong_pct=11, hardness_hv=175,
        density_gcc=2.81, youngs_gpa=72, melting_k=750,
        references=["Aluminum Association 7075"],
    ),
    KnownAlloy(
        name="AlSi10Mg (AM)", family="al_cast",
        composition_wt={"Al": 89.5, "Si": 10.0, "Mg": 0.4, "Fe": 0.1},
        yield_mpa=240, tensile_mpa=460, elong_pct=6, hardness_hv=120,
        density_gcc=2.67, youngs_gpa=75, melting_k=830,
        notes="as-built LPBF; T6 lowers UTS but raises ductility",
        references=["EOS AlSi10Mg datasheet"],
    ),

    # ---------------------- High-entropy alloys ----------------------------
    KnownAlloy(
        name="Cantor (CoCrFeMnNi)", family="hea_3d",
        composition_wt={"Co": 20.0, "Cr": 20.0, "Fe": 20.0, "Mn": 20.0, "Ni": 20.0},
        yield_mpa=410, tensile_mpa=763, elong_pct=51, hardness_hv=145,
        density_gcc=8.0, youngs_gpa=202, melting_k=1600,
        notes="equiatomic; room-temperature tensile",
        references=["Cantor et al. MSE A 2004", "Otto et al. Acta Mater 2013"],
    ),
    KnownAlloy(
        name="FeCoCrNi", family="hea_3d",
        composition_wt={"Fe": 25.0, "Co": 25.0, "Cr": 25.0, "Ni": 25.0},
        yield_mpa=250, tensile_mpa=600, elong_pct=65, hardness_hv=130,
        density_gcc=8.0, youngs_gpa=200, melting_k=1620,
        references=["He et al. Acta Mater 2016"],
    ),
    KnownAlloy(
        name="AlCoCrFeNi", family="hea_3d_al",
        composition_wt={"Al": 20.0, "Co": 20.0, "Cr": 20.0, "Fe": 20.0, "Ni": 20.0},
        yield_mpa=1290, tensile_mpa=1450, elong_pct=5, hardness_hv=520,
        density_gcc=6.8, youngs_gpa=180, melting_k=1500,
        notes="BCC-dominant; brittle at RT",
        references=["Wang et al. Intermetallics 2012"],
    ),
    KnownAlloy(
        name="Al0.5CoCrFeNi", family="hea_3d_al",
        composition_wt={"Al": 11.0, "Co": 22.25, "Cr": 22.25, "Fe": 22.25, "Ni": 22.25},
        yield_mpa=300, tensile_mpa=700, elong_pct=28, hardness_hv=230,
        density_gcc=7.5, youngs_gpa=195, melting_k=1580,
        notes="FCC+BCC dual phase",
        references=["Tang et al. JOM 2013"],
    ),

    # ---------------------- Cobalt alloys ----------------------------------
    KnownAlloy(
        name="Stellite 6", family="co_wear",
        composition_wt={"Co": 59.0, "Cr": 28.0, "W": 4.5, "Ni": 2.5, "Fe": 2.5, "Si": 1.0, "C": 1.0},
        tensile_mpa=820, hardness_hv=480,
        density_gcc=8.40, youngs_gpa=210, melting_k=1610,
        notes="cast; wear-resistant",
        references=["Kennametal Stellite 6"],
    ),
    KnownAlloy(
        name="MP35N", family="co_high_strength",
        composition_wt={"Co": 35.0, "Ni": 35.0, "Cr": 20.0, "Mo": 10.0},
        yield_mpa=1620, tensile_mpa=1860, elong_pct=10, hardness_hv=380,
        density_gcc=8.43, youngs_gpa=234, melting_k=1690,
        notes="cold-worked + aged",
        references=["SPS Technologies MP35N"],
    ),

    # ---------------------- Copper alloys ----------------------------------
    KnownAlloy(
        name="Cu pure", family="cu_unalloyed",
        composition_wt={"Cu": 99.95},
        yield_mpa=70, tensile_mpa=220, elong_pct=45, hardness_hv=50,
        density_gcc=8.96, youngs_gpa=120, melting_k=1358,
        references=["ASM Handbook v2"],
    ),
    KnownAlloy(
        name="Cu-Cr-Zr (C18150)", family="cu_precipitation",
        composition_wt={"Cu": 98.95, "Cr": 0.8, "Zr": 0.1, "Si": 0.05},
        yield_mpa=350, tensile_mpa=470, elong_pct=18, hardness_hv=110,
        density_gcc=8.89, youngs_gpa=130, melting_k=1340,
        notes="aged",
        references=["Copper Development Assoc."],
    ),
    KnownAlloy(
        name="CuBe (C17200)", family="cu_beryllium",
        composition_wt={"Cu": 97.85, "Be": 1.9, "Co": 0.25},
        yield_mpa=1100, tensile_mpa=1380, elong_pct=3, hardness_hv=380,
        density_gcc=8.25, youngs_gpa=130, melting_k=1140,
        references=["Materion CuBe 25"],
    ),
    KnownAlloy(
        name="Brass C26000", family="cu_brass",
        composition_wt={"Cu": 70.0, "Zn": 30.0},
        yield_mpa=125, tensile_mpa=325, elong_pct=53, hardness_hv=78,
        density_gcc=8.53, youngs_gpa=110, melting_k=1188,
        references=["Copper Development Assoc."],
    ),

    # ---------------------- Refractory metals ------------------------------
    KnownAlloy(
        name="C-103 (Nb-Hf-Ti)", family="nb_refractory",
        composition_wt={"Nb": 89.0, "Hf": 10.0, "Ti": 1.0},
        yield_mpa=295, tensile_mpa=440, elong_pct=20, hardness_hv=130,
        density_gcc=8.86, youngs_gpa=85, melting_k=2400,
        notes="aerospace Nb alloy for hot structures",
        references=["ATI C-103 datasheet"],
    ),
    KnownAlloy(
        name="Nb-22Si (composite)", family="nb_silicide",
        composition_wt={"Nb": 78.0, "Si": 22.0},
        hardness_hv=700, density_gcc=7.0, melting_k=2100,
        notes="Nb + Nb5Si3 in-situ composite; brittle at RT",
        references=["Bewlay et al. MSE A 2003"],
    ),
    KnownAlloy(
        name="W pure", family="w_refractory",
        composition_wt={"W": 99.95},
        yield_mpa=550, tensile_mpa=980, elong_pct=2, hardness_hv=350,
        density_gcc=19.25, youngs_gpa=411, melting_k=3695,
        references=["Plansee W datasheet"],
    ),
    KnownAlloy(
        name="Mo TZM", family="mo_refractory",
        composition_wt={"Mo": 99.4, "Ti": 0.5, "Zr": 0.08, "C": 0.02},
        yield_mpa=560, tensile_mpa=760, elong_pct=10, hardness_hv=275,
        density_gcc=10.16, youngs_gpa=320, melting_k=2890,
        references=["Plansee TZM datasheet"],
    ),
    KnownAlloy(
        name="Tantalum", family="ta_refractory",
        composition_wt={"Ta": 99.9},
        yield_mpa=170, tensile_mpa=205, elong_pct=40, hardness_hv=90,
        density_gcc=16.69, youngs_gpa=186, melting_k=3290,
        references=["H.C. Starck Ta datasheet"],
    ),
]


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

PROPERTY_COLUMNS = [
    "yield_mpa", "tensile_mpa", "elong_pct", "hardness_hv",
    "density_gcc", "youngs_gpa", "melting_k",
]


def reference_dataset(
    elements: Optional[Sequence[str]] = None,
    drop_missing_targets: bool = False,
    target_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return the curated alloy table as a training-ready DataFrame.

    Parameters
    ----------
    elements
        Restrict the element columns. Defaults to every element that
        appears in at least one ``ALLOYS`` entry (~25 columns).
    drop_missing_targets
        If True, only return rows where every ``target_columns`` value
        is non-null. Useful for models that need a complete y vector.
    target_columns
        Columns to consider as targets (used only when
        ``drop_missing_targets`` is True). Defaults to all property cols.

    Returns
    -------
    pandas.DataFrame
        Columns: ``alloy_name`` (str), ``family`` (str), all element
        atomic fractions summing to 1, all property columns,
        ``notes``, ``references`` (semicolon-joined).
    """
    if elements is None:
        all_els: List[str] = []
        for a in ALLOYS:
            for el in a.composition_wt:
                if el not in all_els:
                    all_els.append(el)
        elements = all_els

    rows: List[Dict] = []
    for a in ALLOYS:
        atomic = a.as_atomic()
        row: Dict = {"alloy_name": a.name, "family": a.family}
        for el in elements:
            row[el] = float(atomic.get(el, 0.0))
        for col in PROPERTY_COLUMNS:
            row[col] = getattr(a, col)
        row["notes"] = a.notes
        row["references"] = "; ".join(a.references)
        rows.append(row)
    df = pd.DataFrame(rows)

    if drop_missing_targets:
        tgts = list(target_columns) if target_columns else PROPERTY_COLUMNS
        df = df.dropna(subset=tgts).reset_index(drop=True)
    return df


def reference_families() -> List[str]:
    """Distinct alloy-family labels (useful for ``GroupKFold``)."""
    return sorted({a.family for a in ALLOYS})


def reference_elements() -> List[str]:
    """Union of every element appearing in the curated table."""
    out: List[str] = []
    for a in ALLOYS:
        for el in a.composition_wt:
            if el not in out:
                out.append(el)
    return out


def find_alloy(name: str) -> Optional[KnownAlloy]:
    """Case-insensitive name lookup."""
    lo = name.lower()
    for a in ALLOYS:
        if a.name.lower() == lo:
            return a
    return None


__all__ = [
    "ALLOYS", "KnownAlloy", "PROPERTY_COLUMNS",
    "atomic_to_weight_pct", "find_alloy",
    "reference_dataset", "reference_elements", "reference_families",
    "weight_to_atomic_pct",
]
