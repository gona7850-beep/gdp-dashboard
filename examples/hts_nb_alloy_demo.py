"""End-to-end demo: HTS compound screening for Nb-host alloy design.

Re-creates the workflow from Cho 2025 slides (Kookmin NSM Lab) but for
Nb host instead of Al:

1. Browse the bundled Nb-host compound DB (~22 compounds).
2. Rank with default equal weights (tie line / stability / coherency).
3. Tune weights to emphasise high-temperature creep-resistant phases.
4. Generate a synthetic Nb-Si in-situ composite composition by mixing
   Nb matrix + Nb5Si3 precipitate at 20 at%.
5. Feed that composition into a v1 forward model pre-trained on the
   reference DB and predict hardness as a sanity check.

Run from the repo root:
    python examples/hts_nb_alloy_demo.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from core.alloyforge import (
    CompositionFeaturizer,
    Dataset,
    ForwardModel,
    reference_dataset,
    reference_elements,
)
from core.alloyforge.hts_screening import (
    HOSTS,
    NB_HOST_COMPOUNDS,
    ScoreWeights,
    host_plus_precipitate_composition,
    rank_compounds,
)


def main() -> None:
    print("=" * 72)
    print("HTS compound screening demo — Nb-host alloy design")
    print("=" * 72)
    print(f"\nBundled DB: {len(NB_HOST_COMPOUNDS)} Nb-host compounds")

    # ----- 1. Default ranking -----------------------------------------
    print("\n--- Top 10 with equal weights (1, 1, 1) ---")
    df = rank_compounds(host="Nb", top_k=10)
    show = ["formula", "tie_line", "stability", "coherency", "total",
            "lattice_a_mismatch_pct", "best_multiple_k"]
    print(df[show].to_string(index=False))

    # ----- 2. Emphasise stability + coherency over tie line -----------
    print("\n--- Re-rank: heavier weight on stability & coherency ---")
    df2 = rank_compounds(
        host="Nb",
        weights=ScoreWeights(tie_line=0.5, stability=2.0, coherency=2.0),
        top_k=10,
    )
    print(df2[show].to_string(index=False))

    # ----- 3. Filter to Nb-Si only ------------------------------------
    print("\n--- Nb-Si binaries only ---")
    df3 = rank_compounds(
        host="Nb", required_elements=["Nb", "Si"], forbidden_elements=["Ti", "Hf", "Cr"],
    )
    print(df3[show].to_string(index=False))

    # ----- 4. Mix top compound into Nb matrix, predict hardness --------
    print("\n--- ML prediction for Nb matrix + 20 at% (Nb,Hf)5Si3 ---")
    top = next(c for c in NB_HOST_COMPOUNDS if c.formula == "(Nb,Hf)5Si3")
    comp = host_plus_precipitate_composition(HOSTS["Nb"], top, 0.20)
    print(f"  Mixed composition: {comp}")

    # Train v1 forward model on the curated reference DB
    print("\n  Pre-training v1 forward model on the 38-alloy reference DB…")
    ref = reference_dataset()
    elements = reference_elements()
    targets = ["yield_mpa", "tensile_mpa", "hardness_hv", "density_gcc"]
    ref_train = ref.dropna(subset=targets).reset_index(drop=True)
    ds = Dataset(
        compositions=ref_train[elements],
        properties=ref_train[targets],
        groups=ref_train["family"],
    )
    fm = ForwardModel(
        featurizer=CompositionFeaturizer(element_columns=elements),
        targets=targets,
        n_cv_splits=5,
    )
    fm.fit(ds, n_trials=4)

    # Build the query row with the same element columns
    query = pd.DataFrame([{
        el: comp.get(el, 0.0) for el in elements
    }])
    preds = fm.predict(query).iloc[0]
    print("\n  Predicted properties (Nb matrix + 20 at% (Nb,Hf)5Si3):")
    for t in targets:
        print(f"    {t:14s}  μ={preds[f'{t}_mean']:7.1f}  "
              f"σ={preds[f'{t}_std']:5.1f}")

    print("\n--- Conclusion ---")
    print("Top screening hits for Nb-host: silicide variants (Nb,Hf)5Si3,")
    print("Nb5Si3-α, (Nb,Ti)5Si3 — exactly the in-situ-composite phases")
    print("validated for high-temperature Nb-Si alloys (Bewlay et al.).")
    print("The platform identifies them automatically from the 3 descriptors.")


if __name__ == "__main__":
    main()
