"""Tests for phase fraction + HTS descriptor + active-learning planner."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Microstructure (phase fraction) features
# ---------------------------------------------------------------------------

from core.alloyforge.microstructure_features import (
    PHASE_FAMILIES,
    PhaseFractionFeaturizer,
    load_cleaned_nb_si,
    split_columns,
)


def test_phase_featurizer_shape_and_columns():
    feat = PhaseFractionFeaturizer()
    df = pd.DataFrame({
        "Nbss": [1, 0, 1],
        "Nb5Si3": [1, 1, 0],
        "α_Nb5Si3": [0, 1, 0],
        "NbCr2": [0, 0, 1],
        "Nb3Sn": [0, 1, 0],
    })
    X = feat.transform(df)
    assert X.shape == (3, 12)
    assert "has_Nbss" in X.columns
    assert "has_Nb5Si3_family" in X.columns
    assert X["has_Nbss"].tolist() == [1, 0, 1]
    assert X["has_Nb5Si3_family"].tolist() == [1, 1, 0]
    assert X["n_phases"].tolist() == [2, 3, 2]


def test_phase_featurizer_missing_columns_zeroed():
    """Featurizer must survive when many phase columns are absent."""
    feat = PhaseFractionFeaturizer()
    df = pd.DataFrame({"Nbss": [1, 1]})
    X = feat.transform(df)
    assert X.shape == (2, 12)
    # Every family except Nbss should be zero
    for col in feat.feature_names:
        if col == "has_Nbss":
            assert (X[col] == 1).all()


def test_phase_featurizer_on_real_dataset():
    df = load_cleaned_nb_si("data/nb_si/nb_silicide_cleaned.csv")
    assert len(df) > 500
    els, phases, props = split_columns(df)
    assert len(els) > 10
    assert len(phases) > 40
    assert "Vickers_hardness_(Hv)" in props
    X = PhaseFractionFeaturizer().transform(df)
    # On the real Nb-Si DB at least 90% of rows should have Nbss
    assert X["has_Nbss"].mean() > 0.9


def test_phase_families_cover_all_documented_phases():
    df = load_cleaned_nb_si("data/nb_si/nb_silicide_cleaned.csv")
    _, phase_cols, _ = split_columns(df)
    covered = set()
    for members in PHASE_FAMILIES.values():
        covered.update(members)
    from core.alloyforge.microstructure_features import EUTECTIC_COLUMNS
    covered.update(EUTECTIC_COLUMNS)
    # At least 90% of the dataset's phase columns should be in some family
    mapped = [p for p in phase_cols if p in covered]
    assert len(mapped) / len(phase_cols) > 0.85, (
        f"only {len(mapped)}/{len(phase_cols)} phases classified"
    )


# ---------------------------------------------------------------------------
# HTS descriptor
# ---------------------------------------------------------------------------

from core.alloyforge.hts_descriptor import HTSScoreFeaturizer


def test_hts_descriptor_shape():
    feat = HTSScoreFeaturizer(host_symbol="Nb")
    df = pd.DataFrame({"Nb": [0.7, 0.5], "Si": [0.2, 0.3], "Ti": [0.1, 0.2]})
    X = feat.transform(df)
    assert X.shape == (2, 5)
    assert list(X.columns) == feat.feature_names


def test_hts_descriptor_finds_silicide_for_nbsi():
    feat = HTSScoreFeaturizer(host_symbol="Nb")
    df = pd.DataFrame({"Nb": [0.75], "Si": [0.25]})
    X = feat.transform(df)
    # Many Nb-Si compounds exist in DB, so we should find ≥2 matches
    assert X["hts_n_matching_compounds"].iloc[0] >= 2
    # Best score should be high (silicides score well)
    assert X["hts_max_total"].iloc[0] > 0.5


def test_hts_descriptor_pure_nb_has_no_matches():
    feat = HTSScoreFeaturizer(host_symbol="Nb")
    df = pd.DataFrame({"Nb": [1.0], "Si": [0.0]})
    X = feat.transform(df)
    # Pure Nb has no binary compounds among element set {Nb}
    assert X["hts_n_matching_compounds"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Active-learning planner
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model_with_dataset():
    pytest.importorskip("xgboost")
    pytest.importorskip("optuna")
    from core.alloyforge import CompositionFeaturizer, Dataset, ForwardModel

    rng = np.random.default_rng(0)
    comp = rng.dirichlet([2, 1, 1, 0.5], size=60)
    df = pd.DataFrame(comp, columns=["Nb", "Si", "Ti", "Cr"])
    df["hv"] = 200 + 800 * df["Si"] + 400 * df["Cr"] + rng.normal(0, 10, 60)
    df["uts"] = 400 + 1200 * df["Si"] + 500 * df["Cr"] + rng.normal(0, 20, 60)
    ds = Dataset(
        compositions=df[["Nb", "Si", "Ti", "Cr"]],
        properties=df[["hv", "uts"]],
    )
    fm = ForwardModel(
        featurizer=CompositionFeaturizer(
            element_columns=["Nb", "Si", "Ti", "Cr"]
        ),
        targets=["hv", "uts"], n_cv_splits=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm.fit(ds, n_trials=3)
    return fm, ds


def test_planner_proposes_diverse_batch(small_model_with_dataset):
    from core.alloyforge.active_learning_planner import (
        ExperimentPlanner, PlannerWeights,
    )
    fm, ds = small_model_with_dataset
    planner = ExperimentPlanner(
        model=fm, weights=PlannerWeights(sigma=1.0, hts=0.0, doa=0.0),
    ).fit(ds)
    rng = np.random.default_rng(1)
    pool = pd.DataFrame(
        rng.dirichlet([2, 1, 1, 0.5], size=100),
        columns=["Nb", "Si", "Ti", "Cr"],
    )
    picks = planner.propose(pool, targets=["hv", "uts"], batch_size=5)
    assert len(picks) == 5
    # Each pick should have rank 1-5
    assert picks["rank"].tolist() == [1, 2, 3, 4, 5]
    # σ_avg column present and finite
    assert picks["sigma_avg"].notna().all()
    # Diversity: pairwise distances among picks should not all be near zero
    comp = picks[["Nb", "Si", "Ti", "Cr"]].to_numpy()
    n = len(comp)
    dists = [np.linalg.norm(comp[i] - comp[j])
             for i in range(n) for j in range(i + 1, n)]
    assert min(dists) > 0.01


def test_planner_acq_combines_three_signals(small_model_with_dataset):
    from core.alloyforge.active_learning_planner import (
        ExperimentPlanner, PlannerWeights,
    )
    fm, ds = small_model_with_dataset
    planner = ExperimentPlanner(
        model=fm, weights=PlannerWeights(sigma=1.0, hts=1.0, doa=1.0),
    ).fit(ds)
    rng = np.random.default_rng(2)
    pool = pd.DataFrame(
        rng.dirichlet([2, 1, 1, 0.5], size=80),
        columns=["Nb", "Si", "Ti", "Cr"],
    )
    picks = planner.propose(pool, targets=["hv", "uts"], batch_size=3)
    for col in ["sigma_avg", "hts_score", "doa_score", "acq"]:
        assert col in picks.columns
        assert picks[col].notna().all()


# ---------------------------------------------------------------------------
# Streamlit dashboard module loads
# ---------------------------------------------------------------------------

def test_dashboard_module_imports():
    import py_compile
    from pathlib import Path
    target = Path("app/pages/0_통합대시보드.py")
    py_compile.compile(str(target), doraise=True)
