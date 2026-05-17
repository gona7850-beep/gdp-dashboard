"""Tests for the curated alloy reference database + unit conversions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.alloyforge.data_ingestion import (
    convert_value,
    flag_outliers,
    infer_units,
    merge_datasets,
    normalize_composition,
    normalize_units,
)
from core.alloyforge.reference_data import (
    ALLOYS,
    PROPERTY_COLUMNS,
    atomic_to_weight_pct,
    find_alloy,
    reference_dataset,
    reference_elements,
    reference_families,
    weight_to_atomic_pct,
)


# ---------------------------------------------------------------------------
# Reference dataset
# ---------------------------------------------------------------------------

def test_reference_dataset_loads():
    df = reference_dataset()
    assert len(df) >= 35
    assert "alloy_name" in df.columns
    assert "family" in df.columns


def test_reference_compositions_sum_to_one():
    df = reference_dataset()
    els = reference_elements()
    sums = df[els].sum(axis=1).to_numpy()
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)


def test_property_completeness_above_90pct():
    df = reference_dataset()
    for col in PROPERTY_COLUMNS:
        completeness = df[col].notna().mean()
        assert completeness > 0.85, (
            f"{col} only {completeness:.0%} complete; "
            "should be ≥85 % for the reference table"
        )


def test_find_alloy_case_insensitive():
    a = find_alloy("Ti-6Al-4V")
    assert a is not None
    assert a.family == "ti_alpha_beta"
    assert find_alloy("ti-6al-4v") is a


def test_reference_families_are_groupable():
    fams = reference_families()
    assert len(fams) >= 15
    df = reference_dataset()
    # At least one family should have >1 member (otherwise GroupKFold is useless)
    counts = df["family"].value_counts()
    assert counts.max() >= 2


def test_drop_missing_targets_filters_rows():
    df_all = reference_dataset(drop_missing_targets=False)
    df_drop = reference_dataset(
        drop_missing_targets=True,
        target_columns=["yield_mpa", "tensile_mpa"],
    )
    assert len(df_drop) <= len(df_all)
    for col in ["yield_mpa", "tensile_mpa"]:
        assert df_drop[col].notna().all()


# ---------------------------------------------------------------------------
# Composition conversion
# ---------------------------------------------------------------------------

def test_weight_to_atomic_ti6al4v():
    # Ti-6Al-4V: 90/6/4 wt% → 86.2/10.2/3.6 atomic % (within rounding)
    atomic = weight_to_atomic_pct({"Ti": 90.0, "Al": 6.0, "V": 4.0})
    assert atomic["Ti"] == pytest.approx(0.862, abs=1e-3)
    assert atomic["Al"] == pytest.approx(0.102, abs=1e-3)
    assert atomic["V"] == pytest.approx(0.036, abs=1e-3)
    assert sum(atomic.values()) == pytest.approx(1.0, abs=1e-9)


def test_atomic_to_weight_roundtrip():
    wt = {"Fe": 70.0, "Cr": 18.0, "Ni": 9.0, "Mn": 2.0, "Si": 1.0}
    atomic = weight_to_atomic_pct(wt)
    wt_back = atomic_to_weight_pct(atomic)
    # Should reproduce the original weight ratios
    total = sum(wt.values())
    for el in wt:
        assert wt_back[el] == pytest.approx(wt[el] / total, abs=1e-6)


def test_weight_to_atomic_unknown_element_raises():
    with pytest.raises(KeyError):
        weight_to_atomic_pct({"Xx": 50.0, "Fe": 50.0})


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ksi,mpa", [(100, 689.476), (200, 1378.952)])
def test_ksi_to_mpa(ksi, mpa):
    assert convert_value(ksi, "ksi", "mpa") == pytest.approx(mpa, abs=1e-2)


def test_gpa_to_mpa():
    assert convert_value(1.5, "gpa", "mpa") == pytest.approx(1500.0)


@pytest.mark.parametrize("c,k", [(0, 273.15), (100, 373.15), (-40, 233.15)])
def test_celsius_to_kelvin(c, k):
    assert convert_value(c, "celsius", "k") == pytest.approx(k)


@pytest.mark.parametrize("hrc,hv_expected", [(25, 253), (35, 331), (45, 466),
                                              (55, 695), (60, 879), (65, 1132)])
def test_hrc_to_hv_astm_e140(hrc, hv_expected):
    # ASTM E140 table values; allow ±2 HV tolerance from interpolation
    got = convert_value(hrc, "hrc", "hv")
    assert abs(got - hv_expected) < 2


def test_hb_to_hv_monotonic():
    a = convert_value(100, "hb", "hv")
    b = convert_value(300, "hb", "hv")
    c = convert_value(500, "hb", "hv")
    assert a < b < c


def test_convert_same_unit_is_identity():
    assert convert_value(123.4, "mpa", "mpa") == 123.4


def test_convert_unknown_pair_raises():
    with pytest.raises(ValueError):
        convert_value(10, "weird", "alsoweird")


# ---------------------------------------------------------------------------
# Unit auto-detection
# ---------------------------------------------------------------------------

def test_infer_units_from_column_names():
    df = pd.DataFrame({
        "yield_MPa": [300, 500, 700],
        "tensile_ksi": [80, 120, 150],
        "hardness_HRC": [25, 35, 45],
        "Fe": [0.7, 0.6, 0.5],
        "TempC": [25, 100, 500],
    })
    units = infer_units(df)
    assert units["yield_MPa"] == "mpa"
    assert units["tensile_ksi"] == "ksi"
    assert units["hardness_HRC"] == "hrc"
    assert units["Fe"] == "fraction"


def test_normalize_units_ksi_to_mpa():
    df = pd.DataFrame({"uts_ksi": [100.0, 200.0]})
    normed = normalize_units(df, {"uts_ksi": "ksi"})
    assert normed["uts_ksi"].iloc[0] == pytest.approx(689.476, abs=1e-2)


# ---------------------------------------------------------------------------
# Composition normalisation
# ---------------------------------------------------------------------------

def test_normalize_composition_weight_to_atomic():
    df = pd.DataFrame({"Ti": [90.0], "Al": [6.0], "V": [4.0]})
    out = normalize_composition(df, ["Ti", "Al", "V"], "weight_pct")
    assert out["Ti"].iloc[0] == pytest.approx(0.862, abs=1e-3)
    assert out["Al"].iloc[0] + out["Ti"].iloc[0] + out["V"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_normalize_composition_atomic_pct():
    df = pd.DataFrame({"Fe": [70.0, 60.0], "Ni": [30.0, 40.0]})
    out = normalize_composition(df, ["Fe", "Ni"], "atomic_pct")
    np.testing.assert_allclose(out[["Fe", "Ni"]].sum(axis=1), 1.0)


def test_normalize_composition_auto_detect():
    df = pd.DataFrame({"Fe": [70.0], "Ni": [30.0]})
    out = normalize_composition(df, ["Fe", "Ni"], "auto")
    assert out["Fe"].iloc[0] + out["Ni"].iloc[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------

def test_flag_outliers_marks_extreme_only():
    df = pd.DataFrame({"y": [10, 11, 12, 9, 10.5, 11, 200, 9.8, 10.1, 10.0]})
    flagged = flag_outliers(df, ["y"], z_threshold=4.0)
    mask = flagged["is_outlier"].to_numpy()
    assert mask[6] is np.True_ or mask[6] == True  # noqa: E712
    assert mask[:6].sum() == 0
    assert mask[7:].sum() == 0


def test_flag_outliers_no_op_when_clean():
    df = pd.DataFrame({"y": [10, 11, 12, 9, 10.5]})
    flagged = flag_outliers(df, ["y"], z_threshold=4.0)
    assert not flagged["is_outlier"].any()


# ---------------------------------------------------------------------------
# Merge multiple sources
# ---------------------------------------------------------------------------

def test_merge_datasets_adds_source_column():
    a = pd.DataFrame({"Fe": [0.7, 0.6], "Ni": [0.3, 0.4],
                      "yield_mpa": [300, 400]})
    b = pd.DataFrame({"Fe": [0.5], "Ni": [0.5], "yield_mpa": [350]})
    merged, summary = merge_datasets(
        sources={"paper_A": a, "paper_B": b},
        element_columns=["Fe", "Ni"],
        target_columns=["yield_mpa"],
    )
    assert "source" in merged.columns
    assert set(merged["source"]) == {"paper_A", "paper_B"}
    assert summary.n_rows_in == 3


def test_merge_datasets_deduplicates():
    a = pd.DataFrame({"Fe": [0.7], "Ni": [0.3], "yield_mpa": [300]})
    b = pd.DataFrame({"Fe": [0.7], "Ni": [0.3], "yield_mpa": [300]})
    merged, summary = merge_datasets(
        sources={"a": a, "b": b},
        element_columns=["Fe", "Ni"],
        target_columns=["yield_mpa"],
        dedup=True,
    )
    assert len(merged) == 1
    assert summary.duplicated_dropped == 1
