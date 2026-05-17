"""Tests for HTS compound screening + OQMD client."""

from __future__ import annotations

import pandas as pd
import pytest

from core.alloyforge.hts_screening import (
    HOSTS,
    NB_HOST_COMPOUNDS,
    ScoreWeights,
    host_plus_precipitate_composition,
    rank_compounds,
    score_coherency,
    score_compound,
    score_stability,
    score_tie_line,
)
from core.alloyforge.oqmd_client import (
    _parse_formula_stoich,
    query_oqmd,
    to_known_compounds,
)


# ---------------------------------------------------------------------------
# Bundled DB integrity
# ---------------------------------------------------------------------------

def test_bundled_db_nonempty():
    assert len(NB_HOST_COMPOUNDS) >= 15


def test_bundled_db_required_fields():
    for c in NB_HOST_COMPOUNDS:
        assert c.formula
        assert c.elements
        assert c.formation_energy_per_atom_ev <= 0.0, (
            f"{c.formula}: ΔH > 0 — unstable compound shouldn't be in DB"
        )
        assert c.lattice_a > 0
        assert c.volume_per_atom > 0


# ---------------------------------------------------------------------------
# Individual scorers
# ---------------------------------------------------------------------------

def test_tie_line_score_documented():
    nb5si3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "Nb5Si3-alpha")
    s, det = score_tie_line(nb5si3, HOSTS["Nb"])
    assert s == 1.0
    assert det["has_documented_tie_line"] is True


def test_tie_line_score_no_match():
    # NbSi2 doesn't have a direct tie line with Nb in our DB
    nbsi2 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "NbSi2")
    s, _ = score_tie_line(nbsi2, HOSTS["Nb"])
    # Nb IS in the compound, but tie line is not documented → 0.5
    assert s == 0.5


def test_tie_line_score_no_host_in_compound():
    nb5si3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "Nb5Si3-alpha")
    s, _ = score_tie_line(nb5si3, HOSTS["Al"])
    assert s == 0.0


def test_stability_score_negative_dh_in_range():
    nbc = next(c for c in NB_HOST_COMPOUNDS if c.formula == "NbC")
    s, det = score_stability(nbc)
    assert 0.9 < s <= 1.0
    assert det["delta_h_per_atom_ev"] == -0.74


def test_stability_score_clipped():
    # Weakly stable compound
    nbcr2 = next(c for c in NB_HOST_COMPOUNDS
                  if c.formula == "NbCr2-C15")
    s, _ = score_stability(nbcr2)
    assert 0 < s < 0.3


def test_coherency_score_modular_match():
    """Nb5Si3 (a=6.57) should be ~coherent with Nb (a=3.30) via k=2."""
    nb5si3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "Nb5Si3-alpha")
    s, det = score_coherency(nb5si3, HOSTS["Nb"])
    assert det["best_multiple_k"] == 2
    assert det["lattice_a_mismatch_pct_modular"] < 2.0
    assert s > 0.7


def test_coherency_score_poor_match():
    """NbAl3 (a=3.84) doesn't match Nb (a=3.30) under k=1,2,3,4."""
    nbal3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "NbAl3")
    s, det = score_coherency(nbal3, HOSTS["Nb"])
    assert det["lattice_a_mismatch_pct_modular"] > 10
    assert s < 0.5


# ---------------------------------------------------------------------------
# Compound score + ranking
# ---------------------------------------------------------------------------

def test_score_compound_combines_three_descriptors():
    nb5si3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "Nb5Si3-alpha")
    s = score_compound(nb5si3, HOSTS["Nb"])
    assert s.formula == "Nb5Si3-alpha"
    assert 0 <= s.total <= 1.0
    # All three descriptors should be >0 for this compound
    assert s.tie_line_score > 0
    assert s.stability_score > 0
    assert s.coherency_score > 0


def test_rank_compounds_returns_sorted_df():
    df = rank_compounds(host="Nb")
    assert isinstance(df, pd.DataFrame)
    assert (df["total"].diff().dropna() <= 0).all(), "not sorted desc"
    assert "formula" in df.columns
    assert "tie_line" in df.columns and "stability" in df.columns


def test_rank_compounds_required_elements_filter():
    df = rank_compounds(host="Nb", required_elements=["Nb", "Si"])
    assert len(df) >= 4  # Nb5Si3 α/β/γ + Nb3Si + NbSi2 + ternaries
    for _, row in df.iterrows():
        # Either Nb-Si binary or Nb-Si-X ternary
        els = row["elements"].split(",")
        assert "Nb" in els and "Si" in els


def test_rank_compounds_forbidden_elements_filter():
    df = rank_compounds(host="Nb", forbidden_elements=["C"])
    assert "NbC" not in df["formula"].tolist()


def test_rank_compounds_min_tie_line_filter():
    df = rank_compounds(host="Nb", min_tie_line_score=0.6)
    # Every row should have tie_line score ≥ 0.6
    assert (df["tie_line"] >= 0.6).all()


def test_rank_compounds_top_k():
    df = rank_compounds(host="Nb", top_k=3)
    assert len(df) == 3


def test_rank_compounds_unknown_host_raises():
    with pytest.raises(KeyError):
        rank_compounds(host="Pu")


# ---------------------------------------------------------------------------
# Host + precipitate composition
# ---------------------------------------------------------------------------

def test_host_plus_precipitate_sums_to_one():
    nb5si3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "Nb5Si3-alpha")
    comp = host_plus_precipitate_composition(HOSTS["Nb"], nb5si3, 0.20)
    assert sum(comp.values()) == pytest.approx(1.0, abs=1e-6)
    assert comp["Si"] > 0
    assert comp["Nb"] > comp["Si"]


def test_host_plus_precipitate_zero_fraction():
    nbal3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "NbAl3")
    comp = host_plus_precipitate_composition(HOSTS["Nb"], nbal3, 0.0)
    assert comp["Nb"] == pytest.approx(1.0)
    assert comp.get("Al", 0.0) == pytest.approx(0.0)


def test_host_plus_precipitate_bounds():
    nb5si3 = next(c for c in NB_HOST_COMPOUNDS if c.formula == "Nb5Si3-alpha")
    with pytest.raises(ValueError):
        host_plus_precipitate_composition(HOSTS["Nb"], nb5si3, 1.5)


# ---------------------------------------------------------------------------
# OQMD client
# ---------------------------------------------------------------------------

def test_oqmd_query_returns_empty_on_network_failure(monkeypatch):
    """OQMD must never crash the platform — empty df on any error."""
    import httpx

    def bad_get(*_a, **_kw):
        raise httpx.ConnectError("boom")
    monkeypatch.setattr("core.alloyforge.oqmd_client.httpx.get", bad_get)
    df = query_oqmd(elements=["Nb", "Si"])
    assert df.empty
    assert "formula" in df.columns


def test_oqmd_query_parses_response(monkeypatch):
    payload = {"data": [
        {
            "name": "Nb5Si3",
            "element_set": ["Nb", "Si"],
            "natoms": 32,
            "spacegroup": {"symbol": "I4/mcm"},
            "delta_e": -0.65,
            "stability": 0.0,
            "unit_cell": {"volume_per_atom": 16.0, "lattice_a": 6.57},
        }
    ]}

    class MockResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload

    monkeypatch.setattr(
        "core.alloyforge.oqmd_client.httpx.get",
        lambda *_a, **_kw: MockResp(),
    )
    df = query_oqmd(elements=["Nb", "Si"])
    assert len(df) == 1
    assert df.iloc[0]["formula"] == "Nb5Si3"
    assert df.iloc[0]["delta_h_per_atom_ev"] == -0.65
    assert df.iloc[0]["space_group"] == "I4/mcm"


def test_to_known_compounds_from_oqmd_df():
    df = pd.DataFrame([{
        "formula": "Nb3Si",
        "elements": "Nb,Si",
        "n_atoms": 32,
        "space_group": "P42/n",
        "delta_h_per_atom_ev": -0.51,
        "stability": 0.0,
        "volume_per_atom": 18.10,
        "lattice_a": 10.22,
        "source": "oqmd",
    }])
    compounds = to_known_compounds(df)
    assert len(compounds) == 1
    c = compounds[0]
    assert c.formula == "Nb3Si"
    assert "Nb" in c.elements and "Si" in c.elements
    assert c.stoichiometry == {"Nb": 3, "Si": 1}


def test_parse_formula_stoich_simple():
    assert _parse_formula_stoich("Nb5Si3") == {"Nb": 5, "Si": 3}
    assert _parse_formula_stoich("NbAl3") == {"Nb": 1, "Al": 3}
    assert _parse_formula_stoich("Cr2Nb") == {"Cr": 2, "Nb": 1}
    assert _parse_formula_stoich("NbC") == {"Nb": 1, "C": 1}


# ---------------------------------------------------------------------------
# FastAPI router smoke
# ---------------------------------------------------------------------------

def test_hts_router_endpoints_smoke():
    from fastapi.testclient import TestClient
    from backend.main import app

    c = TestClient(app)
    r = c.get("/api/v1/hts/hosts")
    assert r.status_code == 200
    assert "Nb" in r.json()

    r = c.get("/api/v1/hts/compounds")
    assert r.status_code == 200
    assert r.json()["n"] >= 15

    r = c.post("/api/v1/hts/rank", json={
        "host": "Nb",
        "weight_tie_line": 1.0,
        "weight_stability": 1.0,
        "weight_coherency": 1.0,
        "required_elements": ["Nb", "Si"],
        "top_k": 3,
    })
    assert r.status_code == 200
    js = r.json()
    assert js["n"] == 3
    assert all("Nb" in row["elements"] and "Si" in row["elements"]
               for row in js["ranking"])

    r = c.post("/api/v1/hts/compound-mix", json={
        "host": "Nb",
        "compound_formula": "Nb5Si3-alpha",
        "precipitate_atomic_fraction": 0.10,
    })
    assert r.status_code == 200
    comp = r.json()["composition_atomic_fraction"]
    assert sum(comp.values()) == pytest.approx(1.0, abs=1e-6)
    assert "Si" in comp


def test_hts_router_404_unknown_compound():
    from fastapi.testclient import TestClient
    from backend.main import app

    c = TestClient(app)
    r = c.post("/api/v1/hts/compound-mix", json={
        "host": "Nb",
        "compound_formula": "NotARealCompound",
    })
    assert r.status_code == 404
