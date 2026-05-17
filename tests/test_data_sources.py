"""Tests for external_data, llm_table_extractor, accuracy_report,
and the /api/v1/data router. External API calls are mocked so tests
never depend on network."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# External data clients (mocked)
# ---------------------------------------------------------------------------

from core.alloyforge.external_data import (
    materials_project_summary,
    provider_status,
    search_arxiv,
    search_crossref,
    search_openalex,
)


def test_provider_status_keys_present():
    s = provider_status()
    assert s.keys() == {"httpx_available", "openalex", "arxiv",
                         "crossref", "materials_project"}


def test_openalex_returns_empty_on_network_error(monkeypatch):
    """When the HTTP call fails, the client must return an empty df,
    not raise — the rest of the platform depends on this."""
    import httpx
    def bad_get(*_a, **_kw):
        raise httpx.ConnectError("boom")
    monkeypatch.setattr("core.alloyforge.external_data.httpx.get", bad_get)
    df = search_openalex("anything")
    assert df.empty
    assert list(df.columns) == ["title", "authors", "year", "venue",
                                  "doi", "url", "abstract", "source"]


def test_openalex_parses_valid_response(monkeypatch):
    payload = {
        "results": [{
            "title": "A great paper",
            "authorships": [{"author": {"display_name": "Alice"}}],
            "publication_year": 2024,
            "host_venue": {"display_name": "Acta Materialia"},
            "doi": "https://doi.org/10.1/test",
            "primary_location": {"landing_page_url": "https://example.org/x"},
            "abstract_inverted_index": {"high": [0], "entropy": [1], "alloy": [2]},
        }]
    }
    class MockResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload
    monkeypatch.setattr(
        "core.alloyforge.external_data.httpx.get",
        lambda *_a, **_kw: MockResp(),
    )
    df = search_openalex("hea")
    assert len(df) == 1
    assert df.iloc[0]["title"] == "A great paper"
    assert df.iloc[0]["year"] == 2024
    assert df.iloc[0]["doi"] == "10.1/test"
    assert "high entropy alloy" in df.iloc[0]["abstract"]
    assert df.iloc[0]["source"] == "openalex"


def test_arxiv_parses_atom_feed(monkeypatch):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Cool HEA Study</title>
        <summary>We made an alloy and tested it.</summary>
        <published>2024-03-15T00:00:00Z</published>
        <author><name>Bob</name></author>
        <link rel="alternate" href="https://arxiv.org/abs/2403.xxxx"/>
        <link title="pdf" href="https://arxiv.org/pdf/2403.xxxx"/>
      </entry>
    </feed>"""
    class MockResp:
        status_code = 200
        text = xml
        def raise_for_status(self): pass
    monkeypatch.setattr(
        "core.alloyforge.external_data.httpx.get",
        lambda *_a, **_kw: MockResp(),
    )
    df = search_arxiv("hea")
    assert len(df) == 1
    assert df.iloc[0]["title"] == "Cool HEA Study"
    assert df.iloc[0]["year"] == 2024
    assert df.iloc[0]["source"] == "arxiv"


def test_crossref_parses_response(monkeypatch):
    payload = {"message": {"items": [{
        "title": ["Title here"],
        "author": [{"given": "Carol", "family": "Doe"}],
        "issued": {"date-parts": [[2023]]},
        "container-title": ["MSE-A"],
        "DOI": "10.test/abc",
        "URL": "https://doi.org/10.test/abc",
    }]}}
    class MockResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload
    monkeypatch.setattr(
        "core.alloyforge.external_data.httpx.get",
        lambda *_a, **_kw: MockResp(),
    )
    df = search_crossref("anything")
    assert df.iloc[0]["doi"] == "10.test/abc"
    assert df.iloc[0]["year"] == 2023


def test_materials_project_no_key_returns_empty(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    df = materials_project_summary(elements=["Fe", "Ni"])
    assert df.empty


# ---------------------------------------------------------------------------
# LLM table extractor
# ---------------------------------------------------------------------------

from core.alloyforge.llm_table_extractor import extract_alloy_table


def test_extract_heuristic_path_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text = ("AISI 304 contains Fe 71, Cr 18, Ni 9 by weight. "
            "Yield strength = 215 MPa. Hardness 200 HV. Elongation 70%.")
    df, rep = extract_alloy_table(text)
    assert rep.used_llm is False
    assert len(df) == 1
    row = df.iloc[0]
    assert row["yield_mpa"] == pytest.approx(215.0)
    assert row["hardness_hv"] == pytest.approx(200.0)
    # Composition should sum to ~1
    el_cols = [c for c in df.columns if c in ("Fe", "Ni", "Cr")]
    assert row[el_cols].sum() == pytest.approx(1.0, abs=1e-3)
    assert row["confidence"] == "low"


def test_extract_returns_empty_for_no_match():
    df, rep = extract_alloy_table("Lorem ipsum dolor sit amet.")
    assert df.empty
    assert rep.n_rows == 0


def test_extract_with_mocked_llm(monkeypatch):
    """Simulate Claude returning a structured JSON block."""
    json_response = json.dumps({
        "rows": [{
            "composition": {"Fe": 70, "Ni": 30},
            "composition_basis": "weight_pct",
            "properties": {"yield_mpa": 400, "tensile_mpa": 600},
            "alloy_name": "Test alloy",
            "process": "annealed",
            "confidence": "high",
            "warning": "",
        }]
    })
    class _Block:
        type = "text"
        text = json_response
    class _Resp:
        content = [_Block()]
    class _Msgs:
        def create(self, **kw): return _Resp()
    class _Client:
        messages = _Msgs()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    # Patch the Anthropic constructor
    with patch("core.alloyforge.llm_table_extractor.Anthropic",
               return_value=_Client()):
        df, rep = extract_alloy_table(
            "some text about Fe-Ni alloy",
            element_columns=["Fe", "Ni"],
            property_columns=["yield_mpa", "tensile_mpa"],
        )
    assert rep.used_llm is True
    assert len(df) == 1
    assert df.iloc[0]["yield_mpa"] == 400


# ---------------------------------------------------------------------------
# Accuracy report (end-to-end on a small synthetic dataset)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_v1_with_dataset():
    pytest.importorskip("xgboost")
    pytest.importorskip("optuna")
    from core.alloyforge.data_pipeline import CompositionFeaturizer, Dataset
    from core.alloyforge.forward_model import ForwardModel

    rng = np.random.default_rng(0)
    comp = rng.dirichlet([2, 1, 1, 0.5], size=80)
    df = pd.DataFrame(comp, columns=["Fe", "Ni", "Cr", "Mo"])
    df["yield_mpa"] = (200 + 600 * df["Mo"] + 400 * df["Cr"]
                       + rng.normal(0, 10, 80))
    df["tensile_mpa"] = (400 + 900 * df["Mo"] + 500 * df["Cr"]
                          + rng.normal(0, 15, 80))
    ds = Dataset(compositions=df[["Fe", "Ni", "Cr", "Mo"]],
                 properties=df[["yield_mpa", "tensile_mpa"]])
    fm = ForwardModel(
        featurizer=CompositionFeaturizer(
            element_columns=["Fe", "Ni", "Cr", "Mo"]
        ),
        targets=["yield_mpa", "tensile_mpa"], n_cv_splits=3,
    )
    fm.fit(ds, n_trials=3)
    return fm, ds


def test_accuracy_report_runs_and_grades(trained_v1_with_dataset):
    pytest.importorskip("scipy")
    from core.alloyforge.accuracy_report import evaluate_model
    fm, ds = trained_v1_with_dataset
    # Skip permutation in tests — it's a 30+ s integration concern, not a
    # unit-test concern. The integration test below exercises it once
    # with a tiny budget.
    rep = evaluate_model(
        fm, ds, targets=["yield_mpa", "tensile_mpa"],
        n_splits=3, n_seeds=1,
        skip_permutation=True, skip_reliability=True,
        include_reference_check=False,
    )
    assert set(rep.targets) == {"yield_mpa", "tensile_mpa"}
    assert rep.holdout, "holdout section empty"
    assert rep.cv, "cv section empty"
    assert rep.overall_grade in {"A", "B", "C", "D"}
    assert rep.cv["yield_mpa"]["r2_mean"] > 0.5
    assert "Targets" in rep.summary()


def test_accuracy_report_permutation_smoke(trained_v1_with_dataset):
    """Smallest possible permutation run — verifies the code path works."""
    pytest.importorskip("scipy")
    from core.alloyforge.accuracy_report import evaluate_model
    fm, ds = trained_v1_with_dataset
    rep = evaluate_model(
        fm, ds, targets=["yield_mpa"],
        n_splits=2, n_seeds=1, n_permutations=2,
        skip_reliability=True, include_reference_check=False,
    )
    assert rep.permutation, "permutation section empty"
    assert "p_value" in rep.permutation["yield_mpa"]


def test_accuracy_report_reference_check_optional(trained_v1_with_dataset):
    pytest.importorskip("scipy")
    from core.alloyforge.accuracy_report import evaluate_model
    fm, ds = trained_v1_with_dataset
    rep = evaluate_model(
        fm, ds, targets=["yield_mpa"], n_splits=2, n_seeds=1,
        skip_permutation=True, skip_reliability=True,
        include_reference_check=False,
    )
    assert rep.reference_check is None


# ---------------------------------------------------------------------------
# FastAPI router smoke
# ---------------------------------------------------------------------------

def test_data_router_endpoints_smoke():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    r = c.get("/api/v1/data/reference-alloys")
    assert r.status_code == 200
    js = r.json()
    assert js["n_alloys"] >= 35
    assert "Ti-6Al-4V" in [row["alloy_name"] for row in js["rows"]]

    r = c.get("/api/v1/data/reference-alloys/Ti-6Al-4V")
    assert r.status_code == 200
    assert r.json()["family"] == "ti_alpha_beta"

    r = c.get("/api/v1/data/external/status")
    assert r.status_code == 200
    assert "openalex" in r.json()

    # Ingest
    r = c.post("/api/v1/data/ingest", json={
        "sources": {
            "lab_a": [{"Fe": 0.7, "Ni": 0.3, "yield_mpa": 300}],
            "lab_b": [{"Fe": 0.6, "Ni": 0.4, "yield_mpa": 350}],
        },
        "element_columns": ["Fe", "Ni"],
        "target_columns": ["yield_mpa"],
        "composition_basis": "atomic_frac",
    })
    assert r.status_code == 200
    js = r.json()
    assert js["summary"]["n_rows_out"] == 2

    # LLM extract (heuristic path)
    r = c.post("/api/v1/data/llm-extract", json={
        "text": ("Fe 70 Ni 30 yield strength 400 MPa hardness 250 HV"),
        "use_llm": False,
    })
    assert r.status_code == 200
    js = r.json()
    assert js["summary"]["used_llm"] is False


def test_unknown_reference_alloy_returns_404():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    r = c.get("/api/v1/data/reference-alloys/NotARealAlloy")
    assert r.status_code == 404
