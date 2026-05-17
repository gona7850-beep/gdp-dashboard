"""Composition Design Platform — landing page.

Designed so that **anyone visiting the deployed URL** can verify the
platform works in 30 seconds:

  • One button trains a forward model on the 38-alloy reference DB.
  • One number-input row + button predicts hardness / strength for any
    composition (auto-fills with Ti-6Al-4V so first-time users see a
    realistic answer immediately).
  • One button runs HTS compound ranking for Nb-host alloys.

The full power-user UI lives in pages 7-10 in the sidebar.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from core.alloyforge.reference_data import (
    find_alloy,
    reference_dataset,
    reference_elements,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Composition Design Platform",
    page_icon="🧪",
    layout="wide",
)


def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss("demo_model", None)
_ss("demo_elements", None)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧪 Composition Design Platform")
st.caption(
    "ML-driven composition → property prediction · inverse design · "
    "high-throughput compound screening · accuracy auditing. "
    "Open-source research workbench (XGB + GP, NSGA-II, conformal intervals)."
)

# Status row — gives visitors a sense of platform scale
c1, c2, c3, c4 = st.columns(4)
with c1:
    n_alloys = len(reference_dataset())
    st.metric("Curated reference alloys", n_alloys)
with c2:
    n_el = len(reference_elements())
    st.metric("Elements supported", n_el)
with c3:
    try:
        from core.alloyforge.hts_screening import NB_HOST_COMPOUNDS
        st.metric("HTS compounds in DB", len(NB_HOST_COMPOUNDS))
    except ImportError:
        st.metric("HTS compounds in DB", "—")
with c4:
    st.metric("Targets per alloy", "7")  # yield/UTS/elong/HV/density/E/Tm


st.divider()


# ---------------------------------------------------------------------------
# Live demo — anyone visiting the deploy can verify the platform works here
# ---------------------------------------------------------------------------

st.markdown("## 🎬 30-second live demo")
st.caption(
    "Click the buttons in order — no upload, no API key, no setup. "
    "Trains a real ML model on the curated 38-alloy reference set, then "
    "predicts properties for any composition you choose."
)

demo_col1, demo_col2 = st.columns([2, 3])

with demo_col1:
    st.markdown("### 1. Train a model")
    st.caption(
        "Group-aware 5-fold CV on the 38-alloy curated DB "
        "(ASM Handbook + MatWeb + producer datasheets)."
    )
    if st.button("▶ Train now (~10 s)", type="primary",
                 use_container_width=True):
        from core.alloyforge.data_pipeline import (
            CompositionFeaturizer, Dataset,
        )
        from core.alloyforge.forward_model import ForwardModel
        with st.spinner("Training XGBoost + GP residual head…"):
            ref = reference_dataset()
            els = reference_elements()
            targets = ["yield_mpa", "tensile_mpa", "hardness_hv", "density_gcc"]
            df_train = ref.dropna(subset=targets).reset_index(drop=True)
            ds = Dataset(
                compositions=df_train[els],
                properties=df_train[targets],
                groups=df_train["family"],
            )
            fm = ForwardModel(
                featurizer=CompositionFeaturizer(element_columns=els),
                targets=targets,
                n_cv_splits=5,
            )
            fm.fit(ds, n_trials=4)
            st.session_state.demo_model = fm
            st.session_state.demo_elements = els
        st.success("Trained.")
        rep = pd.DataFrame(fm.metrics_).T.round(3)
        rep = rep.rename(columns={"cv_r2": "CV R²",
                                  "cv_mae": "CV MAE",
                                  "n_train": "n"})
        st.dataframe(rep, use_container_width=True)

with demo_col2:
    st.markdown("### 2. Predict properties for any composition")
    st.caption(
        "Try Ti-6Al-4V (Ti 86.2, Al 10.2, V 3.6 at%) — the model should "
        "recover ~880 MPa yield, ~340 HV from literature."
    )
    fm = st.session_state.demo_model
    els = st.session_state.demo_elements or reference_elements()

    # Pre-fill Ti-6Al-4V so the demo is concrete out of the box
    ti = find_alloy("Ti-6Al-4V")
    default_comp = ti.as_atomic() if ti else {}

    comp_cols = st.columns(4)
    comp = {}
    # Show only elements with meaningful default; rest sum to 0
    visible_els = [e for e in els if default_comp.get(e, 0) > 0.001][:8]
    if len(visible_els) < 4:
        visible_els = list(els)[:4]
    for i, e in enumerate(visible_els):
        with comp_cols[i % 4]:
            comp[e] = st.number_input(
                e, 0.0, 1.0, float(default_comp.get(e, 0.0)),
                0.01, format="%.3f", key=f"demo_comp_{e}",
            )
    # Other elements = 0
    for e in els:
        comp.setdefault(e, 0.0)

    total = sum(comp.values())
    norm_msg = "✅" if abs(total - 1) < 1e-3 else f"⚠ sum={total:.3f}"
    st.caption(f"Composition sum: {total:.3f} {norm_msg}")

    if st.button("▶ Predict", use_container_width=True):
        if fm is None:
            st.warning("Train the model first (button on the left).")
        else:
            # Normalise composition to sum=1
            tot = sum(comp.values()) or 1.0
            q = pd.DataFrame([{k: v / tot for k, v in comp.items()}])
            preds = fm.predict(q).iloc[0]

            rows = []
            ti_props = {
                "yield_mpa": ti.yield_mpa if ti else None,
                "tensile_mpa": ti.tensile_mpa if ti else None,
                "hardness_hv": ti.hardness_hv if ti else None,
                "density_gcc": ti.density_gcc if ti else None,
            }
            for t in ["yield_mpa", "tensile_mpa", "hardness_hv", "density_gcc"]:
                rows.append({
                    "property": t,
                    "predicted μ": round(float(preds[f"{t}_mean"]), 2),
                    "σ": round(float(preds[f"{t}_std"]), 2),
                    "Ti-6Al-4V literature": ti_props.get(t),
                })
            st.dataframe(pd.DataFrame(rows),
                         use_container_width=True, hide_index=True)

            mae = np.mean([
                abs(float(preds[f"{t}_mean"]) - ti_props[t])
                for t in ti_props if ti_props[t] is not None
            ])
            st.caption(
                f"Mean absolute error vs literature: **{mae:.1f}** "
                "(in the property's own unit — should be ~5-30 for a "
                "model trained on 36 alloys including this one)."
            )


st.divider()


# ---------------------------------------------------------------------------
# HTS one-click
# ---------------------------------------------------------------------------

st.markdown("## 🔬 High-throughput compound screening")
st.caption(
    "For Nb-host alloys: rank candidate precipitate phases by "
    "(tie line × stability × modular coherency). Bundled "
    "compound DB; OQMD live queries also supported on power-user pages."
)

if st.button("▶ Run HTS ranking for Nb host"):
    try:
        from core.alloyforge.hts_screening import rank_compounds
        df = rank_compounds(host="Nb", top_k=10)
        st.dataframe(
            df[["formula", "tie_line", "stability", "coherency", "total",
                "space_group", "delta_h_per_atom_ev"]],
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Top hits are silicide variants — exactly the in-situ-composite "
            "phases that the Bewlay et al. body of literature has validated "
            "as the primary strengthening phases for Nb-Si alloys."
        )
    except ImportError as exc:
        st.warning(f"HTS module not available: {exc}")


st.divider()


# ---------------------------------------------------------------------------
# Where to go next
# ---------------------------------------------------------------------------

st.markdown("## 🗺️ Where to go next")

nav_cols = st.columns(2)
with nav_cols[0]:
    st.markdown(
        """
**Power-user pages** (사이드바에서 선택)
- **7_조성설계_플랫폼** — Lite RF model + Dirichlet MC + GA inverse design + Claude assistant
- **8_AlloyForge_고급플랫폼** — Stacked XGB+GP+Optuna, conformal intervals, SHAP
- **9_데이터_수집_통합** — Reference DB / CSV upload / external APIs / LLM table extraction
- **10_HTS_화합물_스크리닝** — Compound DB · score + rank · OQMD live · ML mix
"""
    )
with nav_cols[1]:
    st.markdown(
        """
**REST API** (separate process: `uvicorn backend.main:app --reload`)
- `/api/v1/composition` — lite RF + GA inverse
- `/api/v1/alloyforge` — XGB+GP + NSGA-II + SHAP
- `/api/v1/data` — reference / ingest / external / LLM extract
- `/api/v1/hts` — compound screening + OQMD
- `/docs` — OpenAPI Swagger

**CLI demos**
- `python examples/reference_alloys_demo.py`
- `python examples/benchmark_real_nb_si.py`
- `python examples/hts_nb_alloy_demo.py`
- `python examples/accuracy_report_demo.py`
"""
    )

st.divider()

st.caption(
    "Deploy yourself: see [`DEPLOY.md`](https://github.com/gona7850-beep/"
    "gdp-dashboard/blob/main/DEPLOY.md) for one-click Streamlit Cloud / "
    "HF Spaces / Docker / Railway setup."
)
