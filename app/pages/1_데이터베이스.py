"""Page 1 — Hierarchical Database Ingest & Statistics."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `core` importable when Streamlit changes cwd to the page directory
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db import (  # noqa: E402
    DEFAULT_DB_PATH,
    ingest_dataframe,
    init_db,
    materialize_training_set,
    query_property_breakdown,
    query_stats,
)

st.set_page_config(page_title="1 · 데이터베이스", page_icon="🗄️", layout="wide")
st.title("🗄️ Step 1 — Hierarchical Database")

DB_PATH = ROOT / "data" / "alloy.db"
init_db(DB_PATH)

with st.sidebar:
    st.header("CSV 업로드")
    uploaded = st.file_uploader("Composition + property CSV", type=["csv"])
    target_prop = st.text_input(
        "Target property",
        value="HV",
        help="e.g. HV, sigma_compressive, KIC, sigma_y, sigma_UTS",
    )
    process_method = st.selectbox(
        "Process method (default for rows w/o `process_method` col)",
        ["unknown", "cast", "SPS", "LPBF", "DED", "EB-PBF", "ARC"],
    )
    condition = st.text_input("Condition", value="RT")
    T_test_K = st.number_input("Test temperature (K)", value=298.0, step=10.0)
    if st.button("Ingest into DB", type="primary", disabled=uploaded is None):
        df = pd.read_csv(uploaded)
        try:
            summary = ingest_dataframe(
                df,
                target_property=target_prop,
                process_method=process_method,
                condition=condition,
                T_test_K=float(T_test_K),
                db_path=DB_PATH,
            )
            st.success(
                f"✅ {summary.rows_in} rows → "
                f"{summary.alloys_added} new alloys, "
                f"{summary.processes_added} processes, "
                f"{summary.properties_added} properties"
            )
        except Exception as e:
            st.error(f"Ingest failed: {e}")

# --- Stats overview ---
stats = query_stats(DB_PATH)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Alloys", stats["alloys"])
c2.metric("Composition rows", stats["composition_rows"])
c3.metric("Processes", stats["process"])
c4.metric("Microstructure", stats["microstructure"])
c5.metric("Properties", stats["properties"])

st.divider()

st.subheader("Property breakdown")
breakdown = query_property_breakdown(DB_PATH)
if breakdown.empty:
    st.info("아직 데이터가 없습니다. 좌측에서 CSV를 업로드하십시오.")
else:
    st.dataframe(breakdown, use_container_width=True)

# --- Training set preview ---
st.divider()
st.subheader("Training-set materialization")
col1, col2 = st.columns(2)
with col1:
    qtarget = st.text_input("Target", value=target_prop, key="qt")
with col2:
    qcondition = st.text_input("Condition", value=condition, key="qc")

if st.button("Materialize"):
    ts = materialize_training_set(qtarget, qcondition, db_path=DB_PATH)
    if ts.empty:
        st.warning("No matching rows.")
    else:
        st.success(f"{len(ts)} rows / {len(ts.columns)} columns")
        st.dataframe(ts.head(50), use_container_width=True)
        st.download_button(
            "Download as CSV",
            ts.to_csv(index=False).encode("utf-8"),
            file_name=f"training_set_{qtarget}_{qcondition}.csv",
            mime="text/csv",
        )

st.divider()
st.caption(f"DB: `{DB_PATH}`  ·  Schema: alloys / composition / process / microstructure / properties")
