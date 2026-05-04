"""Page 2 — Physics-Informed Feature Engineering."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db import materialize_training_set  # noqa: E402
from core.features import (  # noqa: E402
    PHYSICS_FEATURE_FNS,
    add_physics_features,
    add_process_features,
    build_feature_matrix,
    feature_correlation_report,
)

st.set_page_config(page_title="2 · 피처엔지니어링", page_icon="🧪", layout="wide")
st.title("🧪 Step 2 — Physics-Informed Feature Engineering")

DB_PATH = ROOT / "data" / "alloy.db"

with st.sidebar:
    st.header("Source")
    src = st.radio("Data source", ["DB materialize", "Upload CSV"])
    if src == "DB materialize":
        target = st.text_input("Target", value="HV")
        condition = st.text_input("Condition", value="RT")
        df = materialize_training_set(target, condition, db_path=DB_PATH)
    else:
        upload = st.file_uploader("CSV", type=["csv"])
        target = st.text_input("Target column name", value="HV")
        df = pd.read_csv(upload) if upload is not None else pd.DataFrame()

    st.divider()
    st.subheader("Physics features")
    selected_feats = st.multiselect(
        "Select features to add",
        list(PHYSICS_FEATURE_FNS),
        default=list(PHYSICS_FEATURE_FNS),
    )
    add_proc = st.checkbox("Compute VED + Rosenthal cooling (if process cols exist)", value=True)

if df is None or df.empty:
    st.info("좌측에서 데이터를 선택하십시오 (DB or CSV).")
    st.stop()

st.subheader("Raw data")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"{len(df)} rows · {len(df.columns)} columns")

# Add features
enriched = add_physics_features(df, features=selected_feats)
if add_proc:
    enriched = add_process_features(enriched)

st.divider()
st.subheader("Enriched data")
st.dataframe(enriched.head(20), use_container_width=True)

# Build matrix and show correlation
if target in enriched.columns:
    X, y = build_feature_matrix(enriched, target_col=target)
    if y is not None and len(y) > 0:
        st.divider()
        st.subheader(f"Feature ↔ `{target}` correlation")
        rep = feature_correlation_report(X, y)
        st.dataframe(rep.style.format("{:+.3f}"), use_container_width=True)

        # Diagnostic: composition-only vs composition+physics quick R² with RandomForest
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score

            comp_cols = [c for c in X.columns if c not in PHYSICS_FEATURE_FNS]
            phys_cols = [c for c in X.columns if c in PHYSICS_FEATURE_FNS]
            rng = 42
            rf = RandomForestRegressor(n_estimators=300, random_state=rng, n_jobs=-1)
            n_splits = min(5, max(2, len(y) // 5))
            r2_comp = cross_val_score(rf, X[comp_cols], y, cv=n_splits, scoring="r2").mean()
            r2_full = cross_val_score(rf, X, y, cv=n_splits, scoring="r2").mean()
            r2_phys_only = cross_val_score(rf, X[phys_cols], y, cv=n_splits, scoring="r2").mean() if phys_cols else float("nan")

            c1, c2, c3 = st.columns(3)
            c1.metric("R² (composition only)", f"{r2_comp:+.3f}")
            c2.metric("R² (physics only)", f"{r2_phys_only:+.3f}")
            c3.metric("R² (composition + physics)", f"{r2_full:+.3f}")
            st.caption(
                "**진단**: small-N + RandomForest 환경에서는 raw composition만으로도 비선형 상호작용을 학습합니다 — "
                "물리 피처의 가치는 **interpretability** (SHAP)와 **외삽** 영역에서 발현됩니다."
            )
        except Exception as e:
            st.warning(f"R² 진단 비활성: {e}")

# Download
st.divider()
st.download_button(
    "Download enriched CSV",
    enriched.to_csv(index=False).encode("utf-8"),
    file_name="enriched.csv",
    mime="text/csv",
)
