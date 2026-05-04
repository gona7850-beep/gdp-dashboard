"""Page 4 — SHAP-based XAI (global / local / interaction / physics validation)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db import materialize_training_set  # noqa: E402
from core.features import build_feature_matrix  # noqa: E402
from core.models import available_models  # noqa: E402
from core.shap_analysis import explain, global_importance, physics_validation, top_interactions  # noqa: E402

st.set_page_config(page_title="4 · SHAP XAI", page_icon="🧠", layout="wide")
st.title("🧠 Step 4 — SHAP Explainability + 4.5 Physics Validation")

DB_PATH = ROOT / "data" / "alloy.db"
all_models = available_models()
tree_models = [m for m in ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "GradientBoosting"] if m in all_models]

with st.sidebar:
    st.header("Source")
    src = st.radio("Data", ["DB materialize", "Upload CSV"])
    if src == "DB materialize":
        target = st.text_input("Target", value="HV")
        condition = st.text_input("Condition", value="RT")
        df = materialize_training_set(target, condition, db_path=DB_PATH)
    else:
        upload = st.file_uploader("CSV", type=["csv"])
        target = st.text_input("Target column", value="HV")
        df = pd.read_csv(upload) if upload is not None else pd.DataFrame()

    st.divider()
    model_name = st.selectbox("Tree model (TreeExplainer)", tree_models or ["—"], index=0 if tree_models else 0)
    use_physics = st.checkbox("Add physics features", value=True)
    use_process = st.checkbox("Add process features", value=True)
    compute_inter = st.checkbox("Compute pairwise interactions (slow)", value=True)


def _ready(d, t) -> bool:
    if d is None or d.empty:
        st.info("좌측에서 데이터를 선택하십시오.")
        return False
    if t not in d.columns:
        st.error(f"Target column `{t}` not in data.")
        return False
    if not tree_models:
        st.error("No tree-based model available (need XGBoost / LightGBM / RandomForest).")
        return False
    return True


if _ready(df, target):
    X, y = build_feature_matrix(df, use_physics=use_physics, use_process=use_process, target_col=target)
    st.success(f"X: {X.shape} · y: {len(y)} samples")

    if st.button("Run SHAP analysis", type="primary"):
        with st.spinner(f"Fitting {model_name} + computing SHAP values..."):
            est = all_models[model_name]()
            est.fit(X.values, y.values)
            res = explain(est, X, compute_interactions=compute_inter)

        tab_g, tab_l, tab_i, tab_p = st.tabs(["Global", "Local", "Interaction", "Physics Validation"])

        with tab_g:
            gi = global_importance(res)
            st.subheader("Global feature importance (mean |SHAP|)")
            st.bar_chart(gi.set_index("feature")["mean_abs_shap"])
            st.dataframe(gi, use_container_width=True)

        with tab_l:
            st.subheader("Local explanation")
            idx = st.number_input("Sample index", 0, len(X) - 1, 0)
            sv_row = res.shap_values[int(idx)]
            df_local = pd.DataFrame({
                "feature": res.feature_names,
                "value": X.iloc[int(idx)].values,
                "shap_value": sv_row,
            }).sort_values("shap_value", key=abs, ascending=False)
            st.dataframe(df_local, use_container_width=True)
            pred = float(res.base_value + sv_row.sum())
            st.metric("Predicted", f"{pred:.2f}", delta=f"actual = {y.iloc[int(idx)]:.2f}")

        with tab_i:
            st.subheader("Top pairwise interactions")
            if res.interaction_values is None:
                st.info("Interactions disabled or not supported by this model.")
            else:
                ti = top_interactions(res, k=15)
                st.dataframe(ti, use_container_width=True)
                iv = np.abs(res.interaction_values).mean(axis=0)
                heat = pd.DataFrame(iv, index=res.feature_names, columns=res.feature_names)
                st.dataframe(heat.style.background_gradient(cmap="viridis"), use_container_width=True)

        with tab_p:
            st.subheader(f"Physics validation against priors for `{target}`")
            pv = physics_validation(res, target)
            if pv.empty:
                st.info(f"No physics priors registered for target `{target}`.")
            else:
                st.dataframe(
                    pv.style.format({"mean_signed_shap": "{:+.3f}"}),
                    use_container_width=True,
                )
                agree = (pv["agreement"] == "✅ agree").sum()
                conflict = (pv["agreement"] == "⚠ conflict").sum()
                ambiguous = (pv["agreement"] == "ambiguous prior").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Agreement", agree)
                c2.metric("⚠ Conflict", conflict)
                c3.metric("Ambiguous prior", ambiguous)
                st.caption(
                    "Conflicts are the most interesting — they may indicate either ML mis-fit or a genuine "
                    "novel mechanism worth exploring in your Discussion section."
                )
