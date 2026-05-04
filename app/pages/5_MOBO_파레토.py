"""Page 5 — Multi-Objective Bayesian Optimization (Pareto-front candidate proposal)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.db import materialize_training_set  # noqa: E402
from core.features import detect_element_columns  # noqa: E402
from core.mobo import BOTORCH_OK, fit_mobo, hypervolume, propose_batch  # noqa: E402

st.set_page_config(page_title="5 · MOBO", page_icon="🎯", layout="wide")
st.title("🎯 Step 5 — Multi-Objective Bayesian Optimization (Pareto)")
st.caption(f"Backend: **{'BoTorch qNEHVI' if BOTORCH_OK else 'sklearn GPR + Sobol fallback'}**")

DB_PATH = ROOT / "data" / "alloy.db"

with st.sidebar:
    st.header("Source")
    src = st.radio("Data", ["DB join (paired targets)", "Upload CSV"])
    if src == "Upload CSV":
        upload = st.file_uploader("CSV with composition + ≥2 properties", type=["csv"])
        df = pd.read_csv(upload) if upload is not None else pd.DataFrame()
        obj1 = st.text_input("Objective 1 (column)", value="HV")
        obj2 = st.text_input("Objective 2 (column)", value="sigma_compressive")
    else:
        obj1 = st.text_input("Objective 1 property", value="HV")
        obj2 = st.text_input("Objective 2 property", value="sigma_compressive")
        cond = st.text_input("Condition", value="RT")
        df1 = materialize_training_set(obj1, cond, db_path=DB_PATH)
        df2 = materialize_training_set(obj2, cond, db_path=DB_PATH)
        if not df1.empty and not df2.empty and obj2 in df2.columns:
            df = df1.merge(df2[["alloy_id", obj2]], on="alloy_id", how="inner", suffixes=("", "_dup"))
        else:
            df = pd.DataFrame()

    st.divider()
    st.header("Objective directions")
    min1 = st.checkbox(f"Minimize `{obj1}`", value=False)
    min2 = st.checkbox(f"Minimize `{obj2}`", value=False)
    q = st.slider("Batch size q", 1, 10, 5)


def _ready(d: pd.DataFrame, o1: str, o2: str) -> bool:
    if d is None or d.empty:
        st.info("Paired data needed (≥2 objectives present in same alloys).")
        return False
    if o1 not in d.columns or o2 not in d.columns:
        st.error(f"Need both `{o1}` and `{o2}` columns. Found: {list(d.columns)}")
        return False
    if not detect_element_columns(d):
        st.error("No element columns detected (Nb, Si, Ti, ...).")
        return False
    return True


if _ready(df, obj1, obj2):
    elem_cols = detect_element_columns(df)
    X = df[elem_cols].fillna(0.0).astype(float)
    Y = df[[obj1, obj2]].apply(pd.to_numeric, errors="coerce").dropna()
    X = X.loc[Y.index]
    st.success(f"{len(X)} paired observations · {X.shape[1]} composition variables")
    st.dataframe(pd.concat([X.head(), Y.head()], axis=1), use_container_width=True)

    lo = np.zeros(X.shape[1])
    hi = (X.max(axis=0) * 1.2).clip(lower=1.0).values
    bounds = np.vstack([lo, hi])

    if st.button("Fit surrogate + propose batch", type="primary"):
        with st.spinner("Fitting GP surrogate..."):
            sg = fit_mobo(X, Y, bounds=bounds, minimize=[min1, min2])
        with st.spinner("Optimizing acquisition..."):
            cands = propose_batch(sg, q=q)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(Y[obj1], Y[obj2], c="lightgray", s=40, label="Existing")
        for i, c in enumerate(cands):
            ax.scatter(c.y_pred[obj1], c.y_pred[obj2], marker="*", s=240, c="red", edgecolors="black",
                       label="Proposed" if i == 0 else None)
            ax.errorbar(c.y_pred[obj1], c.y_pred[obj2],
                        xerr=c.y_std[obj1], yerr=c.y_std[obj2],
                        fmt="none", ecolor="black", alpha=0.5)
        ax.set_xlabel(("min " if min1 else "max ") + obj1)
        ax.set_ylabel(("min " if min2 else "max ") + obj2)
        ax.set_title("MOBO Pareto-front candidates")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        rows = []
        for i, c in enumerate(cands):
            row = {"#": i + 1, **{el: round(v, 2) for el, v in c.x.items()}}
            row[f"{obj1}_pred"] = round(c.y_pred[obj1], 2)
            row[f"{obj1}_std"] = round(c.y_std[obj1], 2)
            row[f"{obj2}_pred"] = round(c.y_pred[obj2], 2)
            row[f"{obj2}_std"] = round(c.y_std[obj2], 2)
            row["acq"] = round(c.acq_value or 0.0, 4)
            rows.append(row)
        cand_df = pd.DataFrame(rows)
        st.subheader("Proposed candidates")
        st.dataframe(cand_df, use_container_width=True)

        ref = np.array([Y[obj1].min() - abs(Y[obj1].min()) * 0.1,
                        Y[obj2].min() - abs(Y[obj2].min()) * 0.1])
        hv_now = hypervolume(Y.values, ref=ref, minimize=[min1, min2])
        st.metric("Current dominated hypervolume", f"{hv_now:,.0f}")

        st.download_button(
            "Download proposed batch",
            cand_df.to_csv(index=False).encode("utf-8"),
            file_name="mobo_batch.csv",
            mime="text/csv",
        )
