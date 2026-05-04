"""Page 3 — Multi-algorithm Benchmarking."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark import benchmark, permutation_pvalue, to_dataframe, y_randomization  # noqa: E402
from core.db import materialize_training_set  # noqa: E402
from core.features import build_feature_matrix  # noqa: E402
from core.models import available_models  # noqa: E402

st.set_page_config(page_title="3 · 벤치마킹", page_icon="📊", layout="wide")
st.title("📊 Step 3 — Multi-algorithm Benchmarking")

DB_PATH = ROOT / "data" / "alloy.db"
all_models = available_models()

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
    st.header("Models")
    chosen = st.multiselect(
        "Select models",
        list(all_models),
        default=[m for m in ["RandomForest", "XGBoost", "LightGBM", "PLS1", "BayesianRidge", "GPR"] if m in all_models],
    )

    st.divider()
    st.header("CV strategy")
    cv = st.radio("CV", ["kfold", "group"])
    n_splits = st.slider("n_splits (kfold)", 3, 10, 5)
    n_seeds = st.slider("n_seeds (kfold)", 1, 20, 5)

    st.divider()
    st.header("Validation tests")
    do_perm = st.checkbox("Permutation test (slow)", value=False)
    do_yrand = st.checkbox("Y-randomization (slow)", value=False)

    st.divider()
    use_physics = st.checkbox("Add physics features", value=True)
    use_process = st.checkbox("Add process features (VED, Rosenthal)", value=True)


def _ready(d: pd.DataFrame, t: str) -> bool:
    if d is None or d.empty:
        st.info("좌측에서 데이터를 선택하십시오.")
        return False
    if t not in d.columns:
        st.error(f"Target column `{t}` not in data. Available: {list(d.columns)}")
        return False
    return True


if _ready(df, target):
    X, y = build_feature_matrix(df, use_physics=use_physics, use_process=use_process, target_col=target)
    groups = None
    if cv == "group":
        if "alloy_class" in df.columns:
            groups = df["alloy_class"].iloc[: len(y)].reset_index(drop=True)
        if groups is None or groups.nunique() < 2:
            st.error("Group CV requires `alloy_class` column with ≥2 unique groups.")
            st.stop()

    st.success(f"X: {X.shape}  ·  y: {len(y)} samples · target=`{target}`")

    if st.button("Run benchmark", type="primary"):
        factories = {k: all_models[k] for k in chosen}
        with st.spinner("Cross-validating models..."):
            results = benchmark(X, y, factories, cv=cv, n_splits=n_splits, n_seeds=n_seeds, groups=groups)

        if do_perm or do_yrand:
            with st.spinner("Running permutation / y-randomization tests (slow)..."):
                for r in results:
                    if r.name not in factories:
                        continue
                    fac = factories[r.name]
                    if do_perm:
                        try:
                            _, pval = permutation_pvalue(X, y, fac, n_permutations=50, n_splits=n_splits)
                            r.permutation_p = pval
                        except Exception as e:
                            st.warning(f"perm test {r.name}: {e}")
                    if do_yrand:
                        try:
                            rep = y_randomization(X, y, fac, n_iter=30, n_splits=n_splits)
                            r.y_random_r2_mean = rep["mean"]
                        except Exception as e:
                            st.warning(f"y-rand {r.name}: {e}")

        rep_df = to_dataframe(results)
        st.subheader("Results")
        st.dataframe(rep_df.style.format({
            "r2_mean": "{:+.3f}", "r2_std": "{:.3f}",
            "rmse_mean": "{:.2f}", "rmse_std": "{:.2f}",
            "mae_mean": "{:.2f}", "mae_std": "{:.2f}",
            "runtime_s": "{:.1f}",
            "permutation_p": "{:.4f}", "y_random_r2_mean": "{:+.3f}",
        }, na_rep="—"), use_container_width=True)

        st.download_button(
            "Download report.csv",
            rep_df.to_csv(index=False).encode("utf-8"),
            file_name=f"benchmark_{target}.csv",
            mime="text/csv",
        )

        tex = rep_df[["name", "r2_mean", "r2_std", "rmse_mean", "rmse_std", "n_folds"]].copy()
        tex["R²"] = tex.apply(lambda r: f"{r['r2_mean']:+.3f} ± {r['r2_std']:.3f}", axis=1)
        tex["RMSE"] = tex.apply(lambda r: f"{r['rmse_mean']:.2f} ± {r['rmse_std']:.2f}", axis=1)
        st.code(tex[["name", "R²", "RMSE", "n_folds"]].to_latex(index=False), language="latex")
