from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from nb_alloy_platform.data_preprocessing import (
    compute_correlations,
    load_data,
    prepare_long_format,
    select_top_features,
)
from nb_alloy_platform.model_training import train_models
from nb_alloy_platform.optimization import random_search_optimization

st.set_page_config(page_title="Nb Alloy Platform", page_icon="⚙️", layout="wide")


@st.cache_data
def _load_sample_data() -> pd.DataFrame:
    sample_path = Path(__file__).parent / "data" / "sample_alloy_data.csv"
    return pd.read_csv(sample_path)


def _uploaded_to_df(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def _download_df_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv")


st.title("⚙️ Nb Alloy Composition–Property Platform")
st.caption("전처리 → 상관분석 → 모델 학습 → 조성 최적화까지 한 번에 수행하는 UI/UX 중심 워크벤치")

with st.sidebar:
    st.header("설정")
    data_source = st.radio("데이터 소스", ["샘플 데이터", "파일 업로드"], index=0)
    st.markdown("---")
    st.info(
        "권장 입력 형식: 각 행이 하나의 합금 샘플이며, 조성 컬럼(예: Nb, Ti, Si...)과 "
        "목표 물성 컬럼(예: HV, KIC)을 포함하세요."
    )

if data_source == "샘플 데이터":
    raw_df = _load_sample_data()
else:
    uploaded = st.file_uploader("CSV/XLSX 업로드", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.warning("파일을 업로드하면 분석이 시작됩니다.")
        st.stop()
    raw_df = _uploaded_to_df(uploaded)

st.subheader("1) 데이터 미리보기")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(raw_df.head(20), use_container_width=True)
with col2:
    st.metric("Rows", len(raw_df))
    st.metric("Columns", len(raw_df.columns))

numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("수치형 컬럼이 없습니다. 학습/최적화를 위해 수치형 컬럼을 포함해주세요.")
    st.stop()

st.markdown("---")

st.subheader("2) Long-format 변환")
if st.button("Long-format 생성"):
    long_df = prepare_long_format(raw_df)
    st.session_state["long_df"] = long_df

if "long_df" in st.session_state:
    st.dataframe(st.session_state["long_df"].head(30), use_container_width=True)
    _download_df_button(st.session_state["long_df"], "Long-format CSV 다운로드", "long_format.csv")

st.markdown("---")
st.subheader("3) 상관 분석 + 중요 변수 선택")

left, right = st.columns(2)
with left:
    target_col = st.selectbox("타깃 물성", options=numeric_cols, index=max(0, len(numeric_cols) - 1))
with right:
    default_features = [c for c in numeric_cols if c != target_col]
    feature_cols = st.multiselect("피처 컬럼", options=default_features, default=default_features)

rank_method = st.selectbox("랭킹 방식", ["mic", "pcc", "union"], index=0)
top_n = st.slider("Top N", min_value=3, max_value=min(20, max(3, len(feature_cols))), value=min(8, max(3, len(feature_cols))))

if st.button("상관 분석 실행"):
    try:
        corr_df = compute_correlations(raw_df, target_col=target_col, feature_cols=feature_cols)
        selected = select_top_features(corr_df, top_n=top_n, method=rank_method)
        st.session_state["corr_df"] = corr_df
        st.session_state["selected_features"] = selected
    except Exception as exc:
        st.error(f"상관 분석 실패: {exc}")

if "corr_df" in st.session_state:
    c1, c2 = st.columns(2)
    with c1:
        st.write("상관 분석 결과")
        st.dataframe(st.session_state["corr_df"], use_container_width=True)
    with c2:
        st.write("선택된 피처")
        st.write(st.session_state["selected_features"])

st.markdown("---")
st.subheader("4) 모델 학습")

default_train_features = st.session_state.get("selected_features", feature_cols[: min(5, len(feature_cols))])
train_features = st.multiselect("학습 피처", options=feature_cols, default=default_train_features)
algorithms = st.multiselect(
    "알고리즘",
    ["ElasticNet", "RandomForest", "SVR", "BayesianRidge"],
    default=["RandomForest", "ElasticNet"],
)
n_splits = st.slider("CV Fold", min_value=3, max_value=10, value=5)
with_tol = st.checkbox("Within-tolerance metric 사용", value=True)
tol_rel = st.number_input("tol_rel", min_value=0.0, value=0.10, step=0.01) if with_tol else None

if st.button("모델 학습 실행"):
    if not train_features:
        st.error("학습 피처를 1개 이상 선택하세요.")
    else:
        try:
            models, results_df = train_models(
                raw_df,
                target_col=target_col,
                feature_cols=train_features,
                algorithms=algorithms,
                n_splits=n_splits,
                tol_rel=tol_rel,
            )
            st.session_state["models"] = models
            st.session_state["results_df"] = results_df
            st.session_state["train_features"] = train_features
            st.success("모델 학습 완료")
        except Exception as exc:
            st.error(f"모델 학습 실패: {exc}")

if "results_df" in st.session_state:
    st.dataframe(st.session_state["results_df"], use_container_width=True)
    _download_df_button(st.session_state["results_df"], "학습 결과 CSV 다운로드", "model_results.csv")

    best_algo = st.session_state["results_df"].iloc[0]["Algorithm"]
    model_obj = st.session_state["models"][best_algo]
    buffer = BytesIO()
    joblib.dump(model_obj, buffer)
    st.download_button(
        label=f"최고 성능 모델 다운로드 ({best_algo})",
        data=buffer.getvalue(),
        file_name=f"{target_col}_{best_algo}.joblib",
        mime="application/octet-stream",
    )

st.markdown("---")
st.subheader("5) 조성 최적화 (Random Search)")

if "models" not in st.session_state:
    st.info("먼저 모델 학습을 실행하면 최적화를 진행할 수 있습니다.")
else:
    best_algo = st.session_state["results_df"].iloc[0]["Algorithm"]
    best_model = st.session_state["models"][best_algo]
    opt_features = st.session_state["train_features"]

    st.write(f"현재 사용 모델: **{best_algo}**")
    n_samples = st.slider("샘플 수", min_value=100, max_value=10000, value=2000, step=100)
    objective = st.radio("목표", ["max", "min"], horizontal=True)

    st.write("피처별 탐색 범위")
    ranges = {}
    cols = st.columns(2)
    for i, feat in enumerate(opt_features):
        lo_default = float(np.nanmin(raw_df[feat]))
        hi_default = float(np.nanmax(raw_df[feat]))
        with cols[i % 2]:
            lo = st.number_input(f"{feat} min", value=lo_default, key=f"{feat}_min")
            hi = st.number_input(f"{feat} max", value=hi_default, key=f"{feat}_max")
        ranges[feat] = (min(lo, hi), max(lo, hi))

    if st.button("최적화 실행"):
        try:
            opt_df = random_search_optimization(
                best_model,
                param_ranges=ranges,
                n_samples=n_samples,
                objective=objective,
                enforce_sum=False,
                random_state=42,
            )
            st.session_state["opt_df"] = opt_df
        except Exception as exc:
            st.error(f"최적화 실패: {exc}")

if "opt_df" in st.session_state:
    st.dataframe(st.session_state["opt_df"].head(30), use_container_width=True)
    _download_df_button(st.session_state["opt_df"], "최적화 결과 CSV 다운로드", "optimization_results.csv")

st.markdown("---")
st.caption("Made with Streamlit + nb_alloy_platform")
