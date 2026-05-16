"""Streamlit page: composition design, verification, inverse design, and
Claude-assisted target parsing/explanation.

This page is intentionally self-contained — every interaction goes through
:mod:`core.composition_platform` and :mod:`core.llm_designer`, so the same
workflow can be exercised from the FastAPI router or a notebook.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from core.composition_platform import (
    AVAILABLE_ESTIMATORS,
    CompositionDesigner,
    DesignConstraints,
    PropertyPredictor,
)
from core.llm_designer import LLMDesigner
from core.synthetic_alloy_data import (
    default_elements,
    default_properties,
    generate_synthetic_dataset,
    target_from_quantile,
)

st.set_page_config(page_title="조성 설계 플랫폼", page_icon="🧪", layout="wide")

st.title("🧪 조성 설계 · 검증 · 역설계 플랫폼")
st.caption(
    "Composition design / verification / inverse design with optional Claude "
    "(LLM) assistance. Use the tabs below to walk through the workflow."
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _ss_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


_ss_default("dataset", None)
_ss_default("predictor", None)
_ss_default("designer", None)
_ss_default("candidates", None)
_ss_default("last_target", None)
_ss_default("llm", LLMDesigner())


def _reset_models():
    st.session_state.predictor = None
    st.session_state.designer = None
    st.session_state.candidates = None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_data, tab_train, tab_predict, tab_design, tab_verify, tab_claude = st.tabs([
    "1. 데이터", "2. 모델 학습", "3. 물성 예측",
    "4. 역설계", "5. 검증·타당성", "6. Claude 어시스턴트",
])


# --------------------------------------------------------------------- DATA
with tab_data:
    st.subheader("Dataset")
    st.write(
        "Upload a CSV with one row per composition (element-fraction columns "
        "summing to 1.0) plus one or more property columns. Or generate a "
        "synthetic dataset to try the workflow."
    )
    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader("CSV 업로드", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.dataset = df
                _reset_models()
                st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
            except Exception as exc:
                st.error(f"CSV parsing failed: {exc}")

    with col2:
        st.markdown("**또는 합성 데이터 생성**")
        n = st.number_input("샘플 수", min_value=50, max_value=5000, value=400, step=50)
        noise = st.slider("노이즈 크기 (σ)", 0.0, 0.2, 0.05, 0.01)
        seed = st.number_input("Random seed", value=42, step=1)
        if st.button("합성 데이터 생성", type="secondary"):
            df = generate_synthetic_dataset(
                n_samples=int(n), noise_scale=float(noise), random_state=int(seed)
            )
            st.session_state.dataset = df
            _reset_models()
            st.success(f"Generated {len(df)} synthetic rows.")

    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        st.markdown(f"**미리보기 — shape={df.shape}**")
        st.dataframe(df.head(20), use_container_width=True)
        with st.expander("기술통계 보기"):
            st.dataframe(df.describe(), use_container_width=True)


# -------------------------------------------------------------------- TRAIN
with tab_train:
    st.subheader("모델 학습")
    df = st.session_state.dataset
    if df is None:
        st.info("먼저 데이터를 불러오세요.")
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        c1, c2, c3 = st.columns(3)
        with c1:
            est = st.selectbox("Estimator", AVAILABLE_ESTIMATORS, index=0)
        with c2:
            test_size = st.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
        with c3:
            cv = st.number_input("CV folds", min_value=2, max_value=10, value=5, step=1)

        st.markdown(
            "_컬럼을 비워두면 자동으로 (합 ≈ 1.0인 컬럼 = features, 나머지 = properties)_"
        )
        feat = st.multiselect(
            "Feature columns (선택)", numeric_cols, default=[]
        )
        prop = st.multiselect(
            "Property columns (선택)", numeric_cols, default=[]
        )

        if st.button("학습 시작", type="primary"):
            try:
                predictor = PropertyPredictor(estimator=est, random_state=42)
                report = predictor.train(
                    df,
                    feature_columns=feat or None,
                    property_columns=prop or None,
                    test_size=test_size,
                    cv_folds=int(cv),
                )
                st.session_state.predictor = predictor
                st.session_state.designer = CompositionDesigner(predictor)
                st.success("학습 완료.")
            except Exception as exc:
                st.error(f"학습 실패: {exc}")

        if st.session_state.predictor and st.session_state.predictor.report:
            rep = st.session_state.predictor.report
            st.markdown("### Validation report")
            r_df = pd.DataFrame({
                "property": rep.property_columns,
                "val_R²": [rep.val_r2[p] for p in rep.property_columns],
                "val_MAE": [rep.val_mae[p] for p in rep.property_columns],
                "cv_R²_mean": [rep.cv_r2_mean[p] for p in rep.property_columns],
                "cv_R²_std": [rep.cv_r2_std[p] for p in rep.property_columns],
            })
            st.dataframe(r_df.round(4), use_container_width=True)
            st.caption(
                f"n_samples={rep.n_samples} (train={rep.n_train}, val={rep.n_val}); "
                f"estimator={rep.estimator_name}"
            )


# ------------------------------------------------------------------ PREDICT
with tab_predict:
    st.subheader("Composition → Property 예측")
    predictor = st.session_state.predictor
    if predictor is None:
        st.info("모델을 먼저 학습하세요.")
    else:
        st.markdown(f"**Elements**: {predictor.feature_columns}")
        cols = st.columns(min(4, len(predictor.feature_columns)))
        comp: dict[str, float] = {}
        for i, feat in enumerate(predictor.feature_columns):
            with cols[i % len(cols)]:
                comp[feat] = st.number_input(
                    feat, min_value=0.0, max_value=1.0,
                    value=round(1.0 / len(predictor.feature_columns), 4),
                    step=0.01, format="%.4f", key=f"pred_{feat}",
                )
        total = sum(comp.values())
        st.write(f"Sum = {total:.4f}  {'✅' if abs(total - 1.0) < 1e-3 else '⚠ 1.0이 되도록 조정하세요'}")
        if st.button("예측"):
            try:
                result = predictor.predict(comp)
                left, right = st.columns(2)
                with left:
                    st.markdown("**Predicted properties**")
                    st.dataframe(
                        pd.DataFrame(result.properties.items(),
                                     columns=["property", "predicted"]).round(4),
                        use_container_width=True,
                    )
                with right:
                    if result.uncertainty:
                        st.markdown("**Uncertainty (RF tree std)**")
                        st.dataframe(
                            pd.DataFrame(result.uncertainty.items(),
                                         columns=["property", "std"]).round(4),
                            use_container_width=True,
                        )
                    else:
                        st.info("선택된 estimator에는 분산 추정이 없습니다 (RF만 지원).")
            except Exception as exc:
                st.error(f"예측 실패: {exc}")


# ------------------------------------------------------------------- DESIGN
with tab_design:
    st.subheader("역설계 (Inverse Design)")
    predictor = st.session_state.predictor
    designer = st.session_state.designer
    if predictor is None or designer is None:
        st.info("모델을 먼저 학습하세요.")
    else:
        st.markdown("#### Target properties")
        target_inputs: dict[str, float] = {}
        weight_inputs: dict[str, float] = {}
        df = st.session_state.dataset
        for p in predictor.property_columns:
            default = float(df[p].quantile(0.9)) if df is not None and p in df else 0.0
            c1, c2 = st.columns([3, 1])
            with c1:
                target_inputs[p] = st.number_input(
                    f"target {p}", value=default, key=f"tgt_{p}",
                )
            with c2:
                weight_inputs[p] = st.number_input(
                    f"w {p}", value=1.0, min_value=0.0, step=0.1, key=f"w_{p}",
                )

        st.markdown("#### Constraints (선택)")
        with st.expander("element 별 min / max / fixed 설정"):
            mn = st.text_area(
                "min_fraction (JSON, 예: {\"Fe\": 0.2})", value="", key="c_min"
            )
            mx = st.text_area(
                "max_fraction (JSON, 예: {\"Cr\": 0.3})", value="", key="c_max"
            )
            fx = st.text_area(
                "fixed (JSON, 예: {\"Ni\": 0.1})", value="", key="c_fix"
            )

        st.markdown("#### Sampling")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            strategy = st.selectbox("Strategy", ["dirichlet", "ga"], index=0)
        with s2:
            n_cand = st.number_input(
                "Candidates", min_value=200, max_value=50000, value=5000, step=500
            )
        with s3:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)
        with s4:
            seed = st.number_input("Seed", value=0, step=1, key="design_seed")

        if st.button("역설계 실행", type="primary"):
            try:
                constraints = None
                m_lo = json.loads(mn) if mn.strip() else {}
                m_hi = json.loads(mx) if mx.strip() else {}
                m_fx = json.loads(fx) if fx.strip() else {}
                if m_lo or m_hi or m_fx:
                    constraints = DesignConstraints(
                        min_fraction=m_lo, max_fraction=m_hi, fixed=m_fx
                    )
                cands = designer.design_inverse(
                    target_properties=target_inputs,
                    weights=weight_inputs,
                    num_candidates=int(n_cand),
                    top_k=int(top_k),
                    constraints=constraints,
                    strategy=strategy,
                    random_state=int(seed) or None,
                )
                st.session_state.candidates = cands
                st.session_state.last_target = target_inputs
                st.success(f"{len(cands)} candidates generated.")
            except Exception as exc:
                st.error(f"역설계 실패: {exc}")

        cands = st.session_state.candidates
        if cands:
            rows = []
            for i, c in enumerate(cands, start=1):
                row = {"#": i, "score": round(c.score, 6)}
                row.update({f"x_{k}": round(v, 4) for k, v in c.composition.items()})
                row.update({f"pred_{k}": round(v, 4) for k, v in c.predicted.items()})
                row.update({f"err_{k}": round(v, 4) for k, v in c.rel_errors.items()})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ------------------------------------------------------------------- VERIFY
with tab_verify:
    st.subheader("검증 및 타당성 분석")
    designer = st.session_state.designer
    if designer is None:
        st.info("모델을 먼저 학습하세요.")
    else:
        feats = designer.predictor.feature_columns
        props = designer.predictor.property_columns
        st.markdown("Composition를 입력하고, 원하면 target도 함께 입력하세요.")
        cols = st.columns(min(4, len(feats)))
        comp = {}
        for i, f in enumerate(feats):
            with cols[i % len(cols)]:
                comp[f] = st.number_input(
                    f, min_value=0.0, max_value=1.0,
                    value=round(1.0 / len(feats), 4),
                    step=0.01, format="%.4f", key=f"ver_{f}",
                )
        targets_text = st.text_area(
            "Target properties (JSON, optional)",
            value=json.dumps({p: 0.0 for p in props}, indent=2),
            height=120,
        )
        tol = st.slider("Tolerance (relative)", 0.01, 0.5, 0.1, 0.01)
        if st.button("분석 실행"):
            try:
                tgt = json.loads(targets_text) if targets_text.strip() else None
                if tgt is not None:
                    tgt = {k: float(v) for k, v in tgt.items()}
                analysis = designer.analyse_feasibility(
                    composition=comp, target_properties=tgt, tolerance=tol
                )
                st.json(analysis)
            except Exception as exc:
                st.error(f"분석 실패: {exc}")


# ------------------------------------------------------------------- CLAUDE
with tab_claude:
    st.subheader("Claude 어시스턴트")
    llm: LLMDesigner = st.session_state.llm
    predictor = st.session_state.predictor

    if llm.available:
        st.success(f"Claude API 사용 가능 (model={llm.model})")
    else:
        st.warning(
            "ANTHROPIC_API_KEY가 설정되지 않았거나 `anthropic` 패키지가 없어 "
            "휴리스틱 fallback으로 동작합니다."
        )

    st.markdown("#### A. 자연어 → target 속성 파싱")
    user_req = st.text_area(
        "원하는 합금에 대해 자유롭게 작성하세요",
        value="yield strength around 650 MPa and hardness near 200 HB",
        height=80,
    )
    if st.button("Target 파싱"):
        if predictor is None:
            st.error("먼저 모델을 학습하세요.")
        else:
            target, resp = llm.parse_target(user_req, predictor.property_columns)
            st.write("**파싱 결과**", target)
            st.caption(f"used_llm={resp.used_llm}, model={resp.model}")
            if resp.text:
                with st.expander("LLM raw text"):
                    st.code(resp.text)
            if target:
                st.session_state.last_target = target

    st.markdown("#### B. 후보 조성에 대한 LLM 해설")
    cands = st.session_state.candidates
    if not cands:
        st.info("먼저 4번 탭에서 역설계를 실행하세요.")
    elif st.button("Claude 설명 생성"):
        resp = llm.explain_candidates(
            target=st.session_state.last_target or {},
            candidates=[c.to_dict() for c in cands],
            model_r2=predictor.report.val_r2 if predictor and predictor.report else None,
        )
        st.markdown(resp.text or "_(no output)_")
        st.caption(f"used_llm={resp.used_llm}, model={resp.model}")
