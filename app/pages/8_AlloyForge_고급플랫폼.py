"""Streamlit page: AlloyForge advanced composition platform.

This page exposes the heavier ML stack from :mod:`core.alloyforge`:
stacked XGBoost + Gaussian-Process residual head with Optuna tuning,
conformal prediction intervals, domain-of-applicability scoring, NSGA-II
inverse design, SHAP explainability, and active-learning batch picks.

For a lighter-weight workflow (RF / Dirichlet MC, no heavy deps), use the
"7_조성설계_플랫폼" page instead. The two pages share the synthetic
dataset generator, so you can compare results.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st

# Heavy imports are deferred until the user actually clicks a button so the
# page loads even if AlloyForge deps are missing.
_IMPORT_ERROR: Exception | None = None
try:
    from core.alloyforge import (
        ActiveLearner,
        CompositionFeaturizer,
        ConformalCalibrator,
        Dataset,
        DesignSpec,
        DomainOfApplicability,
        ELEMENT_PROPERTIES,
        Explainer,
        ForwardModel,
        InverseDesigner,
        LLMAssistant,
        default_checker,
    )
except ImportError as exc:
    _IMPORT_ERROR = exc

from core.synthetic_alloy_data import generate_synthetic_dataset


st.set_page_config(page_title="AlloyForge 고급 플랫폼", page_icon="⚙️", layout="wide")

st.title("⚙️ AlloyForge — 고급 조성 설계 플랫폼")
st.caption(
    "Stacked XGBoost + GP · Optuna HPO · Conformal intervals · NSGA-II "
    "역설계 · SHAP · Active learning. 가벼운 워크플로우는 7번 페이지를 사용하세요."
)

if _IMPORT_ERROR is not None:
    st.error(
        f"AlloyForge 의존성이 설치되지 않았습니다: `{_IMPORT_ERROR}`\n\n"
        "설치: `pip install xgboost optuna pymoo shap scipy`"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_default("af_dataset", None)
_ss_default("af_model", None)
_ss_default("af_conformal", None)
_ss_default("af_doa", None)
_ss_default("af_element_cols", None)
_ss_default("af_target_cols", None)
_ss_default("af_candidates", None)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_data, tab_fit, tab_predict, tab_design, tab_explain, tab_al = st.tabs([
    "1. 데이터", "2. 모델 학습", "3. 예측·신뢰구간",
    "4. NSGA-II 역설계", "5. SHAP 설명", "6. Active Learning",
])


# --------------------------------------------------------------------- DATA
with tab_data:
    st.subheader("Dataset")
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("CSV 업로드", type=["csv"], key="af_upload")
        if up is not None:
            df = pd.read_csv(up)
            st.session_state.af_dataset = df
            st.success(f"{len(df)} rows loaded")
    with c2:
        st.markdown("**또는 합성 데이터 생성**")
        n = st.number_input("샘플 수", 50, 5000, 300, 50, key="af_n")
        noise = st.slider("Noise σ", 0.0, 0.2, 0.05, 0.01, key="af_noise")
        if st.button("합성 데이터 생성", key="af_gen"):
            df = generate_synthetic_dataset(n_samples=int(n), noise_scale=float(noise))
            st.session_state.af_dataset = df
            st.success(f"{len(df)} synthetic rows")

    if st.session_state.af_dataset is not None:
        df = st.session_state.af_dataset
        st.markdown(f"**shape = {df.shape}**")
        st.dataframe(df.head(20), use_container_width=True)


# ---------------------------------------------------------------------- FIT
with tab_fit:
    st.subheader("Forward model (Stacked XGB + GP, Optuna-tuned)")
    df = st.session_state.af_dataset
    if df is None:
        st.info("먼저 데이터를 불러오세요.")
    else:
        known_elements = list(ELEMENT_PROPERTIES.keys())
        auto_elems = [c for c in df.columns if c in known_elements]
        auto_targets = [c for c in df.select_dtypes("number").columns if c not in auto_elems]

        c1, c2 = st.columns(2)
        with c1:
            el_cols = st.multiselect("Element columns", df.columns.tolist(),
                                     default=auto_elems)
        with c2:
            tgt_cols = st.multiselect("Target columns", df.columns.tolist(),
                                      default=auto_targets)

        c3, c4, c5 = st.columns(3)
        with c3:
            n_trials = st.number_input("Optuna trials", 3, 100, 15, 1, key="af_trials")
        with c4:
            cv = st.number_input("CV folds", 2, 10, 5, 1, key="af_cv")
        with c5:
            group_col = st.selectbox("Group column (선택)",
                                     ["(none)"] + df.columns.tolist())

        if st.button("학습 시작", type="primary", key="af_fit"):
            if not el_cols or not tgt_cols:
                st.error("element/target 컬럼을 선택하세요.")
            else:
                with st.spinner("Fitting (Optuna search + GP residual head)…"):
                    ds = Dataset(
                        compositions=df[el_cols].copy(),
                        properties=df[tgt_cols].copy(),
                        groups=df[group_col].copy() if group_col != "(none)" else None,
                    )
                    feat = CompositionFeaturizer(element_columns=el_cols)
                    fm = ForwardModel(featurizer=feat, targets=tgt_cols,
                                      n_cv_splits=int(cv))
                    fm.fit(ds, n_trials=int(n_trials))
                    conformal = ConformalCalibrator(alpha=0.1).calibrate(fm, ds)
                    doa = DomainOfApplicability().fit(fm, ds)
                st.session_state.af_model = fm
                st.session_state.af_conformal = conformal
                st.session_state.af_doa = doa
                st.session_state.af_element_cols = el_cols
                st.session_state.af_target_cols = tgt_cols
                st.success("학습 완료.")

        if st.session_state.af_model is not None:
            metrics = pd.DataFrame(st.session_state.af_model.metrics_).T
            st.markdown("### CV metrics (group-aware when groups provided)")
            st.dataframe(metrics.round(4), use_container_width=True)


# ------------------------------------------------------------------ PREDICT
with tab_predict:
    st.subheader("예측 · Conformal 90% 신뢰구간 · DoA score")
    fm = st.session_state.af_model
    if fm is None:
        st.info("먼저 모델을 학습하세요.")
    else:
        el_cols = st.session_state.af_element_cols
        cols = st.columns(min(5, len(el_cols)))
        comp = {}
        for i, e in enumerate(el_cols):
            with cols[i % len(cols)]:
                comp[e] = st.number_input(e, 0.0, 1.0,
                                          round(1.0 / len(el_cols), 4),
                                          step=0.01, format="%.4f",
                                          key=f"af_pred_{e}")
        total = sum(comp.values())
        st.write(f"Sum = {total:.4f}  {'✅' if abs(total - 1) < 1e-3 else '⚠'}")
        if st.button("예측 실행", key="af_predict_btn"):
            q = pd.DataFrame([comp]).reindex(columns=el_cols, fill_value=0)
            preds = fm.predict(q)
            intervals = st.session_state.af_conformal.intervals(preds)
            X = fm.featurizer.transform(q)
            first = next(iter(fm.models_.values()))
            X_s = first.preproc.transform(X[first.feature_names])
            doa_score = float(st.session_state.af_doa.score(X_s)[0])

            st.markdown("**Predicted properties (μ ± conformal 90%)**")
            rows = []
            for tgt in st.session_state.af_target_cols:
                rows.append({
                    "property": tgt,
                    "mean": float(preds[f"{tgt}_mean"].iloc[0]),
                    "std (GP)": float(preds[f"{tgt}_std"].iloc[0]),
                    "lo (90%)": float(intervals[f"{tgt}_lo"].iloc[0]),
                    "hi (90%)": float(intervals[f"{tgt}_hi"].iloc[0]),
                })
            st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
            tag = "✅ in domain" if doa_score <= 0.95 else "⚠ EXTRAPOLATION"
            st.metric("DoA score", f"{doa_score:.3f}", help=tag)


# ------------------------------------------------------------------- DESIGN
with tab_design:
    st.subheader("NSGA-II 역설계 (risk-aware: μ − λσ)")
    fm = st.session_state.af_model
    if fm is None:
        st.info("먼저 모델을 학습하세요.")
    else:
        el_cols = st.session_state.af_element_cols
        tgt_cols = st.session_state.af_target_cols

        st.markdown("#### Objectives")
        objectives = []
        for t in tgt_cols:
            c1, c2 = st.columns([2, 2])
            with c1:
                d = st.selectbox(f"{t}: direction",
                                 ["max", "min", "target", "(skip)"],
                                 key=f"af_obj_{t}")
            with c2:
                tv = st.number_input(f"{t}: target value (direction=target일 때)",
                                     value=0.0, key=f"af_tv_{t}")
            if d != "(skip)":
                objectives.append((t, d, tv if d == "target" else None))

        st.markdown("#### Element bounds")
        bounds: dict[str, tuple[float, float]] = {}
        bc = st.columns(min(5, len(el_cols)))
        for i, e in enumerate(el_cols):
            with bc[i % len(bc)]:
                lo = st.number_input(f"{e} min", 0.0, 1.0, 0.0, 0.01,
                                     key=f"af_lo_{e}")
                hi = st.number_input(f"{e} max", 0.0, 1.0, 0.5, 0.01,
                                     key=f"af_hi_{e}")
                bounds[e] = (lo, hi)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            pop = st.number_input("Population", 16, 400, 60, 4, key="af_pop")
        with c2:
            gens = st.number_input("Generations", 5, 200, 30, 1, key="af_gen")
        with c3:
            lam = st.number_input("Risk λ", 0.0, 5.0, 1.0, 0.1, key="af_lam")
        with c4:
            topk = st.number_input("Top K", 1, 50, 5, 1, key="af_topk")

        if st.button("NSGA-II 실행", type="primary", key="af_design_btn"):
            if not objectives:
                st.error("최소 하나 이상의 objective를 설정하세요.")
            else:
                with st.spinner("Running NSGA-II…"):
                    spec_objs = [(t, d) for t, d, _ in objectives]
                    spec_targets = {t: v for t, d, v in objectives if d == "target" and v is not None}
                    spec = DesignSpec(
                        objectives=spec_objs,
                        element_bounds=bounds,
                        target_values=spec_targets,
                        risk_lambda=float(lam),
                        feasibility=default_checker(el_cols),
                    )
                    designer = InverseDesigner(model=fm, spec=spec,
                                               element_columns=el_cols)
                    front = designer.run_nsga2(pop_size=int(pop), n_gen=int(gens))
                st.session_state.af_candidates = front.head(int(topk))
                st.success(f"Pareto front: {len(front)}")

        if st.session_state.af_candidates is not None:
            st.dataframe(st.session_state.af_candidates.round(4),
                         use_container_width=True)


# ------------------------------------------------------------------ EXPLAIN
with tab_explain:
    st.subheader("SHAP 설명 + LLM 해석")
    fm = st.session_state.af_model
    df = st.session_state.af_dataset
    if fm is None or df is None:
        st.info("먼저 모델을 학습하세요.")
    else:
        el_cols = st.session_state.af_element_cols
        tgt_cols = st.session_state.af_target_cols
        cands = st.session_state.af_candidates
        default_comp = (
            {e: float(cands[e].iloc[0]) for e in el_cols}
            if cands is not None and len(cands) > 0
            else {e: round(1.0 / len(el_cols), 4) for e in el_cols}
        )

        c1, c2 = st.columns([3, 1])
        with c1:
            tgt = st.selectbox("Target to explain", tgt_cols, key="af_exp_tgt")
        with c2:
            llm_btn = st.checkbox("LLM 해석 포함", value=True, key="af_exp_llm")

        cols = st.columns(min(5, len(el_cols)))
        comp = {}
        for i, e in enumerate(el_cols):
            with cols[i % len(cols)]:
                comp[e] = st.number_input(e, 0.0, 1.0, default_comp[e], 0.01,
                                          format="%.4f", key=f"af_exp_{e}")
        if st.button("설명 생성", key="af_exp_btn"):
            with st.spinner("SHAP + (옵션) Claude…"):
                q = pd.DataFrame([comp])
                expl = Explainer(model=fm)
                shap_df = expl.explain(q, target=tgt, background_df=df[el_cols])
                glob = expl.global_importance(tgt, df[el_cols]).head(10)

            sample = shap_df[shap_df["sample_id"] == 0]
            top = sample.reindex(sample.shap.abs()
                                 .sort_values(ascending=False).index).head(10)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Local SHAP (top 10)**")
                st.dataframe(top[["feature", "value", "shap"]].round(4),
                             use_container_width=True)
            with c2:
                st.markdown("**Global importance (top 10)**")
                st.dataframe(glob.round(4), use_container_width=True)

            if llm_btn:
                preds = fm.predict(q).iloc[0].to_dict()
                assistant = LLMAssistant()
                interp = assistant.interpret_prediction(
                    composition=comp,
                    prediction=preds,
                    shap_top=top[["feature", "value", "shap"]].to_dict(orient="records"),
                )
                st.markdown("**Interpretation**")
                st.markdown(interp)
                st.caption(f"used_llm = {assistant.available}")


# ------------------------------------------------------------------------ AL
with tab_al:
    st.subheader("Active Learning — 다음 실험 배치 추천")
    fm = st.session_state.af_model
    df = st.session_state.af_dataset
    if fm is None or df is None:
        st.info("먼저 모델을 학습하세요.")
    else:
        el_cols = st.session_state.af_element_cols
        c1, c2 = st.columns(2)
        with c1:
            pool_n = st.number_input("후보 pool 크기", 50, 5000, 500, 50, key="af_pooln")
        with c2:
            batch = st.number_input("Batch size", 1, 20, 5, 1, key="af_batch")
        if st.button("배치 추천", key="af_al_btn"):
            rng = np.random.default_rng(0)
            pool_comp = rng.dirichlet(np.ones(len(el_cols)), size=int(pool_n))
            pool = pd.DataFrame(pool_comp, columns=el_cols)
            learner = ActiveLearner(model=fm)
            picks = learner.sample_uncertainty(
                candidate_pool=pool,
                element_columns=el_cols,
                batch_size=int(batch),
            )
            st.success(f"{len(picks)} candidates selected")
            st.dataframe(picks.round(4), use_container_width=True)
