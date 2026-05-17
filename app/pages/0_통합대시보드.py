"""Unified dashboard — single entry for the full Nb-alloy design platform.

What you can do on this page:

* See platform status at a glance (data, models, available features).
* Train a forward model on the reference DB + your CSV in 2 clicks.
* Predict properties for a custom composition with calibrated σ.
* Run HTS compound screening for the host matrix.
* Inverse-design candidates that match a target spec.
* Get an active-learning batch of "what to make next" recommendations.
* See an accuracy report with CV / permutation / coverage.

This is intended to be the canonical research-grade UI; other pages
exist for power users who want fine-grained control.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from core.alloyforge import (
    CompositionFeaturizer,
    Dataset,
    ForwardModel,
    NB_HOST_COMPOUNDS,
    PROPERTY_COLUMNS,
    rank_compounds,
    reference_dataset,
    reference_elements,
)
from core.alloyforge.active_learning_planner import (
    ExperimentPlanner,
    PlannerWeights,
)
from core.alloyforge.feasibility import default_checker
from core.alloyforge.hts_descriptor import HTSScoreFeaturizer
from core.alloyforge.hts_screening import HOSTS, ScoreWeights
from core.alloyforge.inverse_design import DesignSpec, InverseDesigner
from core.alloyforge.microstructure_features import (
    PhaseFractionFeaturizer,
    load_cleaned_nb_si,
)

st.set_page_config(
    page_title="통합 대시보드",
    page_icon="🧬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_default("dash_model", None)
_ss_default("dash_dataset", None)
_ss_default("dash_elements", None)
_ss_default("dash_targets", None)
_ss_default("dash_planner", None)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧬 통합 대시보드 — Nb 합금 설계 플랫폼")
st.caption(
    "Reference DB + 사용자 CSV + HTS 화합물 ranking + 미세조직 phase fraction "
    "→ ML 학습 → 예측 / 역설계 / 다음 실험 추천"
)

# Quick status row
c1, c2, c3, c4 = st.columns(4)
with c1:
    n_alloys = len(reference_dataset())
    st.metric("참조 합금", f"{n_alloys}")
with c2:
    st.metric("HTS 화합물 DB", f"{len(NB_HOST_COMPOUNDS)}")
with c3:
    cleaned_path = ROOT / "data" / "nb_si" / "nb_silicide_cleaned.csv"
    n_cleaned = 0
    if cleaned_path.exists():
        n_cleaned = len(load_cleaned_nb_si(str(cleaned_path)))
    st.metric("Nb-Si cleaned rows", f"{n_cleaned}")
with c4:
    if st.session_state.dash_model is not None:
        st.metric("학습된 모델", "✅")
    else:
        st.metric("학습된 모델", "—")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_train, tab_predict, tab_hts, tab_inverse, tab_al, tab_report = st.tabs([
    "1. 학습", "2. 예측", "3. HTS ranking",
    "4. 역설계", "5. 다음 실험", "6. 정확도 리포트",
])


# ============================ Tab 1: training ============================
with tab_train:
    st.subheader("학습 데이터 선택 및 모델 학습")
    c1, c2 = st.columns([3, 2])
    with c1:
        source = st.radio(
            "학습 소스",
            ["참조 DB만 (38 alloys)",
             "참조 DB + Nb-Si cleaned",
             "Nb-Si cleaned만 (622 rows)"],
            index=0,
        )
        st.caption(
            "Cleaned 데이터는 phase fraction features까지 사용. "
            "참조 DB만 쓰면 빠르지만 Nb 합금엔 약함."
        )

    with c2:
        target_options = ["yield_mpa", "tensile_mpa", "hardness_hv",
                          "density_gcc", "Vickers_hardness_(Hv)"]
        targets_pick = st.multiselect(
            "예측할 타겟",
            target_options,
            default=["yield_mpa", "tensile_mpa", "hardness_hv"],
        )
        n_trials = st.slider("Optuna trials", 2, 30, 8)

    if st.button("학습 시작", type="primary"):
        with st.spinner("Training…"):
            try:
                df, els, ds, tgt_final = _build_training_set(
                    source, targets_pick,
                )
                fm = ForwardModel(
                    featurizer=CompositionFeaturizer(element_columns=els),
                    targets=tgt_final, n_cv_splits=5,
                )
                fm.fit(ds, n_trials=int(n_trials))
                st.session_state.dash_model = fm
                st.session_state.dash_dataset = ds
                st.session_state.dash_elements = els
                st.session_state.dash_targets = tgt_final
                st.session_state.dash_planner = None  # invalidate
                st.success(f"학습 완료. CV R²:")
                rep_df = pd.DataFrame(fm.metrics_).T.round(3)
                st.dataframe(rep_df, use_container_width=True)
            except Exception as exc:
                st.error(f"학습 실패: {exc}")
                raise


def _build_training_set(source: str, target_picks: List[str]):
    """Returns (full_df, element_cols, Dataset, target_columns)."""
    if "Nb-Si cleaned" in source:
        df = load_cleaned_nb_si(str(ROOT / "data" / "nb_si" / "nb_silicide_cleaned.csv"))
        # Compose element set from cleaned columns that intersect with what's in df
        all_elements = ["Nb", "Si", "Ti", "Cr", "Al", "Hf", "Mo", "W", "Ta",
                        "Zr", "Y", "B", "C", "Fe", "Ga", "Ge", "Co", "V",
                        "Mg", "Ni", "Sn", "Re"]
        els = [e for e in all_elements if e in df.columns]
        for c in els:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        # Normalise atomic fractions (CSV is in atomic % already in this file)
        sums = df[els].sum(axis=1)
        df[els] = df[els].div(sums.replace(0, 1.0), axis=0)
        # Filter targets to columns that exist
        valid_tgts = [t for t in target_picks if t in df.columns]
        if not valid_tgts:
            # Default fallback: Vickers_hardness_(Hv)
            valid_tgts = [t for t in
                          ["Vickers_hardness_(Hv)", "Yield_Strength_(Mpa)",
                           "Tensile_Strength_(MPa)"]
                          if t in df.columns]
        df = df.dropna(subset=valid_tgts).reset_index(drop=True)
        # Add phase features as process columns
        ms = PhaseFractionFeaturizer()
        X_ms = ms.transform(df)
        process_df = X_ms.copy()
        # Family / group key for GroupKFold: top 2 alloying elements
        df["_family"] = df[els].apply(
            lambda r: "Nb-" + "-".join(
                sorted(
                    [e for e, v in r.items() if e != "Nb" and v > 0.02],
                    key=lambda e: -r[e],
                )[:2]
            ),
            axis=1,
        )
        ds_full = Dataset(
            compositions=df[els],
            properties=df[valid_tgts],
            process=process_df,
            groups=df["_family"],
        )
        if "참조 DB +" in source:
            ref = reference_dataset()
            ref_els = reference_elements()
            common = [e for e in els if e in ref_els]
            ref_filtered = ref.reindex(columns=ref_els + valid_tgts, fill_value=0.0)
            ref_filtered = ref_filtered.dropna(subset=valid_tgts).reset_index(drop=True)
            # Hack: only proceed if at least one target is shared
            if len(ref_filtered):
                # phase features for ref are zero
                phase_zero = pd.DataFrame(
                    0, index=ref_filtered.index, columns=ms.feature_names,
                )
                ref_ds = Dataset(
                    compositions=ref_filtered.reindex(columns=els, fill_value=0.0),
                    properties=ref_filtered[valid_tgts],
                    process=phase_zero,
                    groups=pd.Series(["ref_" + f for f in ref_filtered.get("family", ["misc"] * len(ref_filtered))]),
                )
                # Concat
                comp_all = pd.concat([ds_full.compositions, ref_ds.compositions],
                                      ignore_index=True)
                prop_all = pd.concat([ds_full.properties, ref_ds.properties],
                                      ignore_index=True)
                proc_all = pd.concat([ds_full.process, ref_ds.process],
                                      ignore_index=True)
                grp_all = pd.concat([ds_full.groups, ref_ds.groups],
                                     ignore_index=True)
                ds_full = Dataset(
                    compositions=comp_all,
                    properties=prop_all,
                    process=proc_all,
                    groups=grp_all,
                )
        return df, els, ds_full, valid_tgts
    else:
        # Reference DB only
        ref = reference_dataset()
        els = reference_elements()
        valid_tgts = [t for t in target_picks if t in ref.columns]
        if not valid_tgts:
            valid_tgts = ["yield_mpa", "tensile_mpa", "hardness_hv"]
        df = ref.dropna(subset=valid_tgts).reset_index(drop=True)
        return df, els, Dataset(
            compositions=df[els],
            properties=df[valid_tgts],
            groups=df["family"],
        ), valid_tgts


# ============================ Tab 2: predict ============================
with tab_predict:
    st.subheader("Composition → 물성 예측")
    fm = st.session_state.dash_model
    els = st.session_state.dash_elements
    if fm is None:
        st.info("먼저 모델을 학습하세요 (탭 1).")
    else:
        st.caption("원소 atomic fraction을 입력하세요 (합=1).")
        cols = st.columns(min(6, len(els)))
        comp = {}
        for i, e in enumerate(els):
            with cols[i % len(cols)]:
                default = 0.5 if e == "Nb" else (
                    0.16 if e == "Si" else 0.0
                )
                comp[e] = st.number_input(
                    e, 0.0, 1.0, default, 0.005, format="%.4f",
                    key=f"dash_pred_{e}",
                )
        total = sum(comp.values())
        st.write(f"Sum = {total:.4f}  "
                 f"{'✅' if abs(total-1)<1e-3 else '⚠ normalise 필요'}")

        if st.button("예측 실행", key="dash_predict_btn"):
            q = pd.DataFrame([comp])
            # If model trained with process columns, supply zeros
            try:
                preds = fm.predict(q).iloc[0]
            except Exception:
                ms = PhaseFractionFeaturizer()
                process = pd.DataFrame(
                    0, index=[0], columns=ms.feature_names,
                )
                preds = fm.predict(q, process=process).iloc[0]
            rows = []
            for t in st.session_state.dash_targets:
                rows.append({
                    "property": t,
                    "predicted": round(float(preds[f"{t}_mean"]), 3),
                    "σ": round(float(preds[f"{t}_std"]), 3),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # HTS context for this composition
            st.markdown("**HTS 컨텍스트 (이 조성과 매칭되는 화합물)**")
            hts_feat = HTSScoreFeaturizer()
            scores = hts_feat.transform(q).iloc[0]
            for k, v in scores.items():
                st.text(f"  {k:30s}  {v:.3f}")


# ============================ Tab 3: HTS ranking ============================
with tab_hts:
    st.subheader("HTS 화합물 ranking (host matrix 평형 / 안정성 / 정합성)")
    c1, c2 = st.columns(2)
    with c1:
        host = st.selectbox(
            "Host", list(HOSTS.keys()),
            index=list(HOSTS.keys()).index("Nb"),
            key="dash_hts_host",
        )
    with c2:
        req = st.text_input(
            "required_elements (쉼표)", "Nb,Si", key="dash_hts_req"
        )
    w1, w2, w3 = st.columns(3)
    with w1:
        w_t = st.slider("w(tie_line)", 0.0, 3.0, 1.0, 0.1, key="dash_hts_wt")
    with w2:
        w_s = st.slider("w(stability)", 0.0, 3.0, 1.0, 0.1, key="dash_hts_ws")
    with w3:
        w_c = st.slider("w(coherency)", 0.0, 3.0, 1.0, 0.1, key="dash_hts_wc")
    rl = [e.strip() for e in req.split(",") if e.strip()]
    df_hts = rank_compounds(
        host=host,
        weights=ScoreWeights(w_t, w_s, w_c),
        required_elements=rl or None,
    )
    st.dataframe(df_hts.head(15), use_container_width=True)


# ============================ Tab 4: inverse design =========================
with tab_inverse:
    st.subheader("NSGA-II 역설계 (max hardness/tensile)")
    fm = st.session_state.dash_model
    els = st.session_state.dash_elements
    tgts = st.session_state.dash_targets
    if fm is None:
        st.info("먼저 모델을 학습하세요 (탭 1).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            obj_target = st.selectbox("최대화할 타겟", tgts)
        with c2:
            pop = st.slider("Population", 16, 200, 32, 16)
        with c3:
            gens = st.slider("Generations", 10, 60, 20, 5)

        # Element bounds: Nb-rich Nb-Si-X
        bounds = {
            "Nb": (0.40, 0.85), "Si": (0.05, 0.25),
            "Ti": (0.0, 0.25), "Cr": (0.0, 0.15),
            "Al": (0.0, 0.10), "Hf": (0.0, 0.10),
            "Mo": (0.0, 0.15), "W": (0.0, 0.15),
        }
        for el in els:
            bounds.setdefault(el, (0.0, 1e-6))

        if st.button("역설계 실행", key="dash_inv_btn"):
            with st.spinner("NSGA-II running…"):
                spec = DesignSpec(
                    objectives=[(obj_target, "max")],
                    element_bounds=bounds,
                    risk_lambda=0.5,
                    feasibility=default_checker(els),
                )
                designer = InverseDesigner(
                    model=fm, spec=spec, element_columns=els,
                )
                try:
                    front = designer.run_nsga2(
                        pop_size=int(pop), n_gen=int(gens), seed=0,
                    )
                except Exception as exc:
                    st.error(f"역설계 실패: {exc}")
                    front = pd.DataFrame()
            if not front.empty:
                top = front.sort_values("agg_score").head(5)
                show = [c for c in els[:8] if c in top.columns]
                show += [f"{obj_target}_mean", f"{obj_target}_std", "agg_score"]
                show = [c for c in show if c in top.columns]
                st.dataframe(top[show].round(3), use_container_width=True)


# ============================ Tab 5: active learning ========================
with tab_al:
    st.subheader("다음 실험 추천 (uncertainty + HTS + DoA)")
    fm = st.session_state.dash_model
    els = st.session_state.dash_elements
    tgts = st.session_state.dash_targets
    ds = st.session_state.dash_dataset
    if fm is None or ds is None:
        st.info("먼저 모델을 학습하세요 (탭 1).")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            pool_n = st.slider("Candidate pool size", 100, 5000, 1000, 100)
        with c2:
            batch = st.slider("Batch size", 1, 20, 5)
        with c3:
            w_sigma = st.slider("w(σ)", 0.0, 3.0, 1.0, 0.1)
        w_hts = st.slider("w(HTS)", 0.0, 3.0, 0.5, 0.1)
        w_doa = st.slider("w(DoA)", 0.0, 3.0, 0.5, 0.1)

        if st.button("배치 추천", key="dash_al_btn"):
            with st.spinner("Sampling candidate pool + scoring…"):
                rng = np.random.default_rng(0)
                # Generate Dirichlet candidates within the Nb-Si typical range
                pool_arr = rng.dirichlet(
                    np.array([5.0 if e == "Nb" else 1.0 for e in els]),
                    size=int(pool_n),
                )
                pool = pd.DataFrame(pool_arr, columns=els)
                planner = (
                    st.session_state.dash_planner
                    or ExperimentPlanner(
                        model=fm,
                        weights=PlannerWeights(w_sigma, w_hts, w_doa),
                    ).fit(ds)
                )
                planner.weights = PlannerWeights(w_sigma, w_hts, w_doa)
                # Process pool: zeros for phase features if used
                process_pool = None
                if ds.process is not None:
                    process_pool = pd.DataFrame(
                        0, index=range(len(pool)),
                        columns=ds.process.columns,
                    )
                picks = planner.propose(
                    pool, tgts, process_pool=process_pool,
                    batch_size=int(batch),
                )
                st.session_state.dash_planner = planner
            st.dataframe(picks.round(3), use_container_width=True)


# ============================ Tab 6: accuracy report ========================
with tab_report:
    st.subheader("정확도 & 신뢰도 리포트")
    fm = st.session_state.dash_model
    ds = st.session_state.dash_dataset
    tgts = st.session_state.dash_targets
    if fm is None or ds is None:
        st.info("먼저 모델을 학습하세요 (탭 1).")
    else:
        st.caption("Permutation test는 시간이 걸립니다. 빠른 확인은 fast 모드 사용.")
        fast = st.checkbox("Fast (permutation 건너뜀)", True)
        if st.button("리포트 생성", key="dash_rep_btn"):
            from core.alloyforge.accuracy_report import evaluate_model
            with st.spinner("Evaluating…"):
                rep = evaluate_model(
                    fm, ds, targets=tgts,
                    n_splits=3, n_seeds=1, n_permutations=3,
                    skip_permutation=fast, skip_reliability=False,
                    include_reference_check=False,
                )
            st.code(rep.summary())
            if rep.cv:
                st.markdown("**CV metrics**")
                cv_rows = []
                for t, m in rep.cv.items():
                    cv_rows.append({"target": t, **m})
                st.dataframe(pd.DataFrame(cv_rows).round(4),
                             use_container_width=True)
            if rep.coverage:
                st.markdown("**Conformal coverage**")
                cov_rows = []
                for t, c in rep.coverage.items():
                    cov_rows.append({"target": t, **c})
                st.dataframe(pd.DataFrame(cov_rows).round(3),
                             use_container_width=True)
            st.metric("Overall grade", rep.overall_grade)
