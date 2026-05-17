"""Streamlit page: High-Throughput compound screening (Cho 2025 workflow).

Pulls every compound from the bundled Nb-host DB (or OQMD on demand),
scores them on three thermodynamic descriptors (tie line with host
matrix, ΔH stability, lattice coherency), ranks, and shows the top
candidates with rationale.

The page mirrors the NSM Lab slide-deck design but with weights the
user can adjust at runtime.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from core.alloyforge.hts_screening import (
    HOSTS,
    NB_HOST_COMPOUNDS,
    ScoreWeights,
    host_plus_precipitate_composition,
    rank_compounds,
)
from core.alloyforge.oqmd_client import query_oqmd, to_known_compounds

st.set_page_config(page_title="HTS 화합물 스크리닝", page_icon="⚗️", layout="wide")
st.title("⚗️ High-Throughput Compound Screening")
st.caption(
    "OQMD + curated bundle → 3 descriptor 채점 (host matrix 평형 / 안정성 / "
    "정합성) → ranking → ML 예측. Cho 2025 workflow를 Nb 합금에 적용."
)


def _ss_default(k, v):
    if k not in st.session_state:
        st.session_state[k] = v


_ss_default("hts_ranking", None)
_ss_default("hts_compounds_source", "bundled")


# ---------------------------------------------------------------------------
tab_bundled, tab_rank, tab_oqmd, tab_mix = st.tabs([
    "1. 큐레이팅된 화합물 DB",
    "2. Score · Rank",
    "3. OQMD 검색",
    "4. ML 예측 (host + precipitate)",
])


# Tab 1
with tab_bundled:
    st.subheader(f"Nb-host 화합물 DB ({len(NB_HOST_COMPOUNDS)}개)")
    rows = []
    for c in NB_HOST_COMPOUNDS:
        rows.append({
            "formula": c.formula,
            "elements": ", ".join(c.elements),
            "space_group": c.space_group,
            "a (Å)": c.lattice_a, "c (Å)": c.lattice_c or "-",
            "ΔH (eV/atom)": c.formation_energy_per_atom_ev,
            "tie line": ", ".join(c.has_direct_tie_line_with) or "-",
            "source": c.source,
            "notes": c.notes,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=500)


# Tab 2
with tab_rank:
    st.subheader("Score · Rank")
    c1, c2, c3 = st.columns(3)
    with c1:
        host = st.selectbox("Host matrix", list(HOSTS.keys()),
                             index=list(HOSTS.keys()).index("Nb"))
    with c2:
        req = st.text_input(
            "required_elements (쉼표; 비워두면 무필터)",
            value="Nb,Si",
        )
    with c3:
        forb = st.text_input(
            "forbidden_elements (쉼표)",
            value="",
        )

    st.markdown("**Score weights**")
    w1, w2, w3 = st.columns(3)
    with w1:
        w_tie = st.slider("tie_line", 0.0, 3.0, 1.0, 0.1)
    with w2:
        w_sta = st.slider("stability", 0.0, 3.0, 1.0, 0.1)
    with w3:
        w_coh = st.slider("coherency", 0.0, 3.0, 1.0, 0.1)

    min_tie = st.slider("min_tie_line_score", 0.0, 1.0, 0.0, 0.1)
    top_k = st.number_input("top K", 3, 50, 10, 1)

    if st.button("Rank 실행", type="primary"):
        rl = [e.strip() for e in req.split(",") if e.strip()]
        fl = [e.strip() for e in forb.split(",") if e.strip()]
        df = rank_compounds(
            host=host,
            weights=ScoreWeights(w_tie, w_sta, w_coh),
            required_elements=rl or None,
            forbidden_elements=fl or None,
            min_tie_line_score=min_tie,
            top_k=int(top_k),
        )
        st.session_state.hts_ranking = df
        st.success(f"{len(df)} compounds ranked")

    if st.session_state.hts_ranking is not None:
        st.dataframe(st.session_state.hts_ranking, use_container_width=True)


# Tab 3
with tab_oqmd:
    st.subheader("OQMD 라이브 검색")
    st.caption(
        "OQMD REST API에서 binary / ternary / quaternary 화합물 데이터 fetch. "
        "Sandbox 환경에서는 외부 HTTP 차단으로 빈 결과가 반환될 수 있습니다 — "
        "로컬에서 실행하세요."
    )
    el_text = st.text_input("element_set (쉼표)", "Nb,Si,Ti")
    formula = st.text_input("formula (선택)", "")
    n_atoms = st.number_input("최대 atoms / unit cell", 2, 100, 30)
    page = st.number_input("page_size", 5, 100, 25)
    if st.button("OQMD 조회"):
        with st.spinner("OQMD fetching..."):
            df_oqmd = query_oqmd(
                elements=[e.strip() for e in el_text.split(",") if e.strip()],
                formula=formula or None,
                n_atoms_max=int(n_atoms),
                page_size=int(page),
            )
        st.write(f"**{len(df_oqmd)} hits**")
        st.dataframe(df_oqmd, use_container_width=True)
        if not df_oqmd.empty and st.button("Rank these"):
            compounds = to_known_compounds(df_oqmd)
            ranked = rank_compounds(host="Nb", compounds=compounds)
            st.dataframe(ranked, use_container_width=True)


# Tab 4
with tab_mix:
    st.subheader("ML 예측: host + 석출상 조성으로 forward model 호출")
    c1, c2 = st.columns(2)
    with c1:
        h = st.selectbox("Host", list(HOSTS.keys()),
                          index=list(HOSTS.keys()).index("Nb"), key="mix_h")
    with c2:
        names = [c.formula for c in NB_HOST_COMPOUNDS]
        compound_name = st.selectbox("Compound", names,
                                      index=names.index("Nb5Si3-alpha"))
    frac = st.slider("Precipitate atomic fraction",
                     0.0, 0.5, 0.10, 0.01)
    if st.button("조성 생성", key="mix_btn"):
        comp_def = next(c for c in NB_HOST_COMPOUNDS
                        if c.formula == compound_name)
        comp = host_plus_precipitate_composition(HOSTS[h], comp_def, frac)
        st.markdown("**생성된 조성 (atomic fraction)**")
        st.json({k: round(v, 4) for k, v in comp.items()})
        st.caption(
            "이 조성을 page 7/8의 forward model에 넣어 hardness/strength를 "
            "예측하고, page 5의 NSGA-II 역설계와 비교할 수 있습니다."
        )
