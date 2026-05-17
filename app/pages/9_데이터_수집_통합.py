"""Streamlit page: collect, ingest, and merge data from every source.

Five tabs:

1. **참조 DB**: browse / filter the 38-alloy curated reference table.
2. **CSV 업로드**: drag-and-drop user CSV; auto-detect units; normalise
   composition; flag outliers; preview before merge.
3. **외부 검색**: query OpenAlex / arXiv / CrossRef / Materials Project
   (those without API keys gracefully report status).
4. **LLM 표 추출**: paste paper text → Claude extracts structured rows
   with confidence flags → merge into the working dataset.
5. **통합 + 다운로드**: merge every staged source via `merge_datasets`
   with a `source` group column and download the final CSV.

The "working dataset" lives in ``st.session_state.working_dataset`` so
you can add to it across tabs without losing state.
"""

from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
import streamlit as st

from core.alloyforge.data_ingestion import (
    flag_outliers,
    infer_units,
    merge_datasets,
    normalize_composition,
    normalize_units,
)
from core.alloyforge.external_data import (
    materials_project_summary,
    provider_status,
    search_arxiv,
    search_crossref,
    search_openalex,
)
from core.alloyforge.llm_table_extractor import extract_alloy_table
from core.alloyforge.reference_data import (
    PROPERTY_COLUMNS,
    find_alloy,
    reference_dataset,
    reference_elements,
    reference_families,
)


st.set_page_config(page_title="데이터 수집 통합", page_icon="🗂️", layout="wide")
st.title("🗂️ 데이터 수집 · 통합 · 정규화")
st.caption(
    "Reference DB · user CSV · external APIs · LLM table extraction — "
    "everything funnels into one `merge_datasets()` call with a "
    "`source` group column so train/test never leaks across origins."
)


def _ss_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


_ss_default("data_sources", {})        # name -> DataFrame
_ss_default("ingest_element_cols", ["Fe", "Ni", "Cr", "Mo"])
_ss_default("ingest_target_cols", ["yield_mpa", "tensile_mpa", "hardness_hv"])


def _staged_summary():
    if not st.session_state.data_sources:
        st.info("아직 단계화된 소스가 없습니다.")
        return
    rows = []
    for src, df in st.session_state.data_sources.items():
        rows.append({"source": src, "rows": len(df), "cols": len(df.columns)})
    st.write("**단계화된 소스**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


tab_ref, tab_csv, tab_ext, tab_llm, tab_merge = st.tabs(
    ["1. 참조 DB", "2. CSV 업로드", "3. 외부 검색",
     "4. LLM 추출", "5. 통합·다운로드"]
)


# ---------------------------------------------------------------------------
# Tab 1: reference DB
# ---------------------------------------------------------------------------
with tab_ref:
    st.subheader("38-alloy 참조 데이터베이스")
    c1, c2, c3 = st.columns(3)
    with c1:
        family_filter = st.text_input("Family 필터 (부분 일치)", "")
    with c2:
        drop_missing = st.checkbox("물성 NaN 행 제거", False)
    with c3:
        single = st.text_input("단일 합금 조회 (이름)", "")

    df_ref = reference_dataset(drop_missing_targets=drop_missing)
    if family_filter:
        df_ref = df_ref[df_ref["family"].str.contains(family_filter, case=False)]

    st.write(f"**{len(df_ref)} 합금** · families {len(reference_families())} · "
             f"elements {len(reference_elements())}")
    st.dataframe(df_ref.round(4), use_container_width=True, height=300)

    if single:
        a = find_alloy(single)
        if a is None:
            st.warning(f"'{single}' 없음")
        else:
            st.json({
                "name": a.name, "family": a.family,
                "composition_weight_pct": a.composition_wt,
                "composition_atomic_frac": {k: round(v, 4)
                                             for k, v in a.as_atomic().items()},
                "properties": {
                    p: getattr(a, p) for p in
                    ["yield_mpa", "tensile_mpa", "elong_pct", "hardness_hv",
                     "density_gcc", "youngs_gpa", "melting_k"]
                },
                "references": a.references,
            })

    if st.button("이 참조 DB를 통합용 소스로 추가", key="ref_stage"):
        st.session_state.data_sources["reference_db"] = df_ref.copy()
        st.success(f"reference_db 단계화 완료 ({len(df_ref)} rows)")


# ---------------------------------------------------------------------------
# Tab 2: CSV upload
# ---------------------------------------------------------------------------
with tab_csv:
    st.subheader("사용자 CSV / Excel 업로드")
    up = st.file_uploader("파일 선택 (.csv / .xlsx)",
                          type=["csv", "xlsx"], key="csv_up")
    if up is not None:
        try:
            df = pd.read_excel(up) if up.name.endswith(".xlsx") else pd.read_csv(up)
        except Exception as exc:
            st.error(f"파일 읽기 실패: {exc}")
            df = None
    else:
        df = None

    if df is not None:
        st.write(f"**원본 shape**: {df.shape}")
        st.dataframe(df.head(10), use_container_width=True)

        units = infer_units(df)
        st.write("**자동 감지된 단위:**")
        st.json(units)

        c1, c2 = st.columns(2)
        with c1:
            basis = st.selectbox(
                "Composition basis",
                ["auto", "atomic_frac", "atomic_pct", "weight_pct"],
            )
        with c2:
            stage_name = st.text_input(
                "Source 라벨", value=up.name if up else "user_upload",
            )

        if st.button("정규화 + 단계화", key="csv_stage"):
            # Try infer element columns by intersecting with known element list
            from core.alloyforge.data_pipeline import ELEMENT_PROPERTIES
            el_cols = [c for c in df.columns if c in ELEMENT_PROPERTIES]
            if not el_cols:
                st.error("element 컬럼을 찾을 수 없습니다 (Fe, Ni, Cr ... 등)")
            else:
                normed = normalize_units(df, units)
                normed = normalize_composition(normed, el_cols, basis)
                target_cols_guess = [c for c in normed.columns
                                      if c in PROPERTY_COLUMNS or
                                      c.endswith("_mpa") or c.endswith("_hv")]
                normed = flag_outliers(normed, target_cols_guess)
                st.session_state.data_sources[stage_name] = normed
                st.success(f"{stage_name} 단계화 ({len(normed)} rows; "
                           f"{normed.get('is_outlier', pd.Series([0])).sum()} outliers)")
                st.dataframe(normed.head(10), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: external search
# ---------------------------------------------------------------------------
with tab_ext:
    st.subheader("외부 API 검색 (메타데이터)")
    status = provider_status()
    st.json(status)
    st.caption(
        "OpenAlex / arXiv / CrossRef는 인증 불필요. "
        "Materials Project는 환경 변수 `MP_API_KEY` 필요. "
        "ESC sandbox에서는 외부 HTTP가 차단될 수 있습니다 — "
        "로컬에서 실행하세요."
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        q = st.text_input("쿼리", "high entropy alloy")
    with c2:
        provider = st.selectbox(
            "Provider",
            ["openalex", "arxiv", "crossref", "materials_project"],
        )

    if st.button("검색", key="ext_search"):
        with st.spinner(f"{provider} 조회 중..."):
            if provider == "openalex":
                df = search_openalex(q, per_page=15)
            elif provider == "arxiv":
                df = search_arxiv(q, max_results=15)
            elif provider == "crossref":
                df = search_crossref(q, rows=15)
            else:
                els = [s.strip() for s in q.split(",")]
                df = materials_project_summary(elements=els, page_size=15)
        st.write(f"**{len(df)} hits**")
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: LLM table extraction
# ---------------------------------------------------------------------------
with tab_llm:
    st.subheader("LLM 표 추출 (Claude)")
    st.caption(
        "논문 본문(PDF→텍스트)을 붙여 넣으면 Claude가 합금 조성 + "
        "물성을 추출합니다. API 키 없으면 정규식 fallback로 동작 "
        "(정확도 낮음 — confidence='low'로 표시)."
    )

    text = st.text_area("논문 텍스트", height=200,
                        placeholder="Abstract / methods / results 텍스트...")
    c1, c2 = st.columns(2)
    with c1:
        use_llm = st.checkbox("Claude 사용 (없으면 휴리스틱)", True)
    with c2:
        stage_name = st.text_input("Source 라벨", "llm_extracted")

    if st.button("추출 실행", key="llm_extract") and text.strip():
        with st.spinner("추출 중..."):
            df_out, report = extract_alloy_table(text=text, use_llm=use_llm)
        st.write("**Report:**")
        st.json(report.to_dict())
        if df_out.empty:
            st.warning("추출된 행이 없습니다.")
        else:
            st.dataframe(df_out, use_container_width=True)
            if st.button("이 결과를 단계화", key="llm_stage"):
                st.session_state.data_sources[stage_name] = df_out
                st.success(f"{stage_name} 단계화 ({len(df_out)} rows)")


# ---------------------------------------------------------------------------
# Tab 5: merge + download
# ---------------------------------------------------------------------------
with tab_merge:
    st.subheader("통합 + 다운로드")
    _staged_summary()

    c1, c2 = st.columns(2)
    with c1:
        el_cols_text = st.text_input(
            "Element columns (쉼표)",
            value=",".join(st.session_state.ingest_element_cols),
        )
    with c2:
        tg_cols_text = st.text_input(
            "Target columns (쉼표)",
            value=",".join(st.session_state.ingest_target_cols),
        )
    basis = st.selectbox("Composition basis", ["auto", "weight_pct", "atomic_pct"])
    dedup = st.checkbox("중복 제거", True)

    if st.button("Merge 실행", key="merge_btn"):
        el_cols = [s.strip() for s in el_cols_text.split(",") if s.strip()]
        tg_cols = [s.strip() for s in tg_cols_text.split(",") if s.strip()]
        if not st.session_state.data_sources:
            st.error("단계화된 소스가 없습니다.")
        else:
            merged, summary = merge_datasets(
                sources=st.session_state.data_sources,
                element_columns=el_cols,
                target_columns=tg_cols,
                composition_basis=basis,
                dedup=dedup,
            )
            st.write("**Summary**")
            st.json({
                "n_rows_in": summary.n_rows_in,
                "n_rows_out": summary.n_rows_out,
                "duplicated_dropped": summary.duplicated_dropped,
                "outliers_flagged": summary.outliers_flagged,
                "columns_normalised": summary.columns_normalised,
                "notes": summary.notes,
            })
            st.dataframe(merged, use_container_width=True)
            csv = merged.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 통합 CSV 다운로드", csv,
                "merged_dataset.csv", "text/csv", key="dl_csv",
            )

    if st.button("모든 단계 초기화", key="reset_btn"):
        st.session_state.data_sources = {}
        st.success("초기화됨")
