"""Page 6 — Literature ingestion (legal OA sources only).

Sci-Hub and similar piracy mirrors are NOT integrated — see core/literature.py
for the policy rationale.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.literature import (  # noqa: E402
    NB_SI_PRESET_WITH,
    NB_SI_PRESET_WITHOUT,
    build_query,
    compound_search,
    unpaywall_lookup,
)

st.set_page_config(page_title="6 · 문헌 수집", page_icon="📚", layout="wide")
st.title("📚 Step 6 — Literature Auto-Ingestion (Legal OA only)")
st.caption("Sci-Hub은 COPE/Elsevier 출판윤리 위반 사유로 제외 — Unpaywall로 합법 OA full-text 확보")

with st.sidebar:
    st.header("Keyword bundle")
    with_words = st.text_area("Include any of (one per line)", "\n".join(NB_SI_PRESET_WITH))
    without_words = st.text_area("Exclude (one per line)", "\n".join(NB_SI_PRESET_WITHOUT))
    sources = st.multiselect(
        "Sources", ["crossref", "openalex", "semantic_scholar", "arxiv"],
        default=["crossref", "openalex"],
    )
    rows = st.slider("Rows per source", 5, 100, 25)

query = build_query(
    [w.strip() for w in with_words.splitlines() if w.strip()],
    [w.strip() for w in without_words.splitlines() if w.strip()],
)
st.code(query, language="text")

if st.button("Search", type="primary"):
    with st.spinner("Querying APIs..."):
        papers = compound_search(query, sources=sources, rows=rows)

    if not papers:
        st.warning("No results.")
    else:
        df = pd.DataFrame([
            {
                "DOI": p.doi or "",
                "title": p.title,
                "authors": ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else ""),
                "year": p.year,
                "venue": p.venue,
                "citations": p.citations,
                "OA": p.oa_url or "",
                "source": p.source,
            }
            for p in papers
        ])
        st.success(f"{len(df)} unique papers (de-duplicated by DOI + title)")
        st.dataframe(df, use_container_width=True)
        st.session_state["lit_df"] = df

        st.download_button(
            "Download literature.csv",
            df.to_csv(index=False).encode("utf-8"),
            file_name="literature.csv",
            mime="text/csv",
        )

st.divider()
st.subheader("Unpaywall — resolve legal OA PDF for a single DOI")
doi = st.text_input("DOI", value="10.1016/j.intermet.2021.107172")
if st.button("Resolve"):
    url = unpaywall_lookup(doi)
    if url:
        st.success(f"Legal OA: {url}")
    else:
        st.info("No legal OA found (or UNPAYWALL_EMAIL not set in environment).")
