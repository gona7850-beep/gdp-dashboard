"""Nb-Si AM Alloy Design Platform — main entry (Streamlit multi-page)."""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Nb-Si AM Platform",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Nb-Si Additive Manufacturing Alloy Design Platform")
st.caption("Physics-informed ML workflow for Nb-Si-Ti / C103 PBF·DED research")

st.markdown(
    """
### 8단계 워크플로우 (Phase 1: Steps 1–5 + 4.5 강화)

| # | 단계 | 페이지 |
|---|---|---|
| 1 | Hierarchical Database (Composition / Process / Microstructure / Properties) | `1_데이터베이스` |
| 2 | Physics-Informed Feature Engineering (VEC, Δχ, δ, sd/sp, Larson-Miller) | `2_피처엔지니어링` |
| 3 | Multi-algorithm Benchmarking (RF, XGB, LGBM, CatBoost, GPR, PLS1, …) | `3_벤치마킹` |
| 4 | SHAP-based XAI (global / local / interaction / physics validation) | `4_SHAP_XAI` |
| 5 | Multi-Objective Bayesian Optimization (BoTorch qNEHVI Pareto) | `5_MOBO_파레토` |
| 6 | Literature Auto-Ingestion (CrossRef / OpenAlex / S2 / arXiv / Unpaywall) | `6_문헌_수집` |

**왼쪽 사이드바에서 페이지를 선택**하여 진행하십시오.
"""
)

st.divider()

col1, col2, col3 = st.columns(3)
db_path = Path(__file__).parent.parent / "data" / "alloy.db"
with col1:
    st.metric("DB 상태", "✅ 존재" if db_path.exists() else "⚠ 미초기화")
with col2:
    st.metric("브랜치", "claude/am-ai-research-support-KjY4n")
with col3:
    st.metric("플랫폼 버전", "0.1.0 (Phase 1)")

st.divider()

st.markdown(
    """
### 설계 원칙

1. **물리 기반 ML**: VEC, Δχ, δ, ΔH_mix, Larson-Miller 등 합금 이론 파라미터를 1차 피처로 사용
2. **재현성**: 5-fold × 10 seeds CV + group CV (alloy class) + permutation/Y-randomization 표준 검증
3. **불확실성 정량화**: Bayesian/GPR 모델로 95% PI coverage 보고
4. **설명 가능성**: SHAP global/local/interaction → 물리 메커니즘과 자동 교차 검증
5. **합법 문헌 통로만 사용**: Sci-Hub 등 무허가 통로 명시 제외 (COPE/Elsevier ethics 준수)

### 빠른 시작
```bash
# 의존성 설치
pip install -r requirements.txt

# 1) Streamlit 프론트
streamlit run app/streamlit_app.py

# 2) FastAPI 백엔드 (별도 터미널)
uvicorn backend.main:app --reload

# 3) Gradio 단일 모델 데모
python gradio_apps/mobo_demo.py
```
"""
)
