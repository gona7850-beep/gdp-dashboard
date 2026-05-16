"""Composition Design Platform — Streamlit landing page.

This entry shows the two ML workflows that currently work end-to-end:

* **Page 7 — 조성설계_플랫폼** (lite path, no heavy deps)
  Random Forest / GBR / Ridge / MLP forward model, Dirichlet Monte Carlo
  + GA inverse design, simple feasibility analysis, Claude-assisted
  target parsing.

* **Page 8 — AlloyForge_고급플랫폼** (advanced path)
  Stacked XGBoost + GP forward model with Optuna HPO, conformal 90%
  prediction intervals, domain-of-applicability scoring, NSGA-II
  multi-objective inverse design with risk-aware ``μ - λσ`` objective,
  SHAP explainability, active-learning batch picks.

The legacy Nb-Si pages (1–6) reference ``core/db.py`` and companion
modules that are missing / corrupted on disk; they will fail to load
until those files are restored.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Composition Design Platform",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Composition Design Platform")
st.caption(
    "ML-driven composition / property prediction, inverse design, validation "
    "and AI-assisted explanation. Two backends share the same synthetic-data "
    "generator and prompt catalogue, so you can compare lite vs advanced."
)

st.markdown(
    """
### 사용 가능한 워크플로우

| # | 페이지 | 모델 | 의존성 | 특징 |
|---|---|---|---|---|
| 7 | **조성설계_플랫폼** (lite) | RF / GBR / Ridge / MLP | 가벼움 (sklearn만) | Dirichlet MC + GA 역설계, 빠른 데모, Claude 어시스턴트 |
| 8 | **AlloyForge_고급플랫폼** (advanced) | Stacked XGB + GP | xgboost, optuna, pymoo, shap | Conformal intervals, DoA, NSGA-II, SHAP, active learning |

**왼쪽 사이드바**에서 페이지를 선택하세요. 데이터가 없으면 각 페이지의
"합성 데이터 생성" 버튼으로 즉시 시작할 수 있습니다.
"""
)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("API 엔드포인트", "/api/v1/composition + /api/v1/alloyforge")
with col2:
    st.metric("Streamlit 페이지", "7 (lite) · 8 (advanced)")
with col3:
    st.metric("플랫폼 버전", "0.2.0")

st.divider()

st.markdown(
    """
### 또 다른 접근 방법

* **REST API** — `uvicorn backend.main:app --reload` → http://localhost:8000
  - 단일 페이지 웹 UI: http://localhost:8000/
  - OpenAPI 문서: http://localhost:8000/docs
* **CLI 데모** — `python examples/alloyforge_demo.py`
* **Claude Code** — `.claude/agents/` 6개 서브에이전트, `.claude/commands/` 5개
  슬래시 명령어가 이 repo 안에서 자동 인식됩니다 (`/run-prediction`,
  `/inverse-search`, `/explain-prediction`, `/validate-design`,
  `/generate-doe`).
* **Docker** — `docker compose up` 하면 API(8000)와 Streamlit(8501)이
  동시에 뜹니다.

### Quick start

```bash
pip install -r requirements.txt

# 1) Streamlit (이 화면)
streamlit run app/streamlit_app.py

# 2) FastAPI + 단일페이지 웹 UI
uvicorn backend.main:app --reload
# → 브라우저에서 http://localhost:8000

# 3) CLI 데모 (Fe-Ni-Cr-Mo-Ti 합성 데이터)
python examples/alloyforge_demo.py

# 4) Docker (양쪽 서비스)
docker compose up
```

### Claude API 키 (선택)

`ANTHROPIC_API_KEY` 환경 변수를 설정하면 자연어 → 타겟 파싱과 후보
해설이 실제 Claude 응답으로 동작합니다. 키가 없어도 결정론적인 휴리스틱
fallback으로 모든 엔드포인트가 그대로 동작합니다.
"""
)

st.caption(
    "Legacy Nb-Si 페이지(1~6)는 별도 모듈(`core/db.py`, `core/features.py` 등)이 "
    "복구되어야 동작합니다. 현재 페이지 7·8과 REST API는 독립적으로 작동합니다."
)
