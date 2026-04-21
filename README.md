# ⚙️ Nb Alloy Composition–Property Platform

Streamlit 기반의 Nb 합금 데이터 분석 플랫폼입니다. 아래 기능을 하나의 UI에서 제공합니다.

- 데이터 업로드/미리보기 (CSV/XLSX)
- Wide → Long 포맷 변환
- 상관분석(MIC/PCC) + 중요 변수 선택
- 다중 회귀 모델 학습 및 비교
- 조성 랜덤서치 최적화
- 결과 CSV/모델 다운로드

## Run locally

1. 의존성 설치

```bash
pip install -r requirements.txt
```

2. 앱 실행

```bash
streamlit run streamlit_app.py
```

## Package

핵심 로직은 `nb_alloy_platform/` 패키지에 모듈화되어 있으며, UI 없이 Python 코드로도 재사용 가능합니다.


## Colab

Colab에서 바로 실행하려면 `colab_ready_nb_alloy_platform.py`를 사용하세요. 빠른 절차는 `COLAB_QUICKSTART.md`를 참고하세요.
