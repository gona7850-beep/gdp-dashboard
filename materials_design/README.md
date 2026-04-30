# Nb-base 합금 설계 실무 프로토콜 (DFT HTS + 실험 + 액티브러닝)

이 디렉토리는 Nb-base binary/ternary/quaternary 조성 설계를 위한 실행 템플릿을 제공합니다.

## 구성
- `protocol.md`: 단계별 SOP (데이터 수집 → 점수화 → DOE → 재학습)
- `descriptors_schema.csv`: 필수 descriptor 컬럼 정의
- `scoring_template.csv`: 엑셀에서 바로 사용할 점수 계산 템플릿
- `pipeline.py`: Python 기반 자동 랭킹/Top-N 선정/다음 실험 추천
- `pseudocode.md`: Top-N 후보 자동 랭킹 의사코드
- `doe_matrix.csv`: 용체화/시효 DOE 매트릭스 샘플

## 빠른 시작
```bash
python materials_design/pipeline.py \
  --input materials_design/scoring_template.csv \
  --top_n 5 \
  --hull_th 0.10 --hf_th 0.0 --brittle_th 0.60 \
  --output_dir materials_design/out
```

## 가중치 튜닝 예시
```bash
python materials_design/pipeline.py \
  --input materials_design/scoring_template.csv \
  --w_stability 0.35 --w_coherency 0.25 --w_strength 0.20 --w_brittle 0.10 --w_cost 0.05 --w_process 0.05
```

## 주의
- 본 템플릿은 CALPHAD-free 1차 탐색/우선순위화를 위한 프레임워크입니다.
- 최종 채택 전에는 실험 검증과 최소한의 열역학 보정이 필요합니다.
