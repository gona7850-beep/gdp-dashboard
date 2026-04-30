# 실행 가능한 실무 프로토콜 (SOP)

## Phase 0. 목표/제약 정의 (Day 0)
- 목표: 예) 1000°C 항복강도, 크리프 수명, 산화저항
- 제약: 밀도, 비용, 독성, 가공성
- 산출물: `project_config.yaml` (사내 양식)

## Phase 1. 후보 조합 정의 (Day 1)
- Binary: Nb-X 전체 (X = Ti, Zr, Hf, V, Ta, W, Mo, Al, Cr, Si, B 등)
- Ternary: Nb-X-Y (X/Y는 강화/산화저항군으로 분리)
- Quaternary: Nb-X-Y-Z (공정성/비용 제한 반영)
- 규칙:
  - 단일 원소 상한: 20 at.% (초기), 총 합 100 at.%
  - 취성 유발 원소 조합 제한 규칙 사전 정의

## Phase 2. DFT DB 수집/정규화 (Day 2~4)
- 소스: OQMD, Materials Project, AFLOW
- 수집: 조성, 구조, E_above_hull, ΔH_f, 탄성계수, 밀도
- 정규화:
  - 에너지 단위 통일 (eV/atom)
  - 중복 제거 (조성+구조키)
  - 누락치 태그(`missing_*`)

## Phase 3. 1차 필터 (Day 4)
- `E_above_hull <= 0.10 eV/atom`
- `ΔH_f < 0` 우선
- 공정성 필터: 극단 융점/휘발/독성 제외

## Phase 4. Nb 정합성 계산 (Day 5~7)
- Misfit surrogate:
  - `misfit_pct = abs(a_p - a_nb) / a_nb * 100`
- Interface surrogate:
  - 입력: misfit, 원자반경차, 전기음성도차, 체적차
  - 출력: 상대 계면에너지 점수(낮을수록 우수)

## Phase 5. 다목적 점수화 (Day 7)
- 목적함수:
  - 안정성, 정합성, 강화잠재력, 취성위험, 비용, 공정성
- 가중치 초기값:
  - 안정성 0.25, 정합성 0.20, 강화잠재력 0.20, 취성위험 0.15, 비용 0.10, 공정성 0.10

## Phase 6. Kinetics + Phase-field (Day 8~14)
- Top 10 후보에 대해:
  - 확산/석출 surrogate 계산
  - phase-field 입력 파라미터 생성
- 출력: 석출 크기/분율/조대화 경향

## Phase 7. 실험 DOE (Week 3)
- Top 3~5 조성 선정
- 용체화/시효 매트릭스 적용
- 측정: XRD, SEM/TEM, 경도, 인장, 크리프

## Phase 8. 액티브러닝 재학습 (Week 4)
- 측정값으로 점수함수/가중치 업데이트
- 다음 배치 자동 추천 (`next_batch.csv`)

## Phase 9. 최종 조성/열처리맵 도출
- 성능-비용-공정성 Pareto front
- 추천안 A/B/C + 리스크 기록
