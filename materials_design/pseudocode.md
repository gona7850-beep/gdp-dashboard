# Top-N 자동 랭킹 의사코드

1. 데이터 로드
2. 결측치 처리 및 단위 정규화
3. misfit 미존재 시 계산: abs(a_p-a_nb)/a_nb*100
4. 하드 필터 적용:
   - E_above_hull <= threshold
   - delta_hf <= threshold_hf
   - brittle_risk <= threshold_brittle
5. 각 목적 점수 계산:
   - stability = f(e_above_hull, delta_hf)
   - coherency = f(misfit_pct, interface_surrogate)
   - strength = f(strength_proxy)
   - process = f(processability)
   - cost = f(cost_index)
   - oxidation = f(oxidation_score)
6. 종합점수 계산:
   total = Σ(w_i * score_i) - penalty(brittle_risk)
7. total 내림차순 정렬 → Top-N 추출
8. 다양성 제약 적용 (동일 원소군 과집중 방지)
9. DOE용 후보 배치 생성
10. 실험 결과 병합 후 회귀모델 재학습
11. 획득함수(EI/UCB)로 다음 배치 추천
