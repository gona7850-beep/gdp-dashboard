# Colab Quickstart

Google Colab에서 가장 빠르게 확인하는 방법:

## 1) 패키지 설치 셀 실행
```python
!pip -q install numpy pandas scipy scikit-learn
```

## 2) `colab_ready_nb_alloy_platform.py` 파일 전체를 Colab 셀에 복사/붙여넣기 후 실행

## 3) 데모 실행
```python
out = run_demo()
```

## 4) CSV로 실행
```python
# 예: /content/my_alloy_data.csv 업로드 후
out = run_pipeline_from_csv('/content/my_alloy_data.csv', target_col='HV')
out.results_df
out.optimization_df.head()
```
