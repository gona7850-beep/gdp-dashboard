# Real-data Nb-Si benchmark report

## 1. Hardness-only dataset (184 rows)

- Source: `data/nb_si/nb_silicide_hardness.csv`
- 19 elements (incl. Nb), 1 target (Vickers HV)
- 12 inferred alloy families (used as group key for CV)

```
Targets: ['Vickers_hardness_(Hv)']
  Vickers_hardness_(Hv)  CV R²=+0.235±0.372  MAE=171  perm p=0.667  coverage=85%@90%
Overall grade: D
```

### CV metrics

```
{
  "Vickers_hardness_(Hv)": {
    "cv_r2_mean": 0.23455410228316156,
    "cv_r2_std": 0.37202846845761395,
    "cv_mae_mean": 170.73752338301193,
    "permutation_p": 0.6666666666666666,
    "empirical_coverage": 0.8478260869565217
  }
}
```


## 2. HV + compressive σ_max dataset (51 rows)

```
Targets: ['Vickers_hardness_(Hv)', 'Compressive_strength_σ_max_(Mpa)']
  Vickers_hardness_(Hv)  CV R²=-0.792±2.037  MAE=195  perm p=0.667  coverage=86%@90%
  Compressive_strength_σ_max_(Mpa)  CV R²=-0.436±0.923  MAE=327  perm p=0.167  coverage=90%@90%
Overall grade: D
```

### CV metrics

```
{
  "Vickers_hardness_(Hv)": {
    "cv_r2_mean": -0.7917160291687823,
    "cv_r2_std": 2.0366985155213846,
    "cv_mae_mean": 195.35288629299626,
    "permutation_p": 0.6666666666666666,
    "empirical_coverage": 0.8627450980392157
  },
  "Compressive_strength_\u03c3_max_(Mpa)": {
    "cv_r2_mean": -0.43557900545310835,
    "cv_r2_std": 0.9228444918026362,
    "cv_mae_mean": 326.8335846727642,
    "permutation_p": 0.16666666666666666,
    "empirical_coverage": 0.9019607843137255
  }
}
```


## 3. Temperature-dependent compressive strength (94 rows)

- Process variable: test temperature 25–1400 °C

```
Targets: ['Compressive_strength_σ_max_(Mpa)']
  Compressive_strength_σ_max_(Mpa)  CV R²=+0.672±0.212  MAE=321  perm p=0.167  coverage=93%@90%
Overall grade: C
```

### CV metrics

```
{
  "Compressive_strength_\u03c3_max_(Mpa)": {
    "cv_r2_mean": 0.6716714370548766,
    "cv_r2_std": 0.21240956948923312,
    "cv_mae_mean": 320.7893404820445,
    "permutation_p": 0.16666666666666666,
    "empirical_coverage": 0.925531914893617
  }
}
```


## 4. Inverse design — max HV (Nb-rich, Si 10-25%)

Top 3 by aggregated risk-adjusted score:

```
   Nb    Si    Ti    Cr    Al    Hf    Mo     W    Zr  Vickers_hardness_(Hv)_mean  Vickers_hardness_(Hv)_std  agg_score
0.449 0.205 0.096 0.028 0.028 0.011 0.054 0.077 0.052                     610.156                      37.62   -591.346
```

Nearest documented alloy in the 38-entry reference DB (Euclidean in atomic-fraction space):

- Candidate #1: nearest = **Nb-22Si (composite)** (family: nb_silicide, distance: 0.323)

## Methodology notes

- Composition basis: weight % → atomic fraction via molar-mass normalisation. Rows where Σ wt% > 100 (overflow) are kept verbatim — the conversion preserves ratios.
- Group-aware 5-fold CV by *inferred alloy family* (the set of alloying elements present at ≥1 wt%) to prevent near-duplicate train/test leakage. This is harsher than vanilla KFold and gives the most honest R².
- 90 % conformal coverage is reported; gap from nominal shows where σ is mis-calibrated.
- Permutation p-value uses 10-20 random label shuffles; values < 0.05 mean the real model significantly beats random labels.
- Forward model = v1 (Optuna-tuned XGBoost + Gaussian-process residual head); featurizer is the base 33-feature physics-informed aggregate.

Re-run with: `python examples/benchmark_real_nb_si.py`