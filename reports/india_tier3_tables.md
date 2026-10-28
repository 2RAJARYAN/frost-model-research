## Comparison on the TEST seasons (2018-2024), val-tuned thresholds

| method | threshold | mae | rmse | mae_cold | pod | pofd | pss | pss_lo | pss_hi | precision | alerts | events | val_pss_at_thr | inside_best_CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence | 7.25 | 1.318 | 1.716 | 1.409 | 0.972 | 0.175 | 0.797 | 0.775 | 0.819 | 0.171 | 2467 | 433 | 0.82 | False |
| Climatology+persisted anomaly | 6.25 |  |  |  | 0.859 | 0.0922 | 0.767 | 0.691 | 0.827 | 0.257 | 1450 | 433 | 0.682 | False |
| Logistic regression (classifier) | 0.36 |  |  |  | 0.982 | 0.1515 | 0.83 | 0.808 | 0.851 | 0.194 | 2196 | 433 | 0.842 | True |
| RandomForest | 7.0 | 1.071 | 1.398 | 1.415 | 0.986 | 0.15 | 0.836 | 0.815 | 0.855 | 0.196 | 2181 | 433 | 0.824 | True |
| XGBoost | 6.5 | 1.067 | 1.396 | 1.385 | 0.958 | 0.1124 | 0.846 | 0.821 | 0.868 | 0.24 | 1729 | 433 | 0.823 | True |
| LightGBM | 7.25 | 1.07 | 1.396 | 1.394 | 0.988 | 0.17 | 0.818 | 0.797 | 0.84 | 0.177 | 2416 | 433 | 0.824 | False |
| CatBoost | 6.5 | 1.073 | 1.401 | 1.392 | 0.963 | 0.1142 | 0.849 | 0.825 | 0.872 | 0.238 | 1752 | 433 | 0.83 | True |

## Same, nominal thresholds (4 C for regressors, p = 0.5 for the classifier)

| method | threshold | mae | rmse | mae_cold | pod | pofd | pss | pss_lo | pss_hi | precision | alerts | events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Persistence | 4.0 | 1.318 | 1.716 | 1.409 | 0.591 | 0.0164 | 0.575 | 0.522 | 0.624 | 0.571 | 448 | 433 |
| Climatology+persisted anomaly | 4.0 |  |  |  | 0.524 | 0.0144 | 0.51 | 0.433 | 0.568 | 0.575 | 395 | 433 |
| Logistic regression (classifier) | 0.5 |  |  |  | 0.965 | 0.1161 | 0.849 | 0.826 | 0.87 | 0.235 | 1775 | 433 |
| RandomForest | 4.0 | 1.071 | 1.398 | 1.415 | 0.43 | 0.0077 | 0.422 | 0.344 | 0.497 | 0.674 | 276 | 433 |
| XGBoost | 4.0 | 1.067 | 1.396 | 1.385 | 0.471 | 0.0092 | 0.462 | 0.386 | 0.529 | 0.656 | 311 | 433 |
| LightGBM | 4.0 | 1.07 | 1.396 | 1.394 | 0.49 | 0.0093 | 0.48 | 0.407 | 0.545 | 0.66 | 321 | 433 |
| CatBoost | 4.0 | 1.073 | 1.401 | 1.392 | 0.48 | 0.0079 | 0.473 | 0.4 | 0.542 | 0.693 | 300 | 433 |

## Paired Peirce-skill difference vs strongest baseline (Persistence)

| method | PSS minus Persistence | ci_lo | ci_hi | significant |
|---|---|---|---|---|
| Climatology+persisted anomaly | -0.03 | -0.105 | 0.025 | False |
| Logistic regression (classifier) | 0.033 | 0.019 | 0.047 | True |
| RandomForest | 0.039 | 0.026 | 0.052 | True |
| XGBoost | 0.049 | 0.032 | 0.065 | True |
| LightGBM | 0.021 | 0.008 | 0.037 | True |
| CatBoost | 0.052 | 0.038 | 0.064 | True |

## Paired Peirce-skill difference vs CatBoost (the shipped model)

| method | PSS minus CatBoost | ci_lo | ci_hi | distinguishable |
|---|---|---|---|---|
| Logistic regression (classifier) | -0.019 | -0.031 | -0.004 | True |
| RandomForest | -0.013 | -0.026 | 0.003 | False |
| XGBoost | -0.003 | -0.016 | 0.007 | False |
| LightGBM | -0.031 | -0.045 | -0.011 | True |

## Per-station TEST results (CatBoost, threshold 6.5)

| station | test_rows | cold_nights | alerts | pod | pofd | pss |
|---|---|---|---|---|---|---|
| HISSAR | 915 | 118 | 267 | 0.949 | 0.1945 | 0.755 |
| AMRITSAR | 703 | 96 | 242 | 0.99 | 0.2422 | 0.747 |
| SAFDARJUNG | 909 | 43 | 148 | 0.977 | 0.1224 | 0.854 |
| GANGANAGAR | 925 | 32 | 222 | 0.969 | 0.2139 | 0.755 |
| GWALIOR | 851 | 31 | 124 | 0.968 | 0.1146 | 0.853 |
| PATIALA | 899 | 27 | 184 | 0.963 | 0.1812 | 0.782 |
| BAREILLY | 895 | 24 | 131 | 1.0 | 0.1228 | 0.877 |
| BIKANER | 506 | 19 | 60 | 0.842 | 0.0903 | 0.752 |
| GAYA | 491 | 11 | 41 | 1.0 | 0.0625 | 0.938 |
| GUNA | 937 | 7 | 64 | 0.857 | 0.0624 | 0.795 |
| JAIPUR | 736 | 7 | 48 | 1.0 | 0.0562 | 0.944 |
| LUCKNOW | 602 | 7 | 68 | 1.0 | 0.1025 | 0.897 |
| JAISALMER | 954 | 7 | 72 | 1.0 | 0.0686 | 0.931 |
| SATNA | 906 | 4 | 55 | 0.75 | 0.0576 | 0.692 |
| JODHPUR | 897 | 0 | 26 |  | 0.029 |  |

## Sensor ablation (LightGBM; rows with pressure and wind present; common threshold)

| method | threshold | mae | mae_cold | pod | pofd | pss | pss_lo | pss_hi | dPSS_vs_core | dPSS_ci | dMAE_vs_core | dMAE_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| core (temp+humidity) | 7.25 | 1.079 | 1.62 | 0.981 | 0.1661 | 0.815 | 0.789 | 0.84 |  |  |  |  |
| + pressure change | 7.25 | 1.07 | 1.615 | 0.978 | 0.1659 | 0.812 | 0.783 | 0.839 | -0.003 | [-0.011, 0.002] | -0.009 | [-0.013, -0.004] |
| + pressure change + wind | 7.25 | 1.069 | 1.605 | 0.978 | 0.1655 | 0.812 | 0.784 | 0.839 | -0.003 | [-0.009, 0.002] | -0.01 | [-0.015, -0.006] |

## Leave-one-station-out (LightGBM, core features)

| station | rows | cold_nights | pss model@4 | pss model@7.25 | pss_ci | pod | pofd | pss persist@4 | pss persist@7.25 |
|---|---|---|---|---|---|---|---|---|---|
| AMRITSAR | 2902 | 699 | 0.51 | 0.67 | [0.64, 0.71] | 0.97 | 0.297 | 0.7 | 0.59 |
| HISSAR | 3391 | 279 | 0.47 | 0.73 | [0.68, 0.76] | 0.95 | 0.228 | 0.55 | 0.73 |
| PATIALA | 3926 | 192 | 0.43 | 0.74 | [0.70, 0.77] | 0.99 | 0.252 | 0.42 | 0.73 |
| GWALIOR | 3730 | 129 | 0.34 | 0.79 | [0.74, 0.83] | 0.95 | 0.16 | 0.5 | 0.77 |
| SAFDARJUNG | 4400 | 121 | 0.38 | 0.77 | [0.74, 0.80] | 0.98 | 0.211 | 0.44 | 0.76 |
| BIKANER | 2520 | 109 | 0.18 | 0.81 | [0.76, 0.85] | 0.94 | 0.127 | 0.42 | 0.75 |
| LUCKNOW | 3316 | 84 | 0.33 | 0.86 | [0.83, 0.89] | 0.99 | 0.125 | 0.45 | 0.8 |
| GANGANAGAR | 1317 | 71 | 0.35 | 0.7 | [0.67, 0.74] | 0.96 | 0.258 | 0.49 | 0.71 |
| JAIPUR | 3224 | 67 | 0.15 | 0.84 | [0.75, 0.90] | 0.96 | 0.115 | 0.45 | 0.81 |
| GAYA | 2932 | 42 | 0.19 | 0.87 | [0.77, 0.92] | 0.95 | 0.082 | 0.42 | 0.86 |
| BAREILLY | 3313 | 32 | 0.4 | 0.89 | [0.87, 0.91] | 1.0 | 0.11 | 0.43 | 0.88 |
| JAISALMER | 3836 | 29 | 0.38 | 0.84 | [0.70, 0.92] | 0.93 | 0.089 | 0.48 | 0.81 |
| SATNA | 3261 | 29 | 0.28 | 0.89 | [0.78, 0.93] | 0.97 | 0.079 | 0.41 | 0.76 |
| GUNA | 3225 | 28 | 0.07 | 0.86 | [0.77, 0.94] | 0.93 | 0.074 | 0.21 | 0.81 |

## Negative controls

| model | threshold | station | rows | actual_cold_nights | alerts | alert_rate_% |
|---|---|---|---|---|---|---|
| CatBoost | 6.5 | DR AMBEDKAR INTERNATIONAL | 3414 | 1 | 0 | 0.0 |
| CatBoost | 6.5 | GORAKHPUR | 3324 | 17 | 205 | 6.17 |
| CatBoost | 6.5 | KOTA | 3367 | 10 | 59 | 1.75 |
| CatBoost | 6.5 | PATNA | 3749 | 14 | 123 | 3.28 |
| LightGBM | 7.25 | DR AMBEDKAR INTERNATIONAL | 3414 | 1 | 3 | 0.09 |
| LightGBM | 7.25 | GORAKHPUR | 3324 | 17 | 322 | 9.69 |
| LightGBM | 7.25 | KOTA | 3367 | 10 | 130 | 3.86 |
| LightGBM | 7.25 | PATNA | 3749 | 14 | 215 | 5.73 |
