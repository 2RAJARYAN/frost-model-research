# India Tier 3 — dataset and model results

**Status: proof of concept on rule-derived labels. Not validated against observed frost or crop damage, and not validated for Warora/Vidarbha or any single site.**

## What was built

| Artifact | Path |
|---|---|
| Dataset: 62,665 station-days, 19 stations, features + rule-based labels + season split | `data/processed/india_tier3.csv.gz` |
| Data card (generated from the data, with limitations) | `reports/india_dataset_card.md` |
| Model: CatBoost, regress next-day Tmin, alert if predicted Tmin <= 6.5 C | `models/india_v1_model.cbm` + `models/india_v1_metadata.json` |
| Lighter alternative: logistic regression as plain JSON (numpy only) | `models/india_v1_logreg.json` |
| All result tables | `reports/india_tier3_tables.md`, `experiments/india_tier3_*.csv`, `experiments/runs.csv` |

## Headline

On held-out winters (2018-2024) at 15 stations, tree models forecast the rule's next-day cold nights (Tmin <= 4 C) with Peirce skill about 0.85, against 0.80 for a persistence rule whose alert threshold is tuned the same way. The gain is real (+0.02 to +0.05, paired intervals exclude zero) but modest. It comes from raising fewer false alarms at about the same detection rate, **not** from predicting cold-night temperatures better: error on cold nights is about 1.4 C for every method, persistence included. At the operating point that maximises Peirce skill, roughly three of four alerts are false.

## Setup

- **Data:** NOAA GSOD daily records, Nov-Mar, 1990-2024. Modelling stations (15): Amritsar, Hisar, Patiala, Ganganagar, Gwalior, Bikaner, Safdarjung (Delhi), Lucknow, Jaipur, Bareilly, Jaisalmer, Gaya, Guna, Satna, Jodhpur. Negative controls (4, not used for training): Nagpur, Patna, Kota, Gorakhpur.
- **Label:** next-day Tmin <= 4 C (inclusive), on days with an explicit minimum report only. 4 C is IMD's plains Cold Wave criterion and Attri et al.'s frost-day definition. 3 C, 5 C and a strict variant (plus dew point < 0 C) are in the dataset too. Prevalence is about 3% of station-days, against about 30% for the Europe dry run.
- **Split by winter:** train <= 2013 (29,950 rows, 1,244 cold nights), val 2014-2017 (6,735 / 240), test 2018-2024 (12,126 / 433). Thresholds are tuned on val only.
- **Features (29, deployable with a temperature + humidity sensor):** same-day Tmin/Tmax/Tmean, mean RH, dew point and depression, diurnal range, lags 1-3 and rolling 3/7-day stats of Tmin and RH, day-of-year. No station-identity features.
- **Protocol:** Peirce skill = POD - POFD, with 95% block-bootstrap intervals over (station, season) blocks. Models whose paired difference interval includes zero are not ranked.

## Results (test seasons, validation-tuned alert thresholds)

| method | alert thr | mae | POD | POFD | precision | PSS [95% CI] | alerts |
|---|---|---|---|---|---|---|---|
| Persistence | 7.25 | 1.318 | 0.972 | 0.175 | 0.171 | 0.797 [0.775, 0.819] | 2467 |
| Climatology+persisted anomaly | 6.25 |  | 0.859 | 0.0922 | 0.257 | 0.767 [0.691, 0.827] | 1450 |
| Logistic regression (classifier) | 0.36 |  | 0.982 | 0.1515 | 0.194 | 0.830 [0.808, 0.851] | 2196 |
| RandomForest | 7.0 | 1.071 | 0.986 | 0.15 | 0.196 | 0.836 [0.815, 0.855] | 2181 |
| XGBoost | 6.5 | 1.067 | 0.958 | 0.1124 | 0.24 | 0.846 [0.821, 0.868] | 1729 |
| LightGBM | 7.25 | 1.07 | 0.988 | 0.17 | 0.177 | 0.818 [0.797, 0.840] | 2416 |
| CatBoost | 6.5 | 1.073 | 0.963 | 0.1142 | 0.238 | 0.849 [0.825, 0.872] | 1752 |

Regressors alert when predicted Tmin <= the threshold; the classifier alerts when probability >= its threshold (0.36).

**What the models add.** Against persistence, paired Peirce differences are CatBoost +0.052 [0.038, 0.064], XGBoost +0.049 [0.032, 0.065], Random Forest +0.039 [0.026, 0.052], logistic regression +0.033 [0.019, 0.047], LightGBM +0.021 [0.008, 0.037]. CatBoost raises 1,752 alerts to persistence's 2,467 while detecting roughly 4 fewer of the 433 cold nights (about 715 fewer alert-days, -29%). Mean absolute error falls from 1.318 C to 1.073 C overall, but on cold nights it is 1.409 C (persistence) vs 1.392 C (CatBoost): no real change. Climatology plus persisted anomaly is not better than persistence (-0.030, interval includes zero).

**Which model.** Paired against CatBoost: XGBoost -0.003 [-0.016, 0.007] and Random Forest -0.013 [-0.026, 0.003] are not distinguishable; logistic regression -0.019 [-0.031, -0.004] and LightGBM -0.031 [-0.045, -0.011] are slightly but distinguishably lower. CatBoost is shipped because it has the highest point estimate and the repo already has its export path, **not** because it is shown to beat XGBoost or Random Forest. Logistic regression is exported as dependency-free JSON for a Pi that cannot run CatBoost, at a cost of about 0.02 Peirce skill.

**Classifier vs threshold (the Section 3 prediction).** The prediction was that at rare-event prevalence the pattern would move back toward Omazic et al. (threshold rule higher recall, classifier higher precision), reversing the Basel result. **It is not supported.** At nominal thresholds the class-weighted classifier has far higher recall than a 4 C threshold (POD 0.965 vs 0.591 for persistence, 0.43-0.49 for the regressors) and far lower precision (0.235 vs 0.571): the same direction as Basel. Once thresholds are tuned on validation the gap disappears (classifier 0.830, regressors 0.82-0.85). The nominal gap is calibration: regression pulls cold-tail predictions upward, so the Peirce-optimal alert threshold is 6.5-7.25 C, well above the 4 C label. The tasks also differ: Omazic et al. detect same-day frost, this is a next-day forecast.

**Unseen stations (leave-one-station-out, LightGBM).** Held-out Peirce skill at the pooled alert threshold is 0.67-0.89, and beats tuned persistence at 13 of 14 stations by 0.035 on average (range -0.01 to +0.13). At the nominal 4 C threshold it is only 0.07-0.51. A single pooled threshold fits poorly where base rates differ: at Amritsar (about 24% of nights cold) untuned persistence at 4 C (0.70) beats the model at 7.25 C (0.67). The threshold is tuned on pooled validation data that includes each held-out station's validation winters, so these numbers are slightly optimistic.

**Sensors for Repo B.** On rows where pressure and wind exist (9,685 test rows, 313 cold nights, 15 stations), scored at one common threshold: adding sea-level pressure change changes Peirce skill by -0.003 [-0.011, 0.002] and error by -0.009 C [-0.013, -0.004]; adding wind as well changes them by -0.003 [-0.009, 0.002] and -0.010 C [-0.015, -0.006]. The error differences are statistically detectable but about 1% of the 1.08 C error, with no Peirce gain. **A temperature + humidity station is enough; neither a pressure sensor nor an anemometer is justified by this data.** Caveat: sea-level pressure is missing for roughly half the days at Amritsar, Lucknow, Jaipur and Nagpur, so the ablation under-represents them.

**Negative controls.** Nagpur (Vidarbha) raises 0 alerts in 3,414 days (1 cold night in 35 winters). Patna 3.3%, Kota 1.75%, Gorakhpur 6.2% of days, against actual cold-night rates of 0.1-0.5%. This is the cost of the 6.5 C alert threshold, not a failure to learn.

**Per station (CatBoost, test).** Peirce skill ranges 0.69-0.94 where there are at least 4 cold nights; Hisar 0.755 and Amritsar 0.747 carry most of the events (118 and 96 of 433), so pooled numbers lean on them. Full table in `reports/india_tier3_tables.md`.

## Limitations (read before using or citing)

1. **Rule-derived labels.** Skill is skill at forecasting the rule's conditions. No observed frost, no crop damage, and the Kinnow-specific threshold is unsourced.
2. **Station series have breaks.** Hisar's cold-night rate jumps from 2.8 to 19.2 per season between the 2000s and 2010s while NASA POWER and ERA5 do not follow; Amritsar's falls from 47 to 17 while the grids rise slightly. Hisar and Amritsar are in training and test.
3. **Gridded sources run warm in the cold tail** by about +0.5 to +3.4 C against stations, so gridded labels at the same threshold undercount. No Abohar-specific model exists yet: there is no station at Abohar (nearest is Ganganagar, 37 km, with 56% derived minima).
4. **Most 1990s data is excluded** (derived minima removed as targets), so the effective record is roughly 2000-2024.
5. **Training RH is derived** from GSOD mean temperature and mean dew point. Repo B measures RH directly, so there is a small distribution shift to test on the Pi.
6. **The shipped model is refit on all seasons** and has no held-out evaluation of its own. Reported metrics are from the train-only model.
7. **Operating point.** The alert threshold is tuned to maximise Peirce skill, which rewards detection. Precision is about 0.24. A deployment that cannot tolerate that false-alarm rate should choose a different threshold from `reports/india_tier3_tables.md` and the validation curve.
8. **Not benchmarked on a Raspberry Pi.** Not integrated with Repo B (`feature_pipeline.py` still missing).

## Not done

- Abohar-specific gridded dataset and calibrated threshold (needs a NASA POWER pull and your internet; calibration evidence in `data/raw/india/grid_threshold_calibration.csv`).
- Kinnow cold-injury threshold and phenology (ICAR-CCRI / PAU); IMD station data and station histories.
- Tier 2 (observed-frost transfer test), and `notebooks/06_regional_calibration.ipynb` (replaced by the scripts below).

## Reproduce

```bash
python scripts/gsod_download.py summarize            # needs the downloaded GSOD files
python scripts/build_india_dataset.py                # -> data/processed/india_tier3.csv.gz + data card
python scripts/train_india_models.py --log --export  # ~90 s on one core
python -m pytest tests/ scripts/
```
