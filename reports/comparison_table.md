# Results vs. baselines vs. literature

Source of truth for the numbers below is `experiments/runs.csv` (append-only
run log). This table is updated at the end of each phase — Phase 1 is
filled in; Phase 2/3 rows are placeholders until those notebooks exist.

## Phase 1 — Baselines (Basel, Tier 1, 1-day horizon, `frost_label_broad`)

| Model | Precision | Recall | F1 | Peirce skill | RMSE (°C) | MAE (°C) |
|---|---|---|---|---|---|---|
| Persistence | 0.842 | 0.842 | 0.842 | 0.771 | 2.43 | 1.87 |
| Physical/threshold (linear, physically-restricted features) | 0.858 | 0.846 | 0.852 | 0.783 | 2.35 | 1.81 |
| Logistic regression (class-weighted, direct classifier) | 0.812 | **0.947** | 0.874 | 0.848 | — | — |

*(An earlier, slightly different Persistence run — 0.837/0.837/0.837 — is
also in `runs.csv` from before the pipeline was refactored onto
`src/label_rules.py`/`src/data_loader.py`; the two differ marginally
because of the switch to the centralized loader, not a methodology
change.)*

**Reading this, honestly:** on this particular slice of data, the
class-weighted logistic-regression classifier actually got *higher*
recall than both continuous-then-threshold baselines, which is the
*opposite* of the pattern Omazić et al. (2024) found with their
class-weighted XGBoost (POD 0.81 vs. a plain threshold's POD 0.98). That's
worth reporting as-is rather than forced to match the literature —
possible reasons worth checking in Phase 3 (not concluded yet): Basel's
class imbalance under this threshold is much milder (~3.4:1, 29.6%
positive — see `01_eda.ipynb`) than a true rare-frost setting, and the
physical/threshold baseline here uses far fewer, hand-picked features than
Omazić et al.'s full XGBoost model. Re-check this comparison once Tier 2/3
data (rarer, more literature-comparable class balance) and the full
tree-ensemble models are in.

**Primary approach going forward:** regression-then-threshold (Persistence,
Physical/threshold), per the project's core design decision — logistic
regression stays the documented classifier ablation, not the headline
model, regardless of which one currently scores best on this one city and
threshold.

## Phase 2 — Feature-engineered models

*(No separate model comparison in Phase 2 — that notebook's job was
building the train/val/test tables Phase 3 uses below. A quick linear-
regression sanity check on the engineered features did beat plain
persistence, MAE 1.99°C vs 2.11°C — see `04_feature_engineering.ipynb`.)*

## Phase 3 — RF / XGBoost / LightGBM / CatBoost

**Basis:** 4 cities (Basel, Oslo, Perpignan, De Bilt), date-based
train/test split (test = 2008-07-01 onward), broad label
(`frost_label_broad`, `Tmin < 3°C`), no hyperparameter tuning yet
(`val.csv` reserved for that). Persistence is re-run on this exact test
set below the Phase 1 table, since the numbers above were computed on
Basel alone and are **not directly comparable** to what follows.

| Model | Precision | Recall | F1 | Peirce skill | RMSE (°C) | MAE (°C) |
|---|---|---|---|---|---|---|
| Persistence (4-city test set) | 0.836 | 0.832 | 0.834 | 0.769 | 2.76 | 2.11 |
| Random Forest | 0.878 | 0.835 | 0.856 | 0.790 | 2.18 | 1.72 |
| XGBoost | 0.887 | 0.846 | 0.866 | 0.805 | 2.13 | 1.68 |
| LightGBM | 0.885 | 0.846 | 0.865 | 0.804 | 2.14 | 1.68 |
| CatBoost | 0.882 | 0.851 | **0.866** | **0.807** | 2.16 | 1.71 |

### Hyperparameter tuning — did it help? No.

Tuned on `val.csv` (a modest, non-exhaustive grid; RF's search was
limited to an offline run due to a single-core dev environment), then
evaluated once on the held-out test set:

| Model | Untuned Peirce | Tuned Peirce |
|---|---|---|
| Random Forest | 0.790 | 0.785 |
| XGBoost | 0.805 | 0.803 |
| LightGBM | 0.804 | 0.798 |
| CatBoost | **0.807** | 0.804 |

Every tuned result is marginally *below* its untuned counterpart —
differences small enough to plausibly be noise. Honest conclusion:
default hyperparameters were already reasonable at this data scale; a
larger search on multi-core hardware is future work, not a claim this
search already exhausted the possibilities. **Model exported: untuned
CatBoost** (`models/v1_model.cbm` + `models/v1_metadata.json`).

**Important, concrete limitation of this export, not a generic one:**
its feature list includes one-hot columns for the 4 European training
cities. A single Warora deployment will never populate any of them — this
exact artifact cannot run there as-is. Retraining on Tier 3 data without
cross-city one-hot features is required before this is deployable, not
just before it's validated on Indian data.

All four beat Persistence on both MAE and Peirce skill, on the same
4-city test data. Differences *between* the four models are modest at
this untuned stage — XGBoost/LightGBM edge ahead on MAE, CatBoost
edges ahead on Peirce/recall, but none of this should be read as
"X is the best model" until a real hyperparameter search has run on
`val.csv`.

### Feature importance (Repo B's sensor-decision deliverable)

Separate run, full feature set (including `wind_speed`/`wind_gust`/
`cloud_cover`/`sunshine`, all with per-city gaps), LightGBM
(NaN-tolerant), **gain-based** importance (not the default split-count
metric, which overweights continuous variables like wind purely for
having more distinct values to split on — a real trap worth naming since
it's exactly the question being asked):

| Feature | Share of total gain |
|---|---|
| `temp_mean` | 89.9% |
| `temp_min` | 1.85% |
| `temp_min_roll7_mean` | 1.32% |
| `dew_point_c` | 0.86% |
| `temp_max` | 0.56% |
| `wind_speed` | 0.49% (rank 6 of 30) |
| `cloud_cover` | 0.49% |
| ... | |
| `wind_gust` | 0.18% (rank 17 of 30) |

**`temp_mean` dominating this heavily echoes (in a more extreme form)
Omazić et al. (2024)'s finding that Tmin alone carried about half their
model's total importance** — a temperature-family variable dominating is
consistent with the literature, not a red flag specific to this pipeline.

**Wind verdict for Repo B, stated plainly and provisionally:** a
dedicated anemometer looks like "nice to have, not must-have" on this
data — a real but small contribution (~0.5% of total gain), dominated by
temperature history. **This is a first-pass answer on mid-latitude
European data, not a final one** — re-check once Tier 3 (Warora-region)
data exists before committing to a hardware decision, since local climate
dynamics may weight wind differently there.

## Tier 3 (India) — pooled stations, rule-derived label (next-day Tmin <= 4 C), test winters 2018-2024

Not comparable to the Europe tables above (different data, ~3% prevalence, season split, thresholds tuned on
validation, 95% block-bootstrap CIs). Full results, limitations and the sensor ablation are in
`reports/india_tier3_results.md`.

| Method | Alert threshold | Peirce skill [95% CI] | POD | POFD | Alerts |
|---|---|---|---|---|---|
| Persistence (tuned threshold) | 7.25 C | 0.797 [0.775, 0.819] | 0.972 | 0.175 | 2,467 |
| CatBoost | 6.5 C | 0.849 [0.825, 0.872] | 0.963 | 0.114 | 1,752 |
| XGBoost | 6.5 C | 0.846 | 0.958 | | |
| Logistic regression (class-weighted) | p >= 0.36 | 0.830 | | | |

CatBoost, XGBoost and Random Forest are statistically tied; the models beat tuned persistence by +0.02 to +0.05
(paired intervals exclude zero) via fewer false alarms, not better cold-night error (MAE on cold nights ~1.4 C for
all). The Basel finding (classifier recall above threshold-rule recall) did **not** reverse at rare prevalence:
at nominal thresholds the classifier still has far higher recall; after validation tuning the gap disappears.

## Literature reference points (not directly comparable — different
data, regions, horizons, and label definitions; positioning only, not a
reproduction claim)

| Source | Result |
|---|---|
| Chile (RF) | ~90% accuracy |
| Massachusetts (RF) | 91–95% accuracy; Peirce skill 0.88 vs. Franklin-model baseline 0.68 |
| Korea (RF/SVM) | (see Noh et al. 2021 for exact figures once Tier 2 is attempted) |
| FRUTILLA | MAE 1.4–3.3°C |
| Omazić et al. (2024), Croatia — Method 7 (`Tmin3_Td0`) threshold rule | POD 0.98, POFD 0.18 (all-year) |
| Omazić et al. (2024), Croatia — XGBoost, class-weighted classifier | POD 0.81, POFD 0.06 |
