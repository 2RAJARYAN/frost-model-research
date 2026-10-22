# frost-model-research

Research and modeling half of a 2-person IoT frost/cold-injury alert
project for agriculture (deployment target: Warora/Vidarbha, Maharashtra).
A companion repo, `frost-edge-system` (Repo B), runs on the Raspberry Pi —
sensors, dashboard, alerts. This repo's job is to figure out, from data and
literature, what actually predicts frost/cold risk, and to hand Repo B a
trained model plus a feature list it can act on.

**Operating principle:** research → dataset → baseline → model → *then*
sensors get chosen. Feature-importance results here drive what hardware
Repo B buys, not the other way around.

## Regional framing

Warora/Vidarbha leans toward heatwave/drought/occasional cold-wave citrus
damage rather than classic sub-zero radiation frost. The pipeline is the
same either way — the label threshold is a config value, not a different
model — but pick one framing explicitly with your guide before treating
`label_rules.py`'s defaults as final:

- **General-purpose frost model** (Tmin < 0°C or similar), validated on
  foreign/threshold-derived data, deployed locally as proof-of-concept.
- **Cold-injury / minimum-temperature risk alert** for a specific local
  crop (e.g. citrus), using a documented chilling-injury threshold above
  0°C (commonly 2–5°C in horticultural literature).

## Label strategy

Labels are *computed* from raw temperature (and, for one variant, dew
point) — not learned, clustered, or taken from a dataset column. Two
validated variants (Omazić et al. 2024, *Agricultural and Forest
Meteorology*) are implemented in `src/label_rules.py` and reported
side by side, since they diverge for a real physical reason:

| Variant | Rule | Behavior |
|---|---|---|
| **Broad** | `Tmin < 3°C` | Also catches *black frost* (crop damage with no visible ice, air too dry to form it). Higher recall against real cold-injury risk. |
| **Strict** | `Tmin < 3°C AND Td < 0°C` (Td from Tmin + daily-mean RH) | The paper's best-validated rule against *observed* frost (POD 0.98, POFD 0.18). Tuned to visible (white) frost — under-catches black frost by construction. |

See `notebooks/02_label_definition.ipynb` for the full derivation and a
worked example of where the two variants disagree.

## Data strategy

Three tiers, one consistent label rule so results stay comparable:

1. **Tier 1** (done): [florian-huber/weather_prediction_dataset](https://github.com/florian-huber/weather_prediction_dataset) —
   daily data, 18 European cities, 2000–2010. No frost label; derived via
   the threshold rules above. Used to get the whole pipeline running
   end-to-end on real data.
2. **Tier 2** (time-boxed): a dataset with *real recorded* frost labels —
   Korea ASOS (Noh et al. 2021) or Murcia, Spain (Guillén-Navarro et al.
   2019), if accessible within a few days. Croatian DHMZ data is
   confirmed unavailable (the paper's own Data Availability statement) —
   not pursued.
3. **Tier 3** (weeks 9–10): NASA POWER / AgERA5 grid cell nearest
   Warora/Vidarbha, or a real station via NOAA's Integrated Surface
   Database / GHCNh if one exists nearby. Framed explicitly as a
   calibration/transfer exercise, not a validated local benchmark.

## Repo structure

```
frost-model-research/
├── data/{raw,interim,processed}/
├── notebooks/            01_eda → 02_label_definition → 03_baselines →
│                          04_feature_engineering → 05_model_training →
│                          06_regional_calibration
├── src/
│   ├── data_loader.py     # load raw data, per-city extraction, unit fixes
│   ├── label_rules.py     # both validated label variants + windowing
│   ├── features.py        # SINGLE SOURCE OF TRUTH — Repo B mirrors this exactly
│   ├── baselines.py       # persistence, physical/threshold, logistic regression
│   ├── models.py          # RF / XGBoost / LightGBM / CatBoost wrappers (Phase 3)
│   ├── evaluate.py        # shared metrics: precision/recall/F1, Peirce skill, RMSE/MAE
│   └── export_model.py    # serialize final model for Repo B (Phase 3)
├── experiments/runs.csv   # append-only run log: model, params, dataset, metrics
├── reports/{figures/, comparison_table.md}
├── tests/
└── .github/workflows/ci.yml
```

## Cross-repo contract

`src/features.py` is the single source of truth for feature engineering.
Repo B's `feature_pipeline.py` must reproduce it exactly — same lag
windows, same rolling stats, same units, same column names (documented at
the top of `features.py`). Functions there are kept small and pure
specifically so they can be ported/imported directly rather than
re-implemented from memory.

**Unit gotcha worth knowing before touching a new dataset:** in the Tier-1
CSV, `<CITY>_humidity` is a 0–1 fraction, not a percentage, and
`cloud_cover` is in oktas (0–8), not percent. `get_city_frame()` in
`data_loader.py` derives a `humidity_pct` (0–100) column specifically so
this doesn't get silently fed into a dew-point formula at the wrong scale.

## Status

- [x] Repo structure
- [x] Tier-1 data pulled, loading/per-city extraction working
      (`src/data_loader.py`)
- [x] EDA: distributions, missingness (none in Tier 1), class balance,
      multi-city column-availability scan (`01_eda.ipynb`)
- [x] Both label variants implemented and compared (`src/label_rules.py`,
      `02_label_definition.ipynb`)
- [x] `features.py` core functions: dew point, lag, rolling, rate of
      change, cyclical time encoding — unit-tested
- [x] Phase 1 baselines: persistence, physical/threshold, logistic
      regression, all logged to `experiments/runs.csv`
      (`src/baselines.py`, `03_baselines.ipynb`)
- [x] Phase 2: feature engineering across 4 cities (Basel, Oslo,
      Perpignan, De Bilt), train/val/test tables saved to `data/processed/`
      (`04_feature_engineering.ipynb`) — a plain linear-regression sanity
      check already beats persistence (MAE 1.99°C vs 2.11°C), a good sign
      heading into Phase 3
- [x] Phase 3: RF/XGBoost/LightGBM/CatBoost models — all beat Persistence
      on the same 4-city test set (`src/models.py`,
      `05_model_training.ipynb`); feature importance run, wind flagged as
      "nice-to-have, not must-have" for Repo B pending Tier 3 confirmation
- [x] Hyperparameter tuning on `val.csv` — done, honestly: did NOT
      improve on untuned defaults within this search budget (see
      `reports/comparison_table.md`)
- [x] `src/export_model.py`, feature-importance handoff to Repo B —
      `models/v1_model.cbm` + `models/v1_metadata.json` exported.
      **Not yet deployable as-is**: trained with one-hot city columns
      that don't exist at a single deployment site — flagged explicitly
      in the metadata, not silently left as a surprise for Repo B
- [ ] Regional calibration on Tier 3 data (`06_regional_calibration.ipynb`)
- [ ] Inference latency/memory benchmark on the actual Pi

## Setup

```bash
pip install -r requirements.txt
pytest tests/ -v          # run the test suite
jupyter notebook           # run notebooks in order, 01 → 06
```

## Out of scope this semester

No sensor network, no multi-node deployment, no solar-power modeling, no
computer-vision frost verification. These are future-work material for
the report, not this semester's build.

## Methodological framework (proof of concept)

**Problem.** Many agricultural regions have no locally observed,
frost-labelled data, and building a field network needs site access and
multi-season monitoring that is not always feasible. This project tests a
method for building a frost-risk forecasting pipeline under that
constraint: take a physically defined, literature-validated frost rule,
build and verify a reproducible pipeline on data that is available,
measure how well the rule transfers where real frost observations exist,
then apply it to the target region as a calibrated starting point that
field data can later refine.

**Label basis.** Labels are rule-derived (Omazić et al. 2024; broad and
strict variants, always reported side by side), not observations. Results
on derived labels are read as forecast skill for the rule's conditions.
Claims about frost itself are made only where observed labels exist.

**What each tier is for.**

| Tier | Role | What it can support |
|---|---|---|
| 1 | Pipeline development (European daily data) | The pipeline works mechanically. Not evidence of frost skill. |
| 2 | Rule transfer against real recorded frost (Korea, Noh et al. 2021, or Murcia as fallback) | The only empirical test of whether a rule validated in one climate holds in another. |
| 3 | Application to an Indian region (rule-derived) | A theoretical frost/cold-risk assessment, a reusable derived dataset and a usable pipeline. Labelled theoretical until field-validated. |

Noh et al. 2021 is a different task (hourly classification of observed
frost, March only, balanced training data). Its reported scores are
context. Compare directly only if its label, season and sampling are
reproduced.

**Tier 2 evaluation protocol** (report every stage, whatever the outcome):

1. **Unadjusted transfer:** apply the published rule unchanged; report
   POD, POFD and Peirce skill against observed frost.
2. **Calibrated transfer:** tune the temperature threshold and dew-point
   gate on training years/stations only, then test on held-out
   years/stations.
3. **Forecast evaluation:** compare the models with persistence and a
   climatology-plus-persisted-anomaly baseline, against observed frost.

Report per station group and leave-one-group-out, with block-bootstrap
confidence intervals. Do not rank models whose differences fall inside
the intervals.

**Success criteria (fix BEFORE running Tier 2):**
(a) the unadjusted rule beats the climatology baseline in Peirce skill by
at least `[X]`; (b) calibration recovers at least `[Y]%` of the gap
between the unadjusted rule and the best achievable threshold on held-out
data.

**Reading the outcome.**

- Rule works as-is: robust across climates.
- Works only after recalibration: thresholds are climate-specific. The
  size of the shift sets the sensitivity range used in Tier 3.
- Fails: identify which part broke (temperature level or dew-point gate).
- Models add little over baselines: prefer the simpler approach for
  deployment.

**Refinement loop.** When field observations exist, recalibrate the
threshold, dew-point gate and model weights against them, and version the
updated configuration with its training description and known
limitations. Thresholds are config values and exported metadata records
the data and features used, so this needs no restructuring.

**Scope and limitations.** Tier 2 supports the claim that the method
(rule plus calibration) transfers across the climates tested. It does not
establish performance in India, which stays a hypothesis for field
validation. Derived labels cannot capture black frost, snow-cover
observation effects or local cold-air pooling, and gridded data can
understate extreme local minima. Results are a proof of concept for the
method, not a validated frost model for the target region.
