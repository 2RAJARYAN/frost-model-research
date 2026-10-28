# India Tier 3 dataset card (`data/processed/india_tier3.csv.gz`)

**What it is.** One row per station-day (Nov-Mar, 1990-2024) from NOAA GSOD records of
19 Indian stations. Features describe day *D*; the target is the **next day's
minimum temperature**, and the labels apply the project's rule to it.

**Labels are rule-derived, not observed frost or crop damage.** Results on them measure skill at
forecasting the rule's conditions. Nothing here validates the rule against observed damage.

| Column | Meaning |
|---|---|
| `target_tmin_c` | next day's Tmin (C), explicit-minimum days only |
| `label_le3`, `label_le4`, `label_le5` | `target_tmin_c <= 3 / 4 / 5` (inclusive). **Primary: `label_le4`** (IMD plains Cold Wave criterion; Attri et al. 2020 frost day) |
| `label_strict_le4` | `label_le4` AND Td < 0 C, with Td from Tmin and daily-mean RH (Omazic et al. 2024 form). RHmean is itself derived from GSOD mean temperature and mean dew point, so this is an approximation |
| `split` | by winter season: train <= 2013, val 2014-2017, test >= 2018 |
| `STATION`, `name`, `role` | grouping and bookkeeping only. **Never model features** |
| `today_min_derived` | today's Tmin was derived from synoptic readings (warm-biased); metadata only |

**Features** (32 in the widest set; names follow `src/features.py`):
- core, needs a temperature + humidity sensor (29): `tmin_c, tmax_c, tmean_c, rh_mean_pct, dew_point_c, dew_point_depression_c, diurnal_range_c, tmean_dewp_depression_c`, lags 1-3 and rolling 3/7-day min/mean/max of `tmin_c` and `rh_mean_pct`, `tmin_c_roc1`, `doy_sin`, `doy_cos`
- `+ pressure` (2): `slp_change_1d, slp_change_3d`. Pressure **changes** of sea-level pressure only, never absolute pressure. Optional (may be NaN): sea-level pressure is missing for roughly half or more of the days at Amritsar, Lucknow, Jaipur and Nagpur (see `slp_avail_pct` below). Station pressure (STP) is unusable: in this GSOD extract it is the constant 999.9 missing code
- `+ wind` (1): `wdsp_ms`

## Size and prevalence

| split | rows | stations | seasons | cold_nights_le4 | strict_le4 | prevalence_% |
|---|---|---|---|---|---|---|
| train | 39465 | 19 | 24 | 1269 | 999 | 3.22 |
| val | 8402 | 19 | 4 | 246 | 180 | 2.93 |
| test | 14798 | 19 | 7 | 444 | 288 | 3.0 |

| role | stations | rows | cold_nights_le4 | prevalence_% |
|---|---|---|---|---|
| event_station | 9 | 28726 | 1751 | 6.1 |
| low_event | 6 | 20085 | 166 | 0.83 |
| negative_control | 4 | 13854 | 42 | 0.3 |

Prevalence is the share of station-days whose next-day Tmin is <= 4 C. It is in the rare-event
regime the Europe dry run never reached (Basel was about 30%).

## Per station

| name | role | rows | seasons | le3 | le4 | le5 | strict_le4 | slp_avail_pct | prevalence_le4_% |
|---|---|---|---|---|---|---|---|---|---|
| AMRITSAR | event_station | 2902 | 35 | 494 | 699 | 968 | 538 | 9 | 24.09 |
| HISSAR | event_station | 3391 | 34 | 155 | 279 | 465 | 202 | 100 | 8.23 |
| PATIALA | event_station | 3926 | 35 | 70 | 192 | 421 | 101 | 100 | 4.89 |
| GWALIOR | event_station | 3730 | 35 | 45 | 129 | 283 | 115 | 98 | 3.46 |
| SAFDARJUNG | event_station | 4400 | 35 | 40 | 121 | 304 | 88 | 99 | 2.75 |
| BIKANER | event_station | 2520 | 30 | 57 | 109 | 194 | 103 | 100 | 4.33 |
| LUCKNOW | event_station | 3316 | 35 | 36 | 84 | 189 | 53 | 17 | 2.53 |
| GANGANAGAR | event_station | 1317 | 11 | 27 | 71 | 143 | 49 | 100 | 5.39 |
| JAIPUR | event_station | 3224 | 35 | 26 | 67 | 139 | 62 | 31 | 2.08 |
| GAYA | low_event | 2932 | 33 | 21 | 42 | 105 | 32 | 92 | 1.43 |
| BAREILLY | low_event | 3313 | 33 | 6 | 32 | 94 | 8 | 99 | 0.97 |
| SATNA | low_event | 3261 | 34 | 12 | 29 | 76 | 28 | 100 | 0.89 |
| JAISALMER | low_event | 3836 | 34 | 14 | 29 | 67 | 29 | 97 | 0.76 |
| GUNA | low_event | 3225 | 34 | 9 | 28 | 82 | 28 | 100 | 0.87 |
| GORAKHPUR | negative_control | 3324 | 34 | 5 | 17 | 53 | 10 | 98 | 0.51 |
| PATNA | negative_control | 3749 | 35 | 4 | 14 | 41 | 6 | 67 | 0.37 |
| KOTA | negative_control | 3367 | 35 | 5 | 10 | 25 | 8 | 100 | 0.3 |
| JODHPUR | low_event | 3518 | 35 | 2 | 6 | 26 | 6 | 98 | 0.17 |
| DR AMBEDKAR INTERNATIONAL | negative_control | 3414 | 35 | 0 | 1 | 2 | 1 | 45 | 0.03 |

`AMRITSAR` supplies 36% of all cold nights, so pooled results are dominated by it.
Test-period (2018+) cold nights per station:

| name | test_rows | test_le4 |
|---|---|---|
| HISSAR | 915 | 118 |
| AMRITSAR | 703 | 96 |
| SAFDARJUNG | 909 | 43 |
| GANGANAGAR | 925 | 32 |
| GWALIOR | 851 | 31 |
| PATIALA | 899 | 27 |
| BAREILLY | 895 | 24 |
| BIKANER | 506 | 19 |
| GAYA | 491 | 11 |
| GUNA | 937 | 7 |
| JAIPUR | 736 | 7 |
| LUCKNOW | 602 | 7 |
| KOTA | 913 | 7 |
| JAISALMER | 954 | 7 |
| SATNA | 906 | 4 |
| GORAKHPUR | 824 | 3 |
| DR AMBEDKAR INTERNATIONAL | 496 | 1 |
| JODHPUR | 897 | 0 |
| PATNA | 439 | 0 |

## Known limitations (read before using)

1. **Rule-derived labels.** No observed frost or damage. Crop-specific (Kinnow) threshold is still unsourced.
2. **Station series have breaks.** Hisar's cold-night rate is 2.8 per season in the 2000s and 19.2 in the 2010s while NASA POWER and ERA5 do not jump with it; Amritsar's declines 47 -> 33 -> 17 while the grids rise slightly. Treat station labels as an imperfect reference, not truth.
3. **Gridded sources run warm in the cold tail** by about +0.5 to +3.4 C (station-matched, `data/raw/india/grid_threshold_calibration.csv`), so a gridded label at the same threshold undercounts.
4. **Derived minima excluded as targets.** This removes most 1990s data for Amritsar, Hisar, Bikaner and Kota; effective record is roughly 2000-2024.
5. **Sparse where it matters.** Only four usable stations in the north-west belt, none at Abohar itself (nearest: Ganganagar, 37 km, 56% derived minima).
6. **Vidarbha has no cold nights.** Nagpur: 2 nights <= 4 C in 35 winters. It is a negative control, not training signal.
7. **Not validated for Warora/Vidarbha or any single deployment site.** Pooled across stations.
