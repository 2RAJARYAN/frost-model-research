"""
india_dataset.py — build the Tier 3 (India) station-day dataset with rule-derived
cold-night labels from GSOD winter data (output of scripts/gsod_download.py).

One row = one station-day (Nov-Mar). Features describe that day; the target is
NEXT day's minimum temperature, and the labels are the project's rule applied to it:

    label_le{3,4,5}     next-day Tmin <= 3 / 4 / 5 C        (primary: 4, matching IMD's
                                                              plains Cold Wave criterion
                                                              and Attri et al. 2020)
    label_strict_le4    next-day Tmin <= 4 C AND Td < 0 C   (Td from Tmin and daily-mean RH,
                                                              the Omazic et al. 2024 form;
                                                              RHmean is itself derived from
                                                              GSOD mean T and mean dew point)

These are RULE-DERIVED labels, not observed frost or crop damage. Read results as
skill at forecasting the rule's conditions.

Design decisions (each one is a bug we have already hit or a finding in the data):
  * Targets come from EXPLICIT-minimum days only. Derived minima (GSOD MIN_ATTRIBUTES '*')
    are warm-biased: within the same era, derived days show fewer cold nights in
    23 of 27 station-era cells.
  * Every station-season is reindexed to the full Nov 1 - Mar 31 calendar BEFORE any
    lag/rolling/shift, so a missing day becomes a NaN row instead of silently
    shifting lags, and nothing leaks across seasons or stations.
  * No station-identity features: no one-hot station, lat/lon/elevation, or absolute pressure
    (only pressure CHANGES). One-hot city columns are the flaw that made the v1 Europe model
    undeployable.
  * Pressure and wind are OPTIONAL (may be NaN); only the core temperature + humidity features
    are required for a row to exist. SLP is missing for roughly half the days at Amritsar,
    Lucknow, Jaipur and Nagpur, so requiring it would remove the dominant event station.
  * check_feature_variance() fails loudly on a constant or near-constant feature. A wrong
    missing-value code (STP = 999.9) once produced all-zero pressure features without any
    error, and the sensor ablation looked "identical to core" instead of crashing.
  * Feature names and construction follow src/features.py (the cross-repo contract).
"""

import numpy as np
import pandas as pd

from features import add_daily_range_features, build_feature_set

PRIMARY_THRESHOLD_C = 4
LABEL_THRESHOLDS_C = (3, 4, 5)
TRAIN_END_SEASON = 2013      # train: season <= 2013
VAL_END_SEASON = 2017        # val: 2014-2017 ; test: 2018-2024
KEEP_ROLES = ("event_station", "low_event", "negative_control")

_REQUIRED = ["STATION", "DATE", "season", "tmin_c", "tmax_c", "tmean_c", "dewp_c",
             "rh_mean_pct", "td_paper_c", "explicit_min", "SLP", "wdsp_ms"]

# --- feature groups (names produced by features.build_feature_set + extras below) ---
BASE_FEATURES = (
    ["tmin_c", "tmax_c", "tmean_c", "rh_mean_pct", "dew_point_c", "dew_point_depression_c",
     "diurnal_range_c", "tmean_dewp_depression_c"]
    + [f"{c}_lag{n}" for c in ("tmin_c", "rh_mean_pct") for n in (1, 2, 3)]
    + [f"{c}_roll{w}_{s}" for c in ("tmin_c", "rh_mean_pct") for w in (3, 7) for s in ("min", "mean", "max")]
    + ["tmin_c_roc1", "doy_sin", "doy_cos"]
)
PRESSURE_FEATURES = ["slp_change_1d", "slp_change_3d"]
WIND_FEATURES = ["wdsp_ms"]
CORE_FEATURES = BASE_FEATURES                       # needs a temperature + humidity sensor
PLUS_PRESSURE = BASE_FEATURES + PRESSURE_FEATURES   # + cheap pressure sensor
PLUS_WIND = PLUS_PRESSURE + WIND_FEATURES           # + anemometer


def check_feature_variance(d, min_unique=10):
    """Raise if any model feature is constant / near-constant (usually a missing-value sentinel)."""
    bad = []
    for c in PLUS_WIND:
        v = d[c].dropna()
        if len(v) == 0 or v.nunique() < min_unique:
            bad.append(f"{c} (n_unique={v.nunique()}, n_valid={len(v)})")
    if bad:
        raise ValueError("constant or near-constant features, check missing-value codes: " + "; ".join(bad))


def assign_split(season):
    return np.where(season <= TRAIN_END_SEASON, "train",
                    np.where(season <= VAL_END_SEASON, "val", "test"))


def _regularize(g, season):
    """Reindex one station-season to every day Nov 1 .. Mar 31 (missing days -> NaN rows)."""
    idx = pd.date_range(f"{season}-11-01", f"{season + 1}-03-31", freq="D", name="DATE")
    return g.drop_duplicates("DATE").set_index("DATE").reindex(idx).reset_index()


def _station_season(g, sid, season):
    g = _regularize(g, season)
    g["STATION"], g["season"] = sid, season
    g["explicit_min"] = g["explicit_min"].fillna(False).astype(bool)

    # features that follow the cross-repo contract (dew point, lags, rolling, roc, cyclical)
    f = build_feature_set(
        g, temp_col="tmin_c", rh_percent_col="rh_mean_pct", date_col="DATE",
        lag_columns=["tmin_c", "rh_mean_pct"], lags=(1, 2, 3),
        rolling_columns=["tmin_c", "rh_mean_pct"], rolling_windows=(3, 7))

    # extras: same-day quantities a temperature/humidity/pressure/wind station can measure
    f = add_daily_range_features(f, tmin_col="tmin_c", tmax_col="tmax_c", tmean_col="tmean_c", dewp_col="dewp_c")
    # Pressure CHANGE from sea-level pressure (SLP). Station pressure (STP) is NOT usable: in this GSOD
    # extract it is the constant 999.9 missing code. SLP is missing for ~50-70% of some stations.
    f["slp_change_1d"] = f["SLP"].diff(1)
    f["slp_change_3d"] = f["SLP"].diff(3)

    # target: next day's Tmin, only if that day has an explicit minimum
    nxt_explicit = f["explicit_min"].shift(-1).fillna(False).astype(bool)
    f["target_date"] = f["DATE"].shift(-1)
    f["target_tmin_c"] = f["tmin_c"].shift(-1).where(nxt_explicit)
    f["target_td_c"] = f["td_paper_c"].shift(-1).where(nxt_explicit)
    f["today_min_derived"] = ~f["explicit_min"]
    return f


def build_india_dataset(winter, summary, roles=KEEP_ROLES, validate=True):
    """winter: winter_data.csv.gz as a DataFrame; summary: station_summary.csv as a DataFrame."""
    missing = [c for c in _REQUIRED if c not in winter.columns]
    if missing:
        raise ValueError(f"winter data is missing columns: {missing}")
    s = summary[summary["keep"] & summary["role"].isin(roles)]
    role = s.set_index("STATION")["role"]
    name = s.set_index("STATION")["name"].str.replace(", IN", "", regex=False)
    w = winter[winter["STATION"].isin(role.index)].copy()
    w["DATE"] = pd.to_datetime(w["DATE"])

    frames = [_station_season(g, sid, int(season)) for (sid, season), g in w.groupby(["STATION", "season"])]
    d = pd.concat(frames, ignore_index=True)
    d = d[d["target_tmin_c"].notna()]
    d = d.dropna(subset=BASE_FEATURES)     # drops the first days of each season (no lags yet) and gaps
    if validate:
        check_feature_variance(d)

    d["role"] = d["STATION"].map(role)
    d["name"] = d["STATION"].map(name)
    d["split"] = assign_split(d["season"])
    for t in LABEL_THRESHOLDS_C:
        d[f"label_le{t}"] = (d["target_tmin_c"] <= t).astype(int)
    d["label_strict_le4"] = ((d["target_tmin_c"] <= 4) & (d["target_td_c"] < 0)).astype(int)

    keep = (["STATION", "name", "role", "DATE", "target_date", "season", "split"]
            + list(PLUS_WIND)
            + ["today_min_derived", "target_tmin_c", "target_td_c"]
            + [f"label_le{t}" for t in LABEL_THRESHOLDS_C] + ["label_strict_le4"])
    out = d[keep].copy()
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(3)
    return out.sort_values(["STATION", "DATE"]).reset_index(drop=True)
