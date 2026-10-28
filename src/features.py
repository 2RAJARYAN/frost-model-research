"""
features.py — SINGLE SOURCE OF TRUTH for feature engineering.

Repo B (frost-edge-system)'s `feature_pipeline.py` must reproduce this file
exactly: same lag windows, same rolling stats, same units, same column
names. That's the whole cross-repo contract. Concretely:

  - Functions here are pure (no I/O, no globals) so they can be copied or
    imported into Repo B verbatim.
  - Every function documents its exact output column-naming convention
    below — Repo B's edge pipeline must match these names character-for-
    character, since the exported model was trained on them.

NAMING CONVENTION
------------------
  Lag feature:        "{col}_lag{n}"            n = number of ROWS back
                                                 (rows are one time step
                                                 of whatever cadence the
                                                 input df uses: daily for
                                                 Tier 1, hourly for Tier
                                                 2/3 — NOT literally "n
                                                 hours" unless the input
                                                 is hourly).
  Rolling stat:        "{col}_roll{window}_{stat}"   stat in {min, mean, max, std}
  Rate of change:      "{col}_roc{periods}"
  Dew point:           "dew_point_c"             deg C
  Dew point depression:"dew_point_depression_c"  deg C, = temp - dew_point
  Daily range / depression: "diurnal_range_c" = tmax - tmin,
                       "tmean_dewp_depression_c" = tmean - mean dew point
  Cyclical day-of-year:"doy_sin", "doy_cos"
  Cyclical hour-of-day:"hour_sin", "hour_cos"     (only added if the input
                                                   timestamps carry a
                                                   sub-daily time component)

UNITS WARNING (read this before wiring up a new dataset)
----------------------------------------------------------
Relative humidity is NOT always on the same scale across sources. The
Tier-1 dataset (florian-huber/weather_prediction_dataset) stores humidity
as a FRACTION in [0, 1]. `dew_point_from_rh` below expects PERCENT
(0-100), matching the standard Magnus-formula convention. Convert with
`humidity_fraction_to_percent()` first if your source uses fractions —
silently feeding a 0-1 fraction into a formula that expects 0-100 produces
a dew point that is wrong by a huge margin (log(0.75) vs log(75)), and it
fails silently rather than raising, so check this explicitly for every new
data source instead of assuming.
"""

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Humidity / dew point
# ---------------------------------------------------------------------------

def humidity_fraction_to_percent(rh_fraction: pd.Series) -> pd.Series:
    """Convert a 0-1 relative-humidity fraction to 0-100 percent."""
    return rh_fraction * 100.0


def dew_point_from_rh(temp_c, rh_percent):
    """
    Approximate dew point (deg C) from air temperature and relative
    humidity, via the Magnus formula (the standard practical approximation
    to the Clausius-Clapeyron relation used for this purpose).

    Args:
        temp_c: air temperature in deg C (scalar, array, or pandas Series).
        rh_percent: relative humidity in PERCENT, 0-100 (same shape as
            temp_c). If computing Omazić et al.'s validated frost-label
            variant, pass the *daily mean* RH (RHmean), not a single
            timestamp's reading — their "Method 7" (RHmean-based) scored
            better than "Method 6" (single 7:00 CET reading).

    Returns:
        Dew point in deg C, same shape as input.
    """
    temp_c = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_percent, dtype=float)
    rh = np.clip(rh, 1e-6, 100.0)  # guard against log(0) on bad/zero readings

    a, b = 17.27, 237.7
    alpha = (a * temp_c) / (b + temp_c) + np.log(rh / 100.0)
    td = (b * alpha) / (a - alpha)
    return td


def add_dew_point_features(
    df: pd.DataFrame,
    temp_col: str,
    rh_percent_col: str,
    out_prefix: str = "",
) -> pd.DataFrame:
    """
    Adds `{out_prefix}dew_point_c` and `{out_prefix}dew_point_depression_c`
    to a copy of `df`. `rh_percent_col` must already be on a 0-100 scale —
    use `humidity_fraction_to_percent()` first if it isn't.
    """
    out = df.copy()
    td_col = f"{out_prefix}dew_point_c"
    dep_col = f"{out_prefix}dew_point_depression_c"
    out[td_col] = dew_point_from_rh(out[temp_col], out[rh_percent_col])
    out[dep_col] = out[temp_col] - out[td_col]
    return out


# ---------------------------------------------------------------------------
# Lag / rolling / rate-of-change features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame, columns: Sequence[str], lags: Sequence[int]
) -> pd.DataFrame:
    """
    Adds `{col}_lag{n}` for each column/lag combination. `n` is a number of
    ROWS back, i.e. it means "n time steps" at whatever cadence the input
    df already uses (see module docstring) — it is not automatically hours.
    """
    out = df.copy()
    for col in columns:
        for n in lags:
            out[f"{col}_lag{n}"] = out[col].shift(n)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    windows: Sequence[int],
    stats: Iterable[str] = ("min", "mean", "max"),
) -> pd.DataFrame:
    """
    Adds `{col}_roll{window}_{stat}` for each column/window/stat
    combination, computed over the trailing `window` rows (inclusive of
    the current row), using only past-and-current data so it's safe to
    compute at inference time on a live edge device.
    """
    out = df.copy()
    for col in columns:
        for w in windows:
            rolling = out[col].rolling(window=w, min_periods=1)
            for stat in stats:
                if not hasattr(rolling, stat):
                    raise ValueError(f"Unsupported rolling stat: {stat!r}")
                out[f"{col}_roll{w}_{stat}"] = getattr(rolling, stat)()
    return out


def add_rate_of_change(
    df: pd.DataFrame, columns: Sequence[str], periods: int = 1
) -> pd.DataFrame:
    """Adds `{col}_roc{periods}` = df[col].diff(periods)."""
    out = df.copy()
    for col in columns:
        out[f"{col}_roc{periods}"] = out[col].diff(periods)
    return out


# ---------------------------------------------------------------------------
# Cyclical time encoding
# ---------------------------------------------------------------------------

def add_cyclical_time_features(
    df: pd.DataFrame, date_col: str = "DATE", include_hour: bool = None
) -> pd.DataFrame:
    """
    Adds `doy_sin`/`doy_cos` (day-of-year, period 365.25) always, and
    `hour_sin`/`hour_cos` (period 24) when the timestamps carry a
    sub-daily component.

    Args:
        include_hour: force hour-of-day encoding on/off. Defaults to
            auto-detect (on only if any timestamp has a non-midnight time),
            since Tier-1 data is daily and hour encoding would just be a
            useless constant column for it.
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])

    doy = dt.dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if include_hour is None:
        include_hour = bool((dt.dt.hour != 0).any() or (dt.dt.minute != 0).any())

    if include_hour:
        hour = dt.dt.hour + dt.dt.minute / 60.0
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    return out


def add_daily_range_features(
    df: pd.DataFrame, tmin_col: str, tmax_col: str, tmean_col: str, dewp_col: str
) -> pd.DataFrame:
    """Adds `diurnal_range_c` (tmax - tmin) and `tmean_dewp_depression_c` (tmean - mean dew point).
    Used by the India (Tier 3) feature set; Repo B must compute these identically."""
    out = df.copy()
    out["diurnal_range_c"] = out[tmax_col] - out[tmin_col]
    out["tmean_dewp_depression_c"] = out[tmean_col] - out[dewp_col]
    return out


def build_feature_set(
    df: pd.DataFrame,
    temp_col: str,
    rh_percent_col: str,
    date_col: str = "DATE",
    lag_columns: Sequence[str] = None,
    lags: Sequence[int] = (1, 2, 3),
    rolling_columns: Sequence[str] = None,
    rolling_windows: Sequence[int] = (3, 7),
) -> pd.DataFrame:
    """
    Convenience wrapper chaining the pieces above in one documented order.
    This is the function Repo B should call at inference time on its
    incoming sensor buffer, and the one this repo's training notebooks
    should call, so both sides run identical logic instead of two
    hand-written copies drifting apart.
    """
    if lag_columns is None:
        lag_columns = [temp_col, rh_percent_col]
    if rolling_columns is None:
        rolling_columns = [temp_col, rh_percent_col]

    out = add_dew_point_features(df, temp_col=temp_col, rh_percent_col=rh_percent_col)
    out = add_lag_features(out, columns=lag_columns, lags=lags)
    out = add_rolling_features(out, columns=rolling_columns, windows=rolling_windows)
    out = add_rate_of_change(out, columns=[temp_col])
    out = add_cyclical_time_features(out, date_col=date_col)
    return out
