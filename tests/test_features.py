import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (  # noqa: E402
    add_cyclical_time_features,
    add_daily_range_features,
    add_dew_point_features,
    add_lag_features,
    add_rate_of_change,
    add_rolling_features,
    dew_point_from_rh,
    humidity_fraction_to_percent,
)


def test_humidity_fraction_to_percent():
    fractions = pd.Series([0.0, 0.5, 0.98])
    result = humidity_fraction_to_percent(fractions)
    assert list(result) == [0.0, 50.0, 98.0]


def test_dew_point_equals_air_temp_at_saturation():
    # At 100% RH, dew point == air temperature.
    td = dew_point_from_rh(10.0, 100.0)
    assert abs(td - 10.0) < 0.05


def test_dew_point_is_lower_than_air_temp_when_not_saturated():
    td = dew_point_from_rh(5.0, 50.0)
    assert td < 5.0


def test_dew_point_decreases_as_humidity_drops_at_fixed_temp():
    td_humid = dew_point_from_rh(5.0, 90.0)
    td_dry = dew_point_from_rh(5.0, 40.0)
    assert td_dry < td_humid


def test_add_dew_point_features_adds_expected_columns_and_depression_sign():
    df = pd.DataFrame({"temp_min": [5.0], "humidity_pct": [50.0]})
    out = add_dew_point_features(df, temp_col="temp_min", rh_percent_col="humidity_pct")
    assert "dew_point_c" in out.columns
    assert "dew_point_depression_c" in out.columns
    # Dew point can't exceed air temperature, so the depression is >= 0.
    assert (out["dew_point_depression_c"] >= 0).all()


def test_add_lag_features_naming_and_shift_direction():
    df = pd.DataFrame({"temp_min": [1.0, 2.0, 3.0, 4.0]})
    out = add_lag_features(df, columns=["temp_min"], lags=[1, 2])
    assert "temp_min_lag1" in out.columns
    assert "temp_min_lag2" in out.columns
    # lag1 at row 2 (value 3.0) should be row 1's value (2.0)
    assert out["temp_min_lag1"].iloc[2] == 2.0
    assert out["temp_min_lag2"].iloc[2] == 1.0
    assert pd.isna(out["temp_min_lag1"].iloc[0])


def test_add_rolling_features_naming_and_values():
    df = pd.DataFrame({"temp_min": [4.0, 2.0, 6.0]})
    out = add_rolling_features(df, columns=["temp_min"], windows=[2], stats=["min", "mean"])
    assert "temp_min_roll2_min" in out.columns
    assert "temp_min_roll2_mean" in out.columns
    # Row 1: trailing window of 2 = [4.0, 2.0] -> min 2.0, mean 3.0
    assert out["temp_min_roll2_min"].iloc[1] == 2.0
    assert out["temp_min_roll2_mean"].iloc[1] == 3.0


def test_add_rate_of_change():
    df = pd.DataFrame({"temp_min": [1.0, 3.0, 0.0]})
    out = add_rate_of_change(df, columns=["temp_min"], periods=1)
    assert "temp_min_roc1" in out.columns
    assert out["temp_min_roc1"].iloc[1] == 2.0
    assert out["temp_min_roc1"].iloc[2] == -3.0


def test_cyclical_day_of_year_is_bounded_and_skips_hour_for_daily_data():
    df = pd.DataFrame({"DATE": pd.to_datetime(["2020-01-01", "2020-07-01"])})
    out = add_cyclical_time_features(df, date_col="DATE")
    assert out["doy_sin"].between(-1, 1).all()
    assert out["doy_cos"].between(-1, 1).all()
    # All timestamps are midnight -> hour encoding should be auto-skipped.
    assert "hour_sin" not in out.columns


def test_cyclical_hour_of_day_is_added_for_sub_daily_data():
    df = pd.DataFrame({"DATE": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 13:00"])})
    out = add_cyclical_time_features(df, date_col="DATE")
    assert "hour_sin" in out.columns
    assert "hour_cos" in out.columns


def test_add_daily_range_features():
    df = pd.DataFrame({"tmin": [2.0], "tmax": [20.0], "tmean": [10.0], "dewp": [4.0]})
    out = add_daily_range_features(df, "tmin", "tmax", "tmean", "dewp")
    assert out["diurnal_range_c"].iloc[0] == 18.0
    assert out["tmean_dewp_depression_c"].iloc[0] == 6.0
