import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from india_dataset import (  # noqa: E402
    PLUS_WIND, assign_split, build_india_dataset,
)
from label_rules import label_tmin_only  # noqa: E402


def build(w, summary):
    # synthetic fixtures are constant on purpose; the variance guard has its own tests below
    return build_india_dataset(w, summary, validate=False)


def _winter(sid="S1", season=2000, tmin=None, explicit=None, drop_days=()):
    days = pd.date_range(f"{season}-11-01", f"{season + 1}-03-31")
    n = len(days)
    tmin = np.full(n, 12.0) if tmin is None else np.asarray(tmin, float)
    explicit = np.ones(n, bool) if explicit is None else np.asarray(explicit, bool)
    df = pd.DataFrame({
        "STATION": sid, "DATE": days, "season": season, "tmin_c": tmin, "tmax_c": tmin + 12,
        "tmean_c": tmin + 6, "dewp_c": tmin, "rh_mean_pct": 70.0, "td_paper_c": tmin - 3,
        "explicit_min": explicit, "SLP": 1015.0 + 3 * np.sin(np.arange(n) / 3.0),
        "wdsp_ms": 1.0 + 0.5 * np.cos(np.arange(n) / 5.0)})
    return df.drop(index=list(drop_days)).reset_index(drop=True)


def _summary(sids, role="event_station"):
    return pd.DataFrame({"STATION": sids, "name": [f"{s}, IN" for s in sids],
                         "keep": True, "role": role})


def test_target_is_next_day_explicit_tmin_and_labels_are_inclusive():
    tmin = np.full(151, 12.0)
    tmin[10] = 4.0                                  # tomorrow-of-day-9: exactly 4.0 -> counts (<=)
    d = build(_winter(tmin=tmin), _summary(["S1"]))
    row = d[d["target_date"] == pd.Timestamp("2000-11-11")].iloc[0]
    assert row["target_tmin_c"] == 4.0
    assert row["label_le4"] == 1 and row["label_le3"] == 0 and row["label_le5"] == 1
    assert row["DATE"] == pd.Timestamp("2000-11-10")


def test_derived_next_day_minimum_is_dropped_as_a_target():
    explicit = np.ones(151, bool)
    explicit[20] = False                            # day 20 derived -> day 19 has no valid target
    d = build(_winter(explicit=explicit), _summary(["S1"]))
    assert pd.Timestamp("2000-11-20") not in set(d["DATE"])           # day 19 (Nov 20) row gone
    assert pd.Timestamp("2000-11-21") in set(d["DATE"])               # day 20 itself still a row


def test_missing_day_is_a_gap_not_a_shift_and_nothing_leaks_across_it():
    # remove Dec 10 (index 39): lags must not silently skip over it
    w = _winter(drop_days=[39])
    d = build(w, _summary(["S1"]))
    dec10, dec11 = pd.Timestamp("2000-12-10"), pd.Timestamp("2000-12-11")
    assert dec10 not in set(d["DATE"])
    # Dec 11's lag1 would be Dec 10 (missing): row must be absent, not filled with Dec 9
    assert dec11 not in set(d["DATE"])


def test_first_days_of_each_season_have_no_lags_and_are_dropped():
    d = build(_winter(), _summary(["S1"]))
    assert d["DATE"].min() >= pd.Timestamp("2000-11-04")             # needs 3 prior days
    assert d["DATE"].max() == pd.Timestamp("2001-03-30")             # last day has no tomorrow


def test_strict_label_requires_dry_dew_point():
    tmin = np.full(151, 12.0)
    tmin[10:12] = 2.0
    w = _winter(tmin=tmin)
    w.loc[10, "td_paper_c"] = -3.0                  # dry cold night -> strict
    w.loc[11, "td_paper_c"] = 1.0                   # humid cold night -> broad only
    d = build(w, _summary(["S1"])).set_index("target_date")
    assert d.loc[pd.Timestamp("2000-11-11"), "label_strict_le4"] == 1
    assert d.loc[pd.Timestamp("2000-11-12"), "label_strict_le4"] == 0
    assert d.loc[pd.Timestamp("2000-11-12"), "label_le4"] == 1


def test_labels_agree_with_label_rules_module():
    rng = np.random.default_rng(0)
    d = build(_winter(tmin=rng.uniform(-2, 15, 151)), _summary(["S1"]))
    for _, r in d.sample(25, random_state=1).iterrows():
        assert r["label_le4"] == label_tmin_only(pd.Series([r["target_tmin_c"]]), 4.0)


def test_no_station_identity_or_absolute_pressure_in_features():
    banned = {"STP", "SLP", "STATION", "LATITUDE", "LONGITUDE", "ELEVATION"}
    assert banned.isdisjoint(PLUS_WIND)
    assert not any(c.startswith(("city_", "station_")) for c in PLUS_WIND)


def test_split_is_by_season_and_stations_do_not_mix():
    assert list(assign_split(pd.Series([2000, 2013, 2014, 2017, 2018, 2024]))) == \
        ["train", "train", "val", "val", "test", "test"]
    w = pd.concat([_winter("A", 2000), _winter("B", 2000)])
    d = build(w, _summary(["A", "B"]))
    assert set(d["STATION"]) == {"A", "B"}
    # lags are computed per station: B's first usable row is also day 4 of the season
    assert d.groupby("STATION")["DATE"].min().nunique() == 1


def test_constant_feature_such_as_a_missing_code_fails_loudly():
    w = _winter()
    w["SLP"] = 999.9                       # an unmasked sentinel -> all-zero pressure changes
    try:
        build_india_dataset(w, _summary(["S1"]))
        assert False, "expected a ValueError for a constant pressure feature"
    except ValueError as e:
        assert "slp_change_1d" in str(e)


def test_missing_pressure_does_not_remove_rows():
    w = _winter()
    w.loc[w.index[::2], "SLP"] = np.nan    # half the pressure readings missing
    d = build(w, _summary(["S1"]))
    assert len(d) > 100 and d["slp_change_1d"].isna().any()
