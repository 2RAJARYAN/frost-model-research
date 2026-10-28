"""
Logic-only unit tests for india_label_feasibility.count_events().

These test the counting/windowing logic against constructed series with a
known answer -- no network access, no real API calls. Run with:
    python -m pytest scripts/test_india_label_feasibility.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from india_label_feasibility import count_events  # noqa: E402


def _winter_season(start_year, cold_days=0, cold_temp=1.0, warm_temp=15.0):
    """Build one full Nov 1 -> Mar 31 season as a daily series, with the
    first `cold_days` days of the season at `cold_temp` and the rest at
    `warm_temp`."""
    dates = pd.date_range(f"{start_year}-11-01", f"{start_year + 1}-03-31", freq="D")
    values = [cold_temp] * cold_days + [warm_temp] * (len(dates) - cold_days)
    return pd.Series(values, index=dates)


def test_known_answer_four_cold_days_one_winter():
    """The exact case the handoff asked to keep: a series with 4 days at
    1C in one winter must give 4 positive days, 1 winter (at >=3)."""
    series = _winter_season(2020, cold_days=4, cold_temp=1.0)
    result = count_events(series, thresholds=[2, 3, 4, 5])

    row = result[result.threshold_C == 4].iloc[0]
    assert row.pos_days == 4
    assert row.seasons == 1
    assert row.winters_ge1 == 1
    assert row.winters_ge3 == 1
    assert row.max_days_one_winter == 4


def test_two_winters_only_one_has_cold_days():
    winter_cold = _winter_season(2020, cold_days=4, cold_temp=1.0)
    winter_warm = _winter_season(2021, cold_days=0)
    series = pd.concat([winter_cold, winter_warm])

    result = count_events(series, thresholds=[4])
    row = result.iloc[0]
    assert row.seasons == 2        # both winters counted
    assert row.pos_days == 4       # only from the cold winter
    assert row.winters_ge1 == 1
    assert row.winters_ge3 == 1


def test_threshold_is_inclusive_not_strict():
    """Regression test for the <= fix: a day at EXACTLY the threshold
    temperature must count. The original script used `<` (strict), which
    silently excluded exactly this case -- disagreeing with both IMD's
    Cold Wave criterion and Attri et al.'s frost-day definition, both of
    which are stated as <=."""
    series = _winter_season(2020, cold_days=3, cold_temp=4.0)  # exactly 4.0C
    result = count_events(series, thresholds=[4])
    row = result.iloc[0]
    assert row.pos_days == 3, (
        "Days at exactly the threshold temperature must count (<=), not be excluded (<)"
    )


def test_incomplete_season_is_dropped():
    """A season with fewer than 120 of 151 Nov-Mar days present should be
    excluded entirely, not counted as a (misleadingly complete-looking)
    winter with a low season-day denominator."""
    dates = pd.date_range("2020-11-01", "2020-12-15", freq="D")  # only 45 days
    short_season = pd.Series([1.0] * len(dates), index=dates)

    full_season = _winter_season(2021, cold_days=4, cold_temp=1.0)
    series = pd.concat([short_season, full_season])

    result = count_events(series, thresholds=[4])
    row = result.iloc[0]
    assert row.seasons == 1  # only the complete winter counted
    assert row.pos_days == 4  # the short season's cold days are excluded


def test_zero_cold_days_gives_zero_everywhere():
    """The Vidarbha case: a site with no cold days at all should report
    cleanly as all zeros, not error out or report misleading NaNs."""
    series = _winter_season(2020, cold_days=0)
    result = count_events(series, thresholds=[2, 3, 4, 5])
    assert (result.pos_days == 0).all()
    assert (result.winters_ge1 == 0).all()
    assert (result.winters_ge3 == 0).all()
