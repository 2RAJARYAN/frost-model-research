import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from india_eval import (  # noqa: E402
    block_counts, bootstrap_ci, bootstrap_diff_ci, bootstrap_mean_diff_ci, choose_threshold,
    climatology_persistence_predict, climatology_table, confusion_counts, day_of_season,
    skill_from_counts,
)


def test_counts_and_skill_hand_computed():
    # y_true: 1 1 1 1 0 0 0 0 0 0 ; y_pred: 1 1 1 0 1 0 0 0 0 0
    y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    p = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    assert confusion_counts(y, p) == (3, 1, 1, 5)
    s = skill_from_counts(3, 1, 1, 5)
    assert s["pod"] == 0.75 and abs(s["pofd"] - 1 / 6) < 1e-12
    assert abs(s["pss"] - (0.75 - 1 / 6)) < 1e-12 and s["precision"] == 0.75


def test_skill_with_no_events_is_nan_not_a_crash():
    s = skill_from_counts(0, 2, 0, 8)
    assert np.isnan(s["pod"]) and np.isnan(s["pss"])


def test_block_counts_sum_to_totals():
    blocks = ["a", "a", "b", "b", "c"]
    y = [1, 0, 1, 0, 1]
    p = [1, 1, 0, 0, 1]
    bc = block_counts(blocks, y, p)
    assert bc.shape == (3, 4) and tuple(bc.sum(axis=0)) == confusion_counts(y, p)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_wide_for_few_blocks():
    rng = np.random.default_rng(0)
    n_blocks = 40
    blocks = np.repeat(np.arange(n_blocks), 50)
    y = rng.random(len(blocks)) < 0.1
    p = np.where(rng.random(len(blocks)) < 0.7, y, ~y & (rng.random(len(blocks)) < 0.05))
    bc = block_counts(blocks, y, p)
    point = skill_from_counts(*bc.sum(axis=0))["pss"]
    lo, hi = bootstrap_ci(bc, B=500)
    assert lo <= point <= hi
    lo2, hi2 = bootstrap_ci(bc[:5], B=500)
    assert (hi2 - lo2) > (hi - lo)               # fewer blocks -> wider interval


def test_paired_diff_ci_excludes_zero_for_a_clearly_better_forecast():
    rng = np.random.default_rng(1)
    blocks = np.repeat(np.arange(30), 60)
    y = rng.random(len(blocks)) < 0.1
    good = y.copy()                                         # perfect
    bad = rng.random(len(blocks)) < 0.1                     # random
    lo, hi = bootstrap_diff_ci(block_counts(blocks, y, good), block_counts(blocks, y, bad), B=500)
    assert lo > 0
    e = np.where(y, 0.0, 0.0)
    lo, hi = bootstrap_mean_diff_ci(e, e + 1.0, blocks, B=300)
    assert hi < 0                                           # a is exactly 1.0 better in MAE


def test_choose_threshold_picks_the_separating_value():
    score = np.array([0, 1, 2, 3, 7, 8, 9, 10], float)      # events are the low scores
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    t, s = choose_threshold(y, score, [1.0, 3.0, 5.0, 8.0])
    assert t == 3.0 and s == 1.0


def test_day_of_season_and_climatology_persistence_exactly_recovers_a_seasonal_cycle():
    # Tmin follows a pure seasonal curve + a constant anomaly of +2: the baseline must
    # predict tomorrow = climatology(tomorrow) + 2 exactly.
    rows = []
    for season in range(2000, 2005):
        days = pd.date_range(f"{season}-11-01", f"{season + 1}-03-31")
        dos = np.arange(len(days))
        base = 15 - 0.08 * dos
        rows.append(pd.DataFrame({
            "STATION": "S", "season": season, "DATE": days, "target_date": days + pd.Timedelta(days=1),
            "tmin_c": base + (2 if season == 2004 else 0),
            "target_tmin_c": 15 - 0.08 * (dos + 1) + (2 if season == 2004 else 0)}))
    d = pd.concat(rows)
    assert list(day_of_season(pd.Series(pd.to_datetime(["2000-11-01", "2001-03-31"])), pd.Series([2000, 2000]))) == [0, 150]
    clim = climatology_table(d[d["season"] < 2004], smooth=3)
    # The real dataset never contains the first 3 days of a season (no lags yet) and the climatology
    # is learned from next-day targets, so day 0 has no observation: test the rows that can occur.
    test = d[(d["season"] == 2004) & (d["DATE"] >= "2004-11-04") & (d["DATE"] < "2005-03-30")]
    pred = climatology_persistence_predict(clim, test)
    assert np.allclose(pred, test["target_tmin_c"].to_numpy(), atol=0.05)
