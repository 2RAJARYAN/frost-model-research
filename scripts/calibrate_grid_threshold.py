#!/usr/bin/env python3
"""
Which GRIDDED threshold reproduces a station's rate of Tmin <= 4 C nights?
(Quantile matching; the "calibrated transfer" idea applied to the gridded label.)

For each site, over seasons where the station has >= 100 explicit-minimum days:
    p      = station cold nights / station explicit days     (Nov-Mar)
    t*     = p-quantile of the gridded Nov-Mar Tmin over the SAME seasons
    shift  = t* - 4.0   (> 0: the grid runs warm in the cold tail, so a gridded
                         label needs a HIGHER threshold to match the station)
An 80% interval comes from a block bootstrap over seasons (whole winters are
resampled, because nights within a winter are correlated).

How to read it
  * Shifts that cluster within about 1 C across sites AND eras, with overlapping
    intervals: one calibrated gridded threshold (4 + shift) is defensible.
  * Shifts that spread widely, or change sign across eras: a single correction is
    NOT supported. Report the Tier 3 label across a threshold range instead.
  * Eras with fewer than 5 matched seasons are skipped; pooled rows with few
    seasons (see `seasons`) are noisy, especially Ganganagar vs the Abohar grid.

Run from the repo root, after compare_station_vs_grid.py works:
    python scripts/calibrate_grid_threshold.py
    python scripts/calibrate_grid_threshold.py --selftest
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_station_vs_grid as C  # noqa: E402  (PAIRS, file finder, season table, constants)

N_BOOT = 300
MIN_ERA_SEASONS = 5


def grid_winter(path):
    """season -> array of Nov-Mar Tmin, complete seasons only."""
    raw = pd.read_csv(path)
    if not {"date", "tmin_c"} <= set(raw.columns):
        raise ValueError(f"{path}: expected columns date,tmin_c but found {list(raw.columns)}")
    d = raw.assign(date=pd.to_datetime(raw["date"])).dropna(subset=["tmin_c"])
    d = d[d["date"].dt.month.isin([11, 12, 1, 2, 3])]
    d = d.assign(season=np.where(d["date"].dt.month >= 11, d["date"].dt.year, d["date"].dt.year - 1))
    return {int(s): x["tmin_c"].round(2).to_numpy() for s, x in d.groupby("season")
            if len(x) >= C.MIN_GRID_SEASON_DAYS}


def _shift(draw, st, grid):
    p = st.loc[draw, "hits"].sum() / st.loc[draw, "explicit_days"].sum()
    return np.quantile(np.concatenate([grid[s] for s in draw]), p) - C.THR


def shift_with_interval(seasons, st, grid, seed=0):
    est = _shift(seasons, st, grid)
    rng = np.random.default_rng(seed)
    boots = [_shift(rng.choice(seasons, size=len(seasons), replace=True), st, grid)
             for _ in range(N_BOOT)]
    lo, hi = np.percentile(boots, [10, 90])
    return round(est, 1), round(lo, 1), round(hi, 1)


def calibrate(winter, grid_dir):
    rows = []
    for sid, (label, site) in C.PAIRS.items():
        st = C.station_season_table(winter, sid)
        if st.empty:
            continue
        grids = {}
        for src in ("nasa", "era5"):
            f = C.find_grid_file(grid_dir, site, src)
            if f is None:
                C.WARNINGS.append(f"no {src} file for '{site}' under {grid_dir}")
                continue
            grids[src] = grid_winter(f)
        if len(grids) < 2:
            continue
        matched = [s for s in st.index if all(s in g for g in grids.values())]
        if not matched:
            continue
        era = pd.cut(pd.Series(matched, index=matched), bins=C.ERA_BINS, labels=C.ERA_LABELS)
        groups = [("all", matched)] + [(lab, [s for s in matched if era[s] == lab]) for lab in C.ERA_LABELS]
        for lab, ss in groups:
            if lab != "all" and len(ss) < MIN_ERA_SEASONS:
                continue
            rate = st.loc[ss, "hits"].sum() / st.loc[ss, "explicit_days"].sum() * 151
            if rate < C.MIN_RATE_FOR_RATIO:
                rows.append({"site": label, "era": lab, "seasons": len(ss),
                             "station_per_season": round(rate, 1), "note": "too few cold nights"})
                continue
            row = {"site": label, "era": lab, "seasons": len(ss), "station_per_season": round(rate, 1)}
            for src, g in grids.items():
                row[f"{src}_shift"], row[f"{src}_lo"], row[f"{src}_hi"] = shift_with_interval(ss, st, g)
            rows.append(row)
    return pd.DataFrame(rows)


def selftest():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "nasa").mkdir(); (t / "era5").mkdir()
        def season_grid(y):                      # 151 days, Tmin = 0,1,...,150
            d = pd.date_range(f"{y}-11-01", f"{y + 1}-03-31")
            return pd.DataFrame({"date": d, "tmin_c": np.arange(len(d), dtype=float)})
        g = pd.concat([season_grid(2000), season_grid(2001), season_grid(2002)])  # 2002 has no station
        g.to_csv(t / "nasa" / "hisar.csv", index=False)
        g.to_csv(t / "era5" / "hisar.csv", index=False)
        rows = []
        for y in (2000, 2001):                   # 100 explicit days, 10 cold -> p = 0.10
            le = np.zeros(100, bool); le[:10] = True
            rows.append(pd.DataFrame({"STATION": "42131099999", "season": y,
                                      "derived_min": False, "le_4C": le}))
        winter = pd.concat(rows)
        C.WARNINGS.clear()
        out = calibrate(winter, t)
        r = out[(out["site"] == "Hisar") & (out["era"] == "all")].iloc[0]
        # pooled grid over the 2 matched seasons: each value 0..150 twice; q(0.10) = 15.0; shift = 11.0
        assert r["seasons"] == 2, r
        assert r["nasa_shift"] == 11.0 and r["era5_shift"] == 11.0, r
        assert r["nasa_lo"] <= 11.0 <= r["nasa_hi"], r
        assert not C.WARNINGS, C.WARNINGS
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winter", default="data/raw/india/winter_data.csv.gz")
    ap.add_argument("--grid-dir", default="data/raw/india/tmin")
    ap.add_argument("--out", default="data/raw/india/grid_threshold_calibration.csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    winter = pd.read_csv(a.winter, usecols=["STATION", "season", "derived_min", "le_4C"],
                         dtype={"STATION": str})
    res = calibrate(winter, a.grid_dir)
    pd.set_option("display.width", 220)
    print(f"Gridded-threshold shift (C) that reproduces the station's Tmin <= {C.THR:g} C rate; "
          "interval = 10th-90th percentile of a season block bootstrap\n")
    print(res.to_string(index=False))
    res.to_csv(a.out, index=False)
    print(f"\nsaved -> {a.out}")
    if C.WARNINGS:
        print("\nWARNING:")
        for w in sorted(set(C.WARNINGS)):
            print("   ", w)
        sys.exit(1)