#!/usr/bin/env python3
"""
Is the warm bias of "derived" Tmin real, or just an artefact of WHEN derived days occur?

Observation to test: across stations, days whose Tmin is derived from synoptic
readings (GSOD MIN_ATTRIBUTES "*") show far fewer cold nights than days with an
explicit minimum report. That is what a warm bias would produce. But if derived
days cluster in one era (and the climate or station changed between eras), the
gap could be an era effect instead.

This compares, WITHIN each era, the share of derived days and the cold-night
frequency (Tmin <= 4 C) on explicit vs derived days.

How to read it
  * freq_derived_% clearly BELOW freq_explicit_% in the SAME era  -> warm bias is
    real: label only from explicit-minimum days, or treat derived days as missing.
  * The two frequencies are similar within eras                    -> the earlier gap
    was an era effect; derived days can probably stay.
  * Few days in a cell (see days / derived columns) means noisy percentages.

Run from the repo root, after `gsod_download.py summarize`:
    python scripts/check_derived_min_by_era.py
    python scripts/check_derived_min_by_era.py --all-stations
    python scripts/check_derived_min_by_era.py --selftest
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [1989, 1999, 2009, 2019, 2024]
LABELS = ["1990s", "2000s", "2010s", "2020-24"]
THR_COL = "le_4C"


def by_era(daily):
    d = daily.copy()
    d["era"] = pd.cut(d["season"], bins=BINS, labels=LABELS)
    d["expl_hit"] = d[THR_COL] & ~d["derived_min"]
    d["der_hit"] = d[THR_COL] & d["derived_min"]
    g = d.groupby(["STATION", "era"], observed=True).agg(
        days=(THR_COL, "size"), derived=("derived_min", "sum"),
        expl_hit=("expl_hit", "sum"), der_hit=("der_hit", "sum")).reset_index()
    g["explicit"] = g["days"] - g["derived"]
    g["derived_share_%"] = (100 * g["derived"] / g["days"]).round(1)
    g["freq_explicit_%"] = (100 * g["expl_hit"] / g["explicit"].replace(0, np.nan)).round(2)
    g["freq_derived_%"] = (100 * g["der_hit"] / g["derived"].replace(0, np.nan)).round(2)
    return g[["STATION", "era", "days", "derived", "derived_share_%",
              "freq_explicit_%", "freq_derived_%"]]


def selftest():
    rng = np.random.default_rng(0)
    rows = []
    # one era: 1000 explicit days with 100 cold nights, 1000 derived days with 20
    for derived, n, hits in ((False, 1000, 100), (True, 1000, 20)):
        flag = np.zeros(n, bool)
        flag[:hits] = True
        rows.append(pd.DataFrame({"STATION": "A", "season": 2005, "derived_min": derived, THR_COL: flag}))
    out = by_era(pd.concat(rows, ignore_index=True)).iloc[0]
    assert out["era"] == "2000s" and out["days"] == 2000
    assert out["derived_share_%"] == 50.0
    assert out["freq_explicit_%"] == 10.0 and out["freq_derived_%"] == 2.0, out
    # an era with no derived days must not divide by zero
    none = pd.DataFrame({"STATION": "B", "season": 2021, "derived_min": [False] * 10, THR_COL: [True] + [False] * 9})
    o2 = by_era(none).iloc[0]
    assert o2["era"] == "2020-24" and np.isnan(o2["freq_derived_%"]) and o2["freq_explicit_%"] == 10.0
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winter", default="data/raw/india/winter_data.csv.gz")
    ap.add_argument("--summary", default="data/raw/india/station_summary.csv")
    ap.add_argument("--out", default="data/raw/india/derived_min_by_era.csv")
    ap.add_argument("--all-stations", action="store_true", help="include stations not marked keep")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)

    summ = pd.read_csv(a.summary, dtype={"STATION": str})
    daily = pd.read_csv(a.winter, usecols=["STATION", "season", "derived_min", THR_COL],
                        dtype={"STATION": str})
    if not a.all_stations:
        daily = daily[daily["STATION"].isin(summ.loc[summ["keep"], "STATION"])]
    res = by_era(daily).merge(summ[["STATION", "name", "role"]], on="STATION", how="left")
    res = res.sort_values(["role", "name", "era"])
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    cols = ["name", "role", "era", "days", "derived_share_%", "freq_explicit_%", "freq_derived_%"]
    print(res[cols].to_string(index=False))
    res.to_csv(a.out, index=False)
    print(f"\nsaved -> {a.out}")