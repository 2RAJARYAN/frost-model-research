#!/usr/bin/env python3
"""
Compare station-based (GSOD) cold-night rates with gridded NASA POWER / ERA5,
SEASON BY SEASON, so the sources are compared over the same winters.

Why season-matched: a station's explicit-minimum days come from different eras
than the 1990-2024 gridded average (e.g. Ganganagar's explicit days are all
2010-2024; Amritsar/Hisar have almost none from the 1990s). Comparing the
station's recent-era rate to a 35-year gridded mean mixes eras and climate trend.

Method
  * Station: only days with an EXPLICIT Tmin (derived minima are warm-biased).
    A season is "matched" if the station has >= 100 explicit days in it.
    Station rate = cold nights / explicit days * 151, over matched seasons.
  * Gridded: mean nights/season over the SAME matched seasons.
  * "Cold night" = Tmin <= 4 C (inclusive).
  * Output 1: pooled over all matched seasons. Output 2: split by era, which
    exposes breaks in a station series (a gridded series should not jump).

Run from the repo root, after `gsod_download.py summarize`:
    python scripts/compare_station_vs_grid.py
    python scripts/compare_station_vs_grid.py --selftest

Attri values are as quoted in the project handoff, NOT re-checked, and cover
1980-2015 (a different period).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

THR = 4.0
Y0, Y1 = 1990, 2025                  # seasons 1990 .. 2024
MIN_GRID_SEASON_DAYS = 120
MIN_EXPLICIT_DAYS = 100              # per season, to count a station season as matched
ERA_BINS = [1989, 1999, 2009, 2019, 2024]
ERA_LABELS = ["1990s", "2000s", "2010s", "2020-24"]
MIN_RATE_FOR_RATIO = 0.5             # below this a ratio is noise (e.g. Nagpur)

PAIRS = {  # GSOD station id -> (label, grid site key)
    "42071099999": ("Amritsar", "amritsar"),
    "42131099999": ("Hisar", "hisar"),
    "42101099999": ("Patiala", "patiala"),
    "42348099999": ("Jaipur", "jaipur"),
    "42182099999": ("Delhi (Safdarjung)", "delhi"),
    "42123099999": ("Ganganagar vs Abohar grid (~37 km)", "abohar"),
    "42867099999": ("Nagpur", "nagpur"),
}
ATTRI = {"amritsar": 42, "hisar": 6, "patiala": 5, "jaipur": 3, "delhi": 2}

WARNINGS = []    # a missing file must never look like a result
USED = []


def find_grid_file(grid_dir, site, src):
    """Locate the series for (site, source) under grid_dir, whatever the layout."""
    root = Path(grid_dir)
    hits = [f for f in root.rglob(f"*{site}*.csv")
            if any(src in part.lower() for part in f.relative_to(root).parts)]
    return sorted(hits, key=lambda f: (len(f.parts), str(f)))[0] if hits else None


def grid_season_counts(path):
    """Series: season -> nights with Tmin <= THR, complete Nov-Mar seasons only."""
    if path is None or not Path(path).exists():
        WARNINGS.append(f"missing file: {path}")
        return pd.Series(dtype=float)
    raw = pd.read_csv(path)
    if not {"date", "tmin_c"} <= set(raw.columns):
        raise ValueError(f"{path}: expected columns date,tmin_c but found {list(raw.columns)}")
    d = raw.assign(date=pd.to_datetime(raw["date"])).dropna(subset=["tmin_c"])
    d = d[d["date"].dt.month.isin([11, 12, 1, 2, 3])]
    d["season"] = np.where(d["date"].dt.month >= 11, d["date"].dt.year, d["date"].dt.year - 1)
    d["hit"] = d["tmin_c"].round(2) <= THR
    per = d.groupby("season").agg(days=("hit", "size"), n=("hit", "sum"))
    per = per[(per["days"] >= MIN_GRID_SEASON_DAYS) & per.index.isin(range(Y0, Y1))]
    if per.empty:
        WARNINGS.append(f"no complete seasons in {Y0}-{Y1 - 1}: {path}")
    return per["n"]


def station_season_table(winter, sid):
    """Per season: explicit days and cold nights among them (matched seasons only)."""
    w = winter[winter["STATION"] == sid]
    expl = w[~w["derived_min"]]
    t = expl.groupby("season").agg(explicit_days=("le_4C", "size"), hits=("le_4C", "sum"))
    return t[(t["explicit_days"] >= MIN_EXPLICIT_DAYS) & t.index.isin(range(Y0, Y1))]


def _rate(hits, days):
    return hits / days * 151 if days else np.nan


def compare(winter, grid_dir):
    pooled, by_era = [], []
    for sid, (label, site) in PAIRS.items():
        st = station_season_table(winter, sid)
        if st.empty:
            pooled.append({"site": label, "note": "no explicit-minimum seasons for this station"})
            continue
        files = {src: find_grid_file(grid_dir, site, src) for src in ("nasa", "era5")}
        for src, f in files.items():
            USED.append((site, src, f))
            if f is None:
                WARNINGS.append(f"no {src} file for '{site}' under {grid_dir}")
        grids = {src: grid_season_counts(f) for src, f in files.items()}
        matched = [s for s in st.index if all(s in g.index for g in grids.values() if len(g))]
        if not matched:
            WARNINGS.append(f"{label}: no seasons shared by station and gridded data")
            continue
        sm = st.loc[matched]
        srate = _rate(sm["hits"].sum(), sm["explicit_days"].sum())
        row = {"site": label, "matched_seasons": len(matched), "station": round(srate, 1)}
        for src, g in grids.items():
            row[src] = round(g.reindex(matched).mean(), 1) if len(g) else np.nan
            row[f"{src}/station"] = round(row[src] / srate, 2) if srate >= MIN_RATE_FOR_RATIO else np.nan
        row["attri_quoted"] = ATTRI.get(site, np.nan)
        pooled.append(row)

        era = pd.cut(pd.Series(matched, index=matched), bins=ERA_BINS, labels=ERA_LABELS)
        for lab in ERA_LABELS:
            ss = [s for s in matched if era[s] == lab]
            if not ss:
                continue
            e = sm.loc[ss]
            by_era.append({"site": label, "era": lab, "seasons": len(ss),
                           "station": round(_rate(e["hits"].sum(), e["explicit_days"].sum()), 1),
                           **{src: (round(g.reindex(ss).mean(), 1) if len(g) else np.nan)
                              for src, g in grids.items()}})
    return pd.DataFrame(pooled), pd.DataFrame(by_era)


def selftest():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "nasa").mkdir(); (t / "era5").mkdir()
        # grid: seasons 1995 (9 cold nights), 2000 (6), 2001 (15)
        def season_df(y, n):
            d = pd.date_range(f"{y}-11-01", f"{y + 1}-03-31")
            v = np.full(len(d), 15.0); v[:n] = 4.0          # exactly 4.0 -> counts (inclusive)
            return pd.DataFrame({"date": d, "tmin_c": v})
        grid = pd.concat([season_df(1995, 9), season_df(2000, 6), season_df(2001, 15)])
        grid.to_csv(t / "nasa" / "hisar.csv", index=False)
        grid.to_csv(t / "era5" / "tmin_hisar_era5.csv", index=False)
        assert find_grid_file(t, "hisar", "nasa").name == "hisar.csv"
        assert find_grid_file(t, "amritsar", "nasa") is None
        # station: 1995 explicit 151d/3 hits; 2000 explicit 151d/6 hits; 2001 ALL derived -> unmatched
        def wint(y, n, derived, hits):
            d = pd.date_range(f"{y}-11-01", f"{y + 1}-03-31")
            le = np.zeros(len(d), bool); le[:hits] = True
            return pd.DataFrame({"STATION": "42131099999", "season": y, "derived_min": derived, "le_4C": le})
        winter = pd.concat([wint(1995, 0, False, 3), wint(2000, 0, False, 6), wint(2001, 0, True, 100)])
        WARNINGS.clear(); USED.clear()
        pooled, era = compare(winter, t)
        r = pooled[pooled["site"] == "Hisar"].iloc[0]
        assert r["matched_seasons"] == 2, r                    # the all-derived season is excluded
        assert r["station"] == 4.5 and r["nasa"] == 7.5, r     # (3+6)/302*151 ; (9+6)/2 (2001 excluded)
        assert r["nasa/station"] == 1.67, r
        e = era[era["site"] == "Hisar"].set_index("era")
        assert e.loc["1990s", "station"] == 3.0 and e.loc["1990s", "nasa"] == 9.0, e
        assert e.loc["2000s", "station"] == 6.0 and e.loc["2000s", "nasa"] == 6.0, e
        WARNINGS.clear()
        assert grid_season_counts(t / "nope.csv").empty and any("missing file" in w for w in WARNINGS)
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winter", default="data/raw/india/winter_data.csv.gz")
    ap.add_argument("--grid-dir", default="data/raw/india/tmin",
                    help="searched recursively; any layout with nasa/era5 in the path or name")
    ap.add_argument("--out", default="data/raw/india/station_vs_grid.csv")
    ap.add_argument("--out-era", default="data/raw/india/station_vs_grid_by_era.csv")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)

    winter = pd.read_csv(a.winter, usecols=["STATION", "season", "derived_min", "le_4C"],
                         dtype={"STATION": str})
    pooled, era = compare(winter, a.grid_dir)
    pd.set_option("display.width", 220)
    print(f"Cold nights per Nov-Mar season (Tmin <= {THR:g} C), SEASON-MATCHED, "
          f"station = explicit-minimum days only\n")
    print(pooled.to_string(index=False))
    print("\nBy era (same matched seasons; a gridded series should not jump where a station does):\n")
    print(era.to_string(index=False))
    pooled.to_csv(a.out, index=False)
    era.to_csv(a.out_era, index=False)
    print(f"\nsaved -> {a.out} and {a.out_era}\n\ngridded files used:")
    for site, src, f in USED:
        print(f"   {site:9s} {src:5s} {f}")
    if WARNINGS:
        print("\nWARNING: NaN above is NOT a result for these:")
        for w in sorted(set(WARNINGS)):
            print("   ", w)
        sys.exit(1)