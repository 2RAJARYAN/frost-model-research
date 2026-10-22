#!/usr/bin/env python3
"""
NOAA GSOD pilot for the frost-day dataset (India, Tier 3).

Run this on your computer (requires internet access).

Setup (once):
    python -m pip install --quiet pandas requests

Steps (run in order, in the repository root):
    python scripts/gsod_download.py stations   # choose Indian stations -> stations_selected.csv
    python scripts/gsod_download.py download --every 8   # pilot: every 8th station (~25 stations)
    python scripts/gsod_download.py summarize          # winter-only counts -> station_summary.csv + winter_data.csv.gz

When the pilot looks good, re‑run the download step without "--every" to fetch all stations.

Edit the CONFIG block below to change region, years, thresholds, etc.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# ----------------------------- CONFIG ---------------------------------
START_YEAR = 1990
END_YEAR = 2025
# Expanded region to include Punjab/Haryana plain plus Vidarbha (Warora, Nagpur)
LAT_MIN, LAT_MAX = 18.0, 35.0
LON_MIN, LON_MAX = 68.0, 95.0
MAX_ELEV_M = 1500  # include hill stations up to 1500 m
MIN_YEARS = 15  # keep stations with at least 15 years of record within period
REGIONS = {
    "all_india": dict(lat=(LAT_MIN, LAT_MAX), lon=(LON_MIN, LON_MAX), elev=MAX_ELEV_M),
    "north":      dict(lat=(25.0, 32.5), lon=(69.0, 88.0), elev=700),
    "north_hills": dict(lat=(25.0, 32.5), lon=(69.0, 88.0), elev=1500),
}
WINTER_MONTHS = [11, 12, 1, 2, 3]  # November‑March
DAYS_PER_SEASON = 151  # Approximate days per winter season
# For Tier 3 we focus on the Omazić‑style broad rule (Tmin < 3 °C).  The strict rule
# (Tmin < 3 °C and dew‑point < 0 °C) is computed later as "omazic_approx".
THRESHOLDS_C = [3]  # broad threshold; keep list for compatibility
MIN_COVERAGE = 0.80  # require ≥80 % of winter days covered

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
GSOD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{sid}.csv"
USECOLS = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
           "TEMP", "DEWP", "MIN", "MIN_ATTRIBUTES"]
# ----------------------------------------------------------------------

N_SEASONS = END_YEAR - START_YEAR  # complete seasons between START_YEAR and END_YEAR‑1

def cmd_stations(args):
    """Select Indian stations within the study box from NOAA's ISD list."""
    lat, lon, elev = (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX), MAX_ELEV_M
    if args.region:
        r = REGIONS[args.region]
        lat, lon, elev = r["lat"], r["lon"], r["elev"]
    print(f"region: lat {lat[0]}-{lat[1]}, lon {lon[0]}-{lon[1]}, elevation <= {elev} m, >= {MIN_YEARS} years")
    df = pd.read_csv(args.isd_file or ISD_HISTORY_URL, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["CTRY"] == "IN"].copy()
    for c in ["LAT", "LON", "ELEV(M)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["begin_year"] = df["BEGIN"].str[:4].astype(int)
    df["end_year"] = df["END"].str[:4].astype(int)
    years_in_period = df["end_year"].clip(upper=END_YEAR) - df["begin_year"].clip(lower=START_YEAR) + 1
    sel = df[
        df["LAT"].between(*lat)
        & df["LON"].between(*lon)
        & (df["ELEV(M)"].where(df["ELEV(M)"] > -900, 0).fillna(0) <= elev)
        & (years_in_period >= MIN_YEARS)
    ].copy()
    sel["sid"] = sel["USAF"].str.zfill(6) + sel["WBAN"].str.zfill(5)
    sel = sel.sort_values(["LAT"], ascending=False)
    sel[["sid", "STATION NAME", "STATE", "LAT", "LON", "ELEV(M)", "begin_year", "end_year"]].to_csv(
        "stations_selected.csv", index=False)
    print(f"{len(sel)} Indian stations selected -> stations_selected.csv")
    print(sel[["sid", "STATION NAME", "LAT", "LON", "ELEV(M)", "begin_year", "end_year"]].to_string(index=False))

def _fetch(job, retries=3):
    import requests
    url, dest = job
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(r.content)
                return None
            return f"{url} (HTTP {r.status_code})"
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return f"{url} (network failure)"

def cmd_download(args):
    """Download one CSV per station‑year. Safe to stop and re‑run: existing files are skipped."""
    st = pd.read_csv("stations_selected.csv", dtype={"sid": str})
    if args.every > 1:
        st = st.iloc[:: args.every]
        print(f"pilot mode: every {args.every}th station -> {len(st)} stations")
    out = Path(args.gsod_dir)
    jobs, skipped = [], 0
    for _, row in st.iterrows():
        for year in range(max(START_YEAR, int(row["begin_year"])), min(END_YEAR, int(row["end_year"])) + 1):
            dest = out / str(year) / f"{row['sid']}.csv"
            if dest.exists():
                skipped += 1
            else:
                jobs.append((GSOD_URL.format(year=year, sid=row['sid']), dest))
    print(f"{len(jobs)} files to download, {skipped} already present, {args.workers} workers")
    missing, n = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_fetch, jobs):
            n += 1
            if res:
                missing.append(res)
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} done")
    Path("download_missing.txt").write_text("\n".join(missing))
    print(f"finished. failed/missing: {len(missing)} (see download_missing.txt)")

def f_to_c(f):
    return (f - 32.0) * 5.0 / 9.0

def load_winter(gsod_dir):
    """Read all yearly files, keep only winter rows and needed columns (low‑memory)."""
    files = sorted(Path(gsod_dir).glob("*/*.csv"))
    if Path("stations_selected.csv").exists():
        keep = set(pd.read_csv("stations_selected.csv", dtype={"sid": str})["sid"])
        files = [f for f in files if f.stem in keep]
    if not files:
        sys.exit(f"no CSV files found under {gsod_dir}/<year>/")
    parts, raw_rows = [], 0
    for f in files:
        d = pd.read_csv(f, usecols=USECOLS, dtype={"STATION": str, "MIN_ATTRIBUTES": str})
        raw_rows += len(d)
        d["DATE"] = pd.to_datetime(d["DATE"])
        d = d[d["DATE"].dt.month.isin(WINTER_MONTHS)]
        if len(d):
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["season"] = df["DATE"].dt.year.where(df["DATE"].dt.month >= 11, df["DATE"].dt.year - 1)
    df = df[(df["season"] >= START_YEAR) & (df["season"] < END_YEAR)]
    print(f"read {len(files)} files: {raw_rows:,} rows → {len(df):,} winter rows kept")
    return df

def cmd_summarize(args):
    """Winter‑only frost‑risk night counts per station and per threshold."""
    df = load_winter(args.gsod_dir)
    df = df[df["MIN"] != 9999.9].copy()
    df["tmin_c"] = f_to_c(df["MIN"])
    df["derived_min"] = df["MIN_ATTRIBUTES"].astype(str).str.strip() == "*"
    df["dewp_c"] = f_to_c(df["DEWP"].where(df["DEWP"] != 9999.9))
    for t in THRESHOLDS_C:
        df[f"le_{t}C"] = df["tmin_c"] <= t
    df["omazic_approx"] = (df["tmin_c"] < 3) & (df["dewp_c"] < 0)
    df.to_csv("winter_data.csv.gz", index=False)
    g = df.groupby("STATION")
    summ = g.agg(
        name=("NAME", "first"),
        lat=("LATITUDE", "first"),
        lon=("LONGITUDE", "first"),
        elev_m=("ELEVATION", "first"),
        winter_days_valid=("tmin_c", "size"),
        seasons_with_data=("season", "nunique"),
        pct_min_derived=("derived_min", "mean"),
        **{f"nights_le_{t}C": (f"le_{t}C", "sum") for t in THRESHOLDS_C},
        nights_omazic_approx=("omazic_approx", "sum"),
    ).reset_index()
    summ["coverage"] = (summ["winter_days_valid"] / (N_SEASONS * DAYS_PER_SEASON)).round(2)
    summ["coverage_active"] = (summ["winter_days_valid"] / (summ["seasons_with_data"] * DAYS_PER_SEASON)).round(2)
    summ["pct_min_derived"] = (summ["pct_min_derived"] * 100).round(1)
    summ["keep"] = summ["coverage"] >= MIN_COVERAGE
    summ = summ.sort_values("nights_le_3C", ascending=False)
    summ.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    print(summ.to_string(index=False))
    kept = summ[summ["keep"]]
    print(f"\n{len(kept)} of {len(summ)} stations have >= {MIN_COVERAGE:.0%} winter coverage ({START_YEAR}-{END_YEAR})")
    for t in THRESHOLDS_C:
        print(f"  total frost nights Tmin <= {t} C among kept stations: {int(kept[f'nights_le_{t}C'].sum())}")
    print(f"  Omazic‑style (approx) nights: {int(kept['nights_omazic_approx'].sum())}")
    print(f"saved → {args.out} and winter_data.csv.gz")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("stations")
    sp.add_argument("--region", choices=list(REGIONS), help="preset region (default: all_india)")
    sp.add_argument("--isd-file", help="local isd-history.csv instead of downloading it")
    sp.set_defaults(fn=cmd_stations)
    d = sub.add_parser("download")
    d.add_argument("--gsod-dir", default="data/raw/india/gsod")
    d.add_argument("--every", type=int, default=1, help="pilot: use every Nth station only")
    d.add_argument("--workers", type=int, default=4, help="parallel workers")
    d.set_defaults(fn=cmd_download)
    s = sub.add_parser("summarize")
    s.add_argument("--gsod-dir", default="data/raw/india/gsod")
    s.add_argument("--out", default="station_summary.csv")
    s.set_defaults(fn=cmd_summarize)
    a = p.parse_args()
    a.fn(a)
