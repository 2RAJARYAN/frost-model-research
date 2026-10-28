#!/usr/bin/env python3
"""
NOAA GSOD station pipeline for the frost-day dataset (India, Tier 3).

Run from the repository root (needs internet for steps 1-2):
    python -m pip install pandas numpy requests

    python scripts/gsod_download.py stations            # choose stations -> data/raw/india/stations_selected.csv
    python scripts/gsod_download.py download            # fetch GSOD station-year CSVs (resumable)
    python scripts/gsod_download.py summarize           # per-station summary + winter_data.csv.gz
    python scripts/gsod_download.py selftest            # offline known-answer checks, no network

WHAT CHANGED vs. the first pilot version (and why)
  * Thresholds are now 3, 4 and 5 C (primary = 4 C, matching IMD's plains Cold
    Wave criterion and Attri et al.'s frost-day definition). Counts use <=.
  * F->C conversion is ROUNDED. GSOD is in Fahrenheit, and 39.2 F converts to
    4.000000000000002 C, so a night at exactly 4.0 C failed "<= 4" before.
    Indian stations report to 0.1 C, so exact-threshold nights are common.
  * Region default is now an Indo-Gangetic-plains box capped at 500 m
    (previously 700 m admitted Dehradun, a 682 m foothill valley where IMD's
    plains criteria do not apply). A PRIORITY list always includes the anchor,
    core, secondary and negative-control stations, even if they fall outside
    the box (Nagpur does) or would be skipped by --every.
  * Station keep-rule no longer uses coverage over all 35 seasons (that dropped
    stations with short but good records, and Ganganagar). It now uses
    seasons_with_data, coverage over active seasons, AND label quality
    (share of Tmin values derived from synoptic readings). Failures are
    explained in keep_reason, not silently dropped.
  * New columns: winters with >=3 nights per threshold (the feasibility rule
    from the earlier handoff), nights per season, explicit-minimum-only counts,
    and a role (event_station / low_event / negative_control / excluded).
  * The "strict" (Omazic-style) label now uses Td computed from Tmin and the
    daily-mean RH, as in the paper, instead of GSOD's daily-MEAN dew point.
    RHmean is itself derived from GSOD mean temperature and mean dew point, so
    this is still an approximation.
  * winter_data.csv.gz now keeps the other daily variables (tmax, pressure,
    wind, precip, visibility, FRSHTT) so the modelling table can be built
    from it without re-reading 3000+ files.

CAVEATS
  * pct_min_derived assumes GSOD's MIN_ATTRIBUTES "*" means the minimum was
    derived from the available observations rather than an explicit minimum
    report. Derived minima are biased warm and undercount cold nights.
  * Nothing here validates labels against observed frost or crop damage.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# Single source of truth for the Magnus dew-point formula (repo: src/features.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import dew_point_from_rh  # noqa: E402

# ----------------------------- CONFIG ---------------------------------
START_YEAR = 1990
END_YEAR = 2025          # seasons START_YEAR .. END_YEAR-1 (Nov-Mar)
MIN_YEARS = 15           # ISD-listed span within the period (cheap pre-filter)

REGIONS = {
    # Indo-Gangetic plains + Rajasthan/MP/UP; <=500 m keeps hill/foothill stations out
    "ig_plains": dict(lat=(24.0, 32.5), lon=(69.0, 88.0), elev=500),
    "all_india": dict(lat=(18.0, 35.0), lon=(68.0, 95.0), elev=1500),
}
DEFAULT_REGION = "ig_plains"

# Always included (by 11-digit station id = USAF + WBAN). Edit freely.
PRIORITY_SIDS = {
    "42123099999": "ANCHOR: Sri Ganganagar, ~37 km from Abohar (primary Kinnow site)",
    "42131099999": "core: Hisar",
    "42101099999": "core: Patiala",
    "42071099999": "core, FLAGGED: Amritsar (outlier vs gridded data and Attri; run with and without)",
    "42182099999": "secondary: Delhi Safdarjung (do not also count IGI: same city)",
    "42165099999": "secondary: Bikaner",
    "42348099999": "secondary: Jaipur",
    "42361099999": "secondary: Gwalior",
    "42369099999": "secondary: Lucknow",
    "42492099999": "negative control: Patna",
    "42452099999": "negative control: Kota",
    "42379099999": "negative control: Gorakhpur",
    "42867099999": "negative control: Nagpur (Vidarbha, original deployment region)",
}
NEGATIVE_CONTROL_SIDS = {s for s, d in PRIORITY_SIDS.items() if d.startswith("negative control")}

# Waive ONLY the label-quality test for these (coverage test still applies).
# Ganganagar is the closest real station to Abohar but ~56% of its Tmin is derived.
# Set to set() to apply the quality rule to everyone.
ANCHOR_EXCEPTIONS = {"42123099999"}

WINTER_MONTHS = [11, 12, 1, 2, 3]
DAYS_PER_SEASON = 151
THRESHOLDS_C = [3, 4, 5]
PRIMARY_THRESHOLD_C = 4

MIN_SEASONS = 20               # seasons with any data
MIN_COVERAGE_ACTIVE = 0.75     # valid winter days / (seasons_with_data * 151)
MAX_PCT_MIN_DERIVED = 50.0     # above this, Tmin is mostly warm-biased
MIN_EVENT_WINTERS = 10         # winters with >=3 nights at the primary threshold

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
GSOD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{sid}.csv"
STATIONS_FILE = "data/raw/india/stations_selected.csv"
WINTER_FILE = "data/raw/india/winter_data.csv.gz"

WANTED_COLS = {"STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME",
               "TEMP", "DEWP", "MAX", "MIN", "MIN_ATTRIBUTES",
               "SLP", "STP", "VISIB", "WDSP", "PRCP", "FRSHTT"}
# GSOD "missing" sentinels
# NOTE: GSOD's missing code for STP is 999.9 (NOT 9999.9). With the wrong code, STP stayed a
# constant 999.9 in winter_data.csv.gz and silently produced all-zero pressure features.
MISSING = {"TEMP": 9999.9, "DEWP": 9999.9, "MAX": 9999.9, "MIN": 9999.9,
           "SLP": 9999.9, "STP": 999.9, "VISIB": 999.9, "WDSP": 999.9, "PRCP": 99.99}
# Must match src/features.py::dew_point_from_rh (checked in selftest)
_MAGNUS_A, _MAGNUS_B = 17.27, 237.7
# ----------------------------------------------------------------------

N_SEASONS = END_YEAR - START_YEAR


# --------------------------- helpers ----------------------------------
def f_to_c(f):
    """Fahrenheit -> Celsius, rounded to 2 dp so exact thresholds survive float error."""
    return ((f - 32.0) * 5.0 / 9.0).round(2)


def rh_from_dew_point(temp_c, dewp_c):
    """Relative humidity (%) from temperature and dew point (Magnus)."""
    a, b = _MAGNUS_A, _MAGNUS_B
    rh = 100.0 * np.exp(a * dewp_c / (b + dewp_c) - a * temp_c / (b + temp_c))
    return np.clip(rh, 0.0, 100.0)


def select_pilot(st, every):
    """Priority stations are always kept; --every N thins only the rest."""
    if every <= 1:
        return st
    pri = st["sid"].isin(PRIORITY_SIDS)
    return pd.concat([st[pri], st[~pri].iloc[::every]]).sort_values("LAT", ascending=False)


# --------------------------- step 1 -----------------------------------
def cmd_stations(args):
    """Select Indian stations from NOAA's ISD station list."""
    r = REGIONS[args.region]
    lat, lon, elev = r["lat"], r["lon"], r["elev"]
    print(f"region '{args.region}': lat {lat[0]}-{lat[1]}, lon {lon[0]}-{lon[1]}, "
          f"elevation <= {elev} m, >= {MIN_YEARS} years in {START_YEAR}-{END_YEAR}")
    df = pd.read_csv(args.isd_file or ISD_HISTORY_URL, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["CTRY"] == "IN"].copy()
    for c in ["LAT", "LON", "ELEV(M)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["begin_year"] = pd.to_numeric(df["BEGIN"].str[:4], errors="coerce")
    df["end_year"] = pd.to_numeric(df["END"].str[:4], errors="coerce")
    df = df.dropna(subset=["LAT", "LON", "begin_year", "end_year"]).copy()
    df["begin_year"] = df["begin_year"].astype(int)
    df["end_year"] = df["end_year"].astype(int)
    df["sid"] = df["USAF"].str.zfill(6) + df["WBAN"].str.zfill(5)

    years_in_period = df["end_year"].clip(upper=END_YEAR) - df["begin_year"].clip(lower=START_YEAR) + 1
    in_region = (
        df["LAT"].between(*lat)
        & df["LON"].between(*lon)
        & (df["ELEV(M)"].where(df["ELEV(M)"] > -900, 0).fillna(0) <= elev)
        & (years_in_period >= MIN_YEARS)
    )
    sel = df[in_region | df["sid"].isin(PRIORITY_SIDS)].copy()
    sel["note"] = sel["sid"].map(PRIORITY_SIDS).fillna("")
    sel = sel.sort_values("LAT", ascending=False)

    missing = [f"{s} ({d})" for s, d in PRIORITY_SIDS.items() if s not in set(df["sid"])]
    out = sel[["sid", "STATION NAME", "STATE", "LAT", "LON", "ELEV(M)", "begin_year", "end_year", "note"]]
    Path(STATIONS_FILE).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATIONS_FILE, index=False)
    print(f"{len(sel)} Indian stations selected -> {STATIONS_FILE}")
    print(out.drop(columns=["note"]).to_string(index=False))
    if missing:
        print("\nWARNING: priority stations NOT found in the ISD station list "
              "(check the id before relying on them):")
        for m in missing:
            print("   ", m)


# --------------------------- step 2 -----------------------------------
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
    """Download one CSV per station-year. Safe to stop and re-run: existing files are skipped."""
    st = pd.read_csv(STATIONS_FILE, dtype={"sid": str})
    if args.every > 1:
        st = select_pilot(st, args.every)
        print(f"pilot mode: priority stations + every {args.every}th other -> {len(st)} stations")
    out = Path(args.gsod_dir)
    jobs, skipped = [], 0
    for _, row in st.iterrows():
        for year in range(max(START_YEAR, int(row["begin_year"])), min(END_YEAR, int(row["end_year"])) + 1):
            dest = out / str(year) / f"{row['sid']}.csv"
            if dest.exists():
                skipped += 1
            else:
                jobs.append((GSOD_URL.format(year=year, sid=row["sid"]), dest))
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
    print(f"finished. failed/missing: {len(missing)} (see download_missing.txt; "
          "HTTP 404 for a year a station did not report is normal)")


# --------------------------- step 3 -----------------------------------
def load_winter(gsod_dir):
    """Read all yearly files, keep only winter rows and the needed columns."""
    files = sorted(Path(gsod_dir).glob("*/*.csv"))
    if Path(STATIONS_FILE).exists():
        keep = set(pd.read_csv(STATIONS_FILE, dtype={"sid": str})["sid"])
        files = [f for f in files if f.stem in keep]
    if not files:
        sys.exit(f"no CSV files found under {gsod_dir}/<year>/")
    parts, raw_rows = [], 0
    for f in files:
        d = pd.read_csv(f, usecols=lambda c: c in WANTED_COLS,
                        dtype={"STATION": str, "MIN_ATTRIBUTES": str, "FRSHTT": str})
        raw_rows += len(d)
        d["DATE"] = pd.to_datetime(d["DATE"])
        d = d[d["DATE"].dt.month.isin(WINTER_MONTHS)]
        if len(d):
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["season"] = df["DATE"].dt.year.where(df["DATE"].dt.month >= 11, df["DATE"].dt.year - 1)
    df = df[(df["season"] >= START_YEAR) & (df["season"] < END_YEAR)].copy()
    for col, bad in MISSING.items():
        if col in df.columns:
            df[col] = df[col].mask(df[col] == bad)
    print(f"read {len(files)} files: {raw_rows:,} rows -> {len(df):,} winter rows kept")
    return df


def build_daily(df):
    """Add Celsius columns, derived flags, and threshold labels to winter rows."""
    df = df[df["MIN"].notna()].copy()
    df["tmin_c"] = f_to_c(df["MIN"])
    df["tmax_c"] = f_to_c(df["MAX"]) if "MAX" in df else np.nan
    df["tmean_c"] = f_to_c(df["TEMP"])
    df["dewp_c"] = f_to_c(df["DEWP"])
    df["rh_mean_pct"] = rh_from_dew_point(df["tmean_c"], df["dewp_c"])
    # Omazic-style Td: from Tmin and daily-mean RH (approximation, see docstring)
    df["td_paper_c"] = dew_point_from_rh(df["tmin_c"], df["rh_mean_pct"])
    df["derived_min"] = df["MIN_ATTRIBUTES"].astype(str).str.strip() == "*"
    df["explicit_min"] = ~df["derived_min"]
    if "WDSP" in df:
        df["wdsp_ms"] = df["WDSP"] * 0.514444   # knots -> m/s
    for t in THRESHOLDS_C:
        df[f"le_{t}C"] = df["tmin_c"] <= t
        df[f"strict_le_{t}C"] = df[f"le_{t}C"] & (df["td_paper_c"] < 0)
    p = PRIMARY_THRESHOLD_C
    df[f"explicit_le_{p}C"] = df[f"le_{p}C"] & df["explicit_min"]
    return df


def station_summary(df):
    p = PRIMARY_THRESHOLD_C
    g = df.groupby("STATION")
    summ = g.agg(
        name=("NAME", "first"), lat=("LATITUDE", "first"), lon=("LONGITUDE", "first"),
        elev_m=("ELEVATION", "first"),
        winter_days_valid=("tmin_c", "size"), seasons_with_data=("season", "nunique"),
        pct_min_derived=("derived_min", "mean"), explicit_winter_days=("explicit_min", "sum"),
        **{f"nights_le_{t}C": (f"le_{t}C", "sum") for t in THRESHOLDS_C},
        **{f"nights_strict_le_{t}C": (f"strict_le_{t}C", "sum") for t in THRESHOLDS_C},
        **{f"nights_explicit_le_{p}C": (f"explicit_le_{p}C", "sum")},
    ).reset_index()

    per_season = df.groupby(["STATION", "season"])[[f"le_{t}C" for t in THRESHOLDS_C]].sum()
    for t in THRESHOLDS_C:
        w3 = (per_season[f"le_{t}C"] >= 3).groupby(level=0).sum().rename(f"winters_ge3_le_{t}C")
        summ = summ.merge(w3, left_on="STATION", right_index=True, how="left")

    summ["coverage"] = (summ["winter_days_valid"] / (N_SEASONS * DAYS_PER_SEASON)).round(2)
    summ["coverage_active"] = (summ["winter_days_valid"] / (summ["seasons_with_data"] * DAYS_PER_SEASON)).round(2)
    summ["pct_min_derived"] = (summ["pct_min_derived"] * 100).round(1)
    summ[f"per_season_le_{p}C"] = (summ[f"nights_le_{p}C"] / summ["seasons_with_data"]).round(2)

    summ["keep_coverage"] = (summ["seasons_with_data"] >= MIN_SEASONS) & \
                            (summ["coverage_active"] >= MIN_COVERAGE_ACTIVE)
    summ["keep_quality"] = summ["pct_min_derived"] <= MAX_PCT_MIN_DERIVED
    waived = summ["STATION"].isin(ANCHOR_EXCEPTIONS) & summ["keep_coverage"] & ~summ["keep_quality"]
    summ["keep"] = (summ["keep_coverage"] & summ["keep_quality"]) | waived
    summ["keep_reason"] = np.select(
        [waived, summ["keep"], ~summ["keep_coverage"]],
        ["anchor exception (derived-min share waived)", "ok", "short record / low coverage"],
        default="Tmin mostly derived (warm-biased)")
    summ["role"] = np.select(
        [summ["STATION"].isin(NEGATIVE_CONTROL_SIDS),
         summ["keep"] & (summ[f"winters_ge3_le_{p}C"] >= MIN_EVENT_WINTERS),
         summ["keep"]],
        ["negative_control", "event_station", "low_event"], default="excluded")
    summ["note"] = summ["STATION"].map(PRIORITY_SIDS).fillna("")
    return summ.sort_values(f"nights_le_{p}C", ascending=False)


def cmd_summarize(args):
    """Per-station winter event counts + daily winter table for modelling."""
    df = build_daily(load_winter(args.gsod_dir))
    Path(WINTER_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WINTER_FILE, index=False)
    summ = station_summary(df)
    summ.to_csv(args.out, index=False)

    p = PRIMARY_THRESHOLD_C
    show = ["STATION", "name", "elev_m", "seasons_with_data", "coverage_active", "pct_min_derived",
            "nights_le_3C", f"nights_le_{p}C", "nights_le_5C", f"winters_ge3_le_{p}C",
            f"per_season_le_{p}C", f"nights_strict_le_{p}C", "role", "keep_reason"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 50)
    print(summ[show].to_string(index=False))
    print("\nroles:", summ["role"].value_counts().to_dict())
    kept = summ[summ["keep"]]
    for t in THRESHOLDS_C:
        print(f"  kept stations, total nights Tmin <= {t} C: {int(kept[f'nights_le_{t}C'].sum())}")
    top = kept[f"nights_le_{p}C"].max() / max(kept[f"nights_le_{p}C"].sum(), 1)
    print(f"  largest single-station share of kept <= {p} C nights: {top:.0%} "
          "(if this is large, one station dominates a pooled model)")
    print(f"saved -> {args.out} and {WINTER_FILE}")


# --------------------------- selftest ---------------------------------
def _selftest():
    import argparse as ap
    import tempfile
    global STATIONS_FILE, WINTER_FILE, MIN_SEASONS, MIN_COVERAGE_ACTIVE

    # 1. float boundary: exact thresholds survive conversion
    assert f_to_c(pd.Series([37.4, 39.2, 41.0])).tolist() == [3.0, 4.0, 5.0]

    # 2. Magnus constants here match src/features.py (round trip)
    T, Td = pd.Series([4.4, 10.0, 15.6]), pd.Series([3.9, 7.2, -6.7])
    rt = dew_point_from_rh(T, rh_from_dew_point(T, Td))
    assert np.allclose(rt, Td, atol=1e-6), rt

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        STATIONS_FILE = str(tmp / "stations_selected.csv")
        WINTER_FILE = str(tmp / "winter.csv.gz")
        MIN_SEASONS, MIN_COVERAGE_ACTIVE = 1, 0.5

        # 3. cmd_stations: region + priority + elevation + short-record filters
        hist = pd.DataFrame([
            # USAF, WBAN, name, ctry, state, lat, lon, elev, begin, end
            ["421310", "99999", "HISSAR", "IN", "", 29.18, 75.76, 213, "19900101", "20250824"],
            ["428670", "99999", "NAGPUR", "IN", "", 21.09, 79.05, 310, "19900101", "20250824"],
            ["421110", "99999", "DEHRADUN", "IN", "", 30.32, 78.03, 682, "19900101", "20250824"],
            ["421999", "99999", "SHORT", "IN", "", 29.00, 76.00, 200, "20180101", "20250824"],
            ["470880", "99999", "SANGJU", "KS", "", 36.40, 128.28, 100, "19900101", "20250824"],
        ], columns=["USAF", "WBAN", "STATION NAME", "CTRY", "STATE", "LAT", "LON", "ELEV(M)", "BEGIN", "END"])
        hist.to_csv(tmp / "isd.csv", index=False)
        cmd_stations(ap.Namespace(region="ig_plains", isd_file=str(tmp / "isd.csv")))
        got = set(pd.read_csv(STATIONS_FILE, dtype={"sid": str})["sid"])
        assert got == {"42131099999", "42867099999"}, got

        # 4. pilot thinning never drops priority stations
        st = pd.DataFrame({"sid": ["42123099999"] + [f"0000{i:07d}" for i in range(20)],
                           "LAT": np.linspace(30, 25, 21)})
        assert "42123099999" in set(select_pilot(st, 8)["sid"])
        assert len(select_pilot(st, 8)) == 1 + len(range(0, 20, 8))

        # 5. summarize: known-answer season for one synthetic station
        sid = "99999099999"
        pd.DataFrame({"sid": [sid], "STATION NAME": ["T"], "begin_year": [2000], "end_year": [2001]}) \
            .to_csv(STATIONS_FILE, index=False)
        days = pd.date_range("2000-11-01", "2001-03-31", freq="D")
        assert len(days) == 151
        d = pd.DataFrame({"STATION": sid, "DATE": days, "LATITUDE": 29.0, "LONGITUDE": 75.0,
                          "ELEVATION": 200.0, "NAME": "TEST, IN",
                          "TEMP": 65.0, "TEMP_ATTRIBUTES": "24", "DEWP": 55.0, "MAX": 75.0,
                          "MIN": 59.0, "MIN_ATTRIBUTES": " ", "SLP": 1015.0, "STP": 990.0,
                          "VISIB": 6.0, "WDSP": 3.0, "PRCP": 0.0, "FRSHTT": "000000"})
        def setrow(i, **kw):
            for k, v in kw.items():
                d.loc[i, k] = v
        for i in (0, 1):                                  # cold and DRY
            setrow(i, MIN=33.8, TEMP=60.0, DEWP=20.0)
        for i in (2, 3):                                  # cold and HUMID
            setrow(i, MIN=33.8, TEMP=40.0, DEWP=39.0)
        setrow(4, MIN=33.8, MIN_ATTRIBUTES="*", TEMP=40.0, DEWP=39.0)   # derived min
        setrow(5, MIN=39.2, TEMP=50.0, DEWP=45.0)         # exactly 4.0 C
        setrow(6, MIN=9999.9)                             # missing -> dropped
        for year in (2000, 2001):
            (tmp / "gsod" / str(year)).mkdir(parents=True, exist_ok=True)
            d[d["DATE"].dt.year == year].to_csv(tmp / "gsod" / str(year) / f"{sid}.csv", index=False)

        cmd_summarize(ap.Namespace(gsod_dir=str(tmp / "gsod"), out=str(tmp / "summary.csv")))
        s = pd.read_csv(tmp / "summary.csv").iloc[0]
        checks = {
            "winter_days_valid": 150, "seasons_with_data": 1,
            "nights_le_3C": 5, "nights_le_4C": 6, "nights_le_5C": 6,   # 4.0 C counted at <=4
            "winters_ge3_le_3C": 1, "winters_ge3_le_4C": 1,
            "nights_explicit_le_4C": 5, "explicit_winter_days": 149,
            "nights_strict_le_3C": 2, "nights_strict_le_4C": 2,        # only the DRY cold nights
        }
        for k, v in checks.items():
            assert s[k] == v, f"{k}: expected {v}, got {s[k]}"
        assert s["pct_min_derived"] == 0.7, s["pct_min_derived"]
        assert s["role"] in ("low_event", "event_station", "excluded"), s["role"]
    print("selftest passed")


# --------------------------- CLI --------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("stations")
    sp.add_argument("--region", choices=list(REGIONS), default=DEFAULT_REGION)
    sp.add_argument("--isd-file", help="local isd-history.csv instead of downloading it")
    sp.set_defaults(fn=cmd_stations)
    d = sub.add_parser("download")
    d.add_argument("--gsod-dir", default="data/raw/india/gsod")
    d.add_argument("--every", type=int, default=1, help="pilot: thin NON-priority stations to every Nth")
    d.add_argument("--workers", type=int, default=4, help="parallel workers")
    d.set_defaults(fn=cmd_download)
    s = sub.add_parser("summarize")
    s.add_argument("--gsod-dir", default="data/raw/india/gsod")
    s.add_argument("--out", default="data/raw/india/station_summary.csv")
    s.set_defaults(fn=cmd_summarize)
    t = sub.add_parser("selftest")
    t.set_defaults(fn=lambda a: _selftest())
    a = p.parse_args()
    a.fn(a)
    