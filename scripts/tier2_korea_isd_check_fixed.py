#!/usr/bin/env python3
"""
Tier 2 feasibility check: test NOAA Global Hourly / ISD present-weather data
for South Korean stations.

Install:
    pip install pandas requests

Examples:
    python tier2_korea_isd_check.py
    python tier2_korea_isd_check.py --years 2008 2012 2016 --months 3 12 1 2
    python tier2_korea_isd_check.py --no-bbox --max-stations 10
    python tier2_korea_isd_check.py --selftest

NOAA retired its old NCEI HTTPS/FTP ISD download service in July 2026.
This script uses the NOAA Open Data Dissemination (NODD) S3 bucket for
station-year CSVs and validates responses before passing them to pandas.

Caveats:
- Candidate frost codes {48, 49, 76} are unverified; check NOAA's format docs.
- ISD present-weather is reported at observation times, not necessarily
  evidence that frost occurred overnight.
- ISD/Global Hourly has been superseded and NOAA notes that this product is
  no longer updated beyond 2025-08-24. Consider GHCNh for newer data.
"""
import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# The old history URL is retained as a first choice; the NODD bucket is fallback.
HISTORY_URLS = (
    "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
    "https://noaa-global-hourly-pds.s3.amazonaws.com/isd-history.csv",
)
# NOAA's NODD ISD CSV bucket (replacement for the retired NCEI WAF path).
DATA_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{sid}.csv"

DEFAULT_BBOX = (36.8, 38.2, 126.4, 127.9)  # lat_min, lat_max, lon_min, lon_max
CANDIDATE_FROST_CODES = {48, 49, 76}  # UNVERIFIED; confirm in NOAA format docs
CODE_PREFIXES = ("MW", "AW")
PAST_PREFIXES = ("AY", "AZ")
FROST_REGEX = r"frost|rime|hoar"
SUMMARY_COLUMNS = [
    "station", "name", "year", "n_obs", "n_days", "code_columns",
    "wx_nonnull", "past_wx_nonnull", "candidate_frost_hits",
    "rem_frost_hits", "top_codes",
]
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "frost-model-research/1.0 (research data validation)"})


def read_csv_response(url, *, dtype=str, timeout=45):
    """Download CSV safely; reject HTTP errors, HTML/error pages, and empty bodies."""
    response = SESSION.get(url, timeout=timeout)
    response.raise_for_status()
    body = response.content
    if not body or not body.strip():
        raise ValueError("server returned an empty response")
    start = body[:500].lstrip().lower()
    if start.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        excerpt = response.text[:180].replace("\n", " ")
        raise ValueError(f"server returned HTML instead of CSV: {excerpt}")
    try:
        frame = pd.read_csv(io.BytesIO(body), dtype=dtype, low_memory=False)
    except Exception as exc:
        excerpt = response.text[:180].replace("\n", " ")
        raise ValueError(f"response is not a valid CSV ({exc}); body starts: {excerpt}") from exc
    if frame.empty and len(frame.columns) <= 1:
        raise ValueError("response did not contain a recognizable CSV table")
    return frame


def load_korean_stations(begin_max=20080101, end_min=20250101, bbox=DEFAULT_BBOX):
    last_error = None
    st = None
    for url in HISTORY_URLS:
        try:
            st = read_csv_response(url)
            break
        except Exception as exc:
            last_error = exc
            print(f"Station-list source failed: {url}\n  -> {type(exc).__name__}: {exc}")
    if st is None:
        raise RuntimeError(f"Could not download NOAA station history from any source: {last_error}")

    required = {"USAF", "WBAN", "CTRY", "BEGIN", "END", "LAT", "LON"}
    missing = required - set(st.columns)
    if missing:
        raise ValueError(f"Station history is missing expected columns: {sorted(missing)}")

    st["USAF"] = st["USAF"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    st["WBAN"] = st["WBAN"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    ks = st[st["CTRY"].astype(str).str.strip() == "KS"].copy()
    ks["BEGIN"] = pd.to_numeric(ks["BEGIN"], errors="coerce")
    ks["END"] = pd.to_numeric(ks["END"], errors="coerce")
    ks["LAT"] = pd.to_numeric(ks["LAT"], errors="coerce")
    ks["LON"] = pd.to_numeric(ks["LON"], errors="coerce")
    ks = ks[(ks["BEGIN"] <= begin_max) & (ks["END"] >= end_min)]
    if bbox:
        lat0, lat1, lon0, lon1 = bbox
        ks = ks[ks["LAT"].between(lat0, lat1) & ks["LON"].between(lon0, lon1)]
    ks["SID"] = ks["USAF"].str.zfill(6) + ks["WBAN"].str.zfill(5)
    return ks.reset_index(drop=True)


def fetch_station_year(sid, year, retries=2, pause=2.0):
    url = DATA_URL.format(year=year, sid=sid)
    for attempt in range(retries + 1):
        try:
            df = read_csv_response(url)
            if "DATE" not in df.columns:
                raise ValueError(f"CSV has no DATE column; columns: {list(df.columns)[:12]}")
            return df
        except Exception as exc:
            if attempt == retries:
                print(f"    download failed: {url}\n      -> {type(exc).__name__}: {exc}")
                return None
            time.sleep(pause * (attempt + 1))
    return None


def first_code(series):
    """ISD groups look like '02,1' (code,quality); return numeric code."""
    return pd.to_numeric(series.astype("string").str.split(",").str[0], errors="coerce")


def analyze(df, months):
    """Return (summary_dict, sample_rows_df) for one station-year dataframe."""
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    sub = df[df["DATE"].dt.month.isin(months)].copy()
    code_cols = [c for c in sub.columns if c.startswith(CODE_PREFIXES)]
    past_cols = [c for c in sub.columns if c.startswith(PAST_PREFIXES)]
    summary = {
        "n_obs": len(sub),
        "n_days": sub["DATE"].dt.date.nunique() if len(sub) else 0,
        "code_columns": ",".join(code_cols),
        "wx_nonnull": 0,
        "past_wx_nonnull": int(sum(sub[c].notna().sum() for c in past_cols)),
        "candidate_frost_hits": 0,
        "rem_frost_hits": 0,
        "top_codes": "",
    }
    all_codes = []
    for col in code_cols:
        codes = first_code(sub[col]).dropna().astype(int)
        summary["wx_nonnull"] += len(codes)
        all_codes.append(codes)
    if all_codes:
        codes = pd.concat(all_codes, ignore_index=True)
        summary["candidate_frost_hits"] = int(codes.isin(CANDIDATE_FROST_CODES).sum())
        summary["top_codes"] = str(codes.value_counts().head(8).to_dict())

    sample_mask = pd.Series(False, index=sub.index)
    for col in code_cols:
        sample_mask |= sub[col].notna()
    if "REM" in sub.columns:
        rem_hit = sub["REM"].astype("string").str.contains(FROST_REGEX, case=False, regex=True, na=False)
        summary["rem_frost_hits"] = int(rem_hit.sum())
        sample_mask |= rem_hit

    keep = [c for c in ["DATE", "NAME"] + code_cols + (["REM"] if "REM" in sub.columns else [])
            if c in sub.columns]
    return summary, sub.loc[sample_mask, keep]


def selftest():
    rows = pd.DataFrame({
        "DATE": ["2015-03-01T00:00:00", "2015-03-01T03:00:00",
                 "2015-03-02T00:00:00", "2015-04-01T00:00:00"],
        "NAME": ["TEST"] * 4,
        "MW1": ["48,1", "02,1", None, "48,1"],
        "AW1": [None, None, "10,1", None],
        "AY1": [None, "3,1,06,1", None, None],
        "REM": ["SYN FROST", "SYN", "SYN", "SYN"],
    })
    summary, sample = analyze(rows, months=[3])
    assert summary["n_obs"] == 3, summary
    assert summary["wx_nonnull"] == 3, summary
    assert summary["candidate_frost_hits"] == 1, summary
    assert summary["rem_frost_hits"] == 1, summary
    assert summary["past_wx_nonnull"] == 1, summary
    assert len(sample) == 3, sample
    empty_codes = rows.drop(columns=["MW1", "AW1"])
    summary2, _ = analyze(empty_codes, months=[3])
    assert summary2["wx_nonnull"] == 0 and summary2["code_columns"] == "", summary2

    # Ensure the empty-results verdict path never indexes missing columns.
    empty_results = pd.DataFrame(columns=SUMMARY_COLUMNS)
    verdict(empty_results)
    print("selftest passed")


def verdict(results):
    # Always normalize columns so an empty DataFrame is safe to inspect.
    results = results.reindex(columns=SUMMARY_COLUMNS)
    print("\n" + "=" * 70)
    if results.empty:
        print("NO DATA: no station-year CSVs were successfully downloaded.")
        print("Check the NODD URL, internet access, and station/year availability.")
    else:
        wx = pd.to_numeric(results["wx_nonnull"], errors="coerce").fillna(0)
        candidate = pd.to_numeric(results["candidate_frost_hits"], errors="coerce").fillna(0)
        remarks = pd.to_numeric(results["rem_frost_hits"], errors="coerce").fillna(0)
        populated = results[wx > 0]
        hits = results[(candidate > 0) | (remarks > 0)]
        if hits.empty and populated.empty:
            print("NO-GO: present-weather columns are empty or absent for every downloaded station-year.")
        elif hits.empty:
            print("NO-GO (provisional): present weather exists, but no candidate frost/rime codes or remarks were found.")
            print("Inspect top_codes and verify codes against NOAA's ISD format documentation.")
        else:
            print("POSSIBLE GO: candidate frost/rime codes or remarks found at:")
            print(hits[["station", "name", "year", "candidate_frost_hits", "rem_frost_hits"]].to_string(index=False))
            print("Verify codes against NOAA's ISD format documentation before drawing conclusions.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2010, 2015])
    parser.add_argument("--months", type=int, nargs="+", default=[3],
                        help="Noh et al. used March; add 12 1 2 for the wider season")
    parser.add_argument("--max-stations", type=int, default=6)
    parser.add_argument("--no-bbox", action="store_true",
                        help="ignore the Gapyeong/Namyang box; use any active KS station")
    parser.add_argument("--out-dir", default="tier2_korea_check")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if any(month < 1 or month > 12 for month in args.months):
        parser.error("--months values must be between 1 and 12")
    if args.max_stations < 1:
        parser.error("--max-stations must be at least 1")
    if args.selftest:
        selftest()
        return 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        stations = load_korean_stations(bbox=None if args.no_bbox else DEFAULT_BBOX)
    except Exception as exc:
        print(f"Could not load station history: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"{len(stations)} candidate Korean stations after filtering")
    if stations.empty:
        print("None matched; try --no-bbox.")
        return 1
    stations = stations.head(args.max_stations)
    display_cols = ["SID", "STATION NAME", "LAT", "LON", "BEGIN", "END"]
    print(stations[display_cols].to_string(index=False))

    rows = []
    for _, station in stations.iterrows():
        for year in args.years:
            print(f"\n-> {station['SID']} {station['STATION NAME']} {year}")
            df = fetch_station_year(str(station["SID"]), year)
            if df is None:
                continue
            summary, sample = analyze(df, args.months)
            summary.update(station=str(station["SID"]), name=station["STATION NAME"], year=year)
            rows.append(summary)
            print(f"    obs={summary['n_obs']} wx_nonnull={summary['wx_nonnull']} "
                  f"candidate_frost={summary['candidate_frost_hits']} "
                  f"rem_frost={summary['rem_frost_hits']} codes={summary['top_codes'] or '-'}")
            if not sample.empty:
                sample.to_csv(out / f"sample_{station['SID']}_{year}.csv", index=False)

    results = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = out / "summary.csv"
    results.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}" + (" and per-station sample files" if rows else " (header-only: no successful downloads)"))
    verdict(results)
    return 0 if not results.empty else 2


if __name__ == "__main__":
    sys.exit(main())
