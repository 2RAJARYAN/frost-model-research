"""
Label feasibility check across Vidarbha and the Indo-Gangetic Plains.

For each data source, builds a daily Tmin series, then for each threshold
counts (Nov-Mar seasons, matching Attri et al. 2020, MAUSAM 71(1), 95-102):
  - positive days and share of season days
  - number of winters with >= 1 positive day
  - winters with >= 3 positive days (rough "usable winter" proxy)

Sources:
  nasa  : NASA POWER daily T2M_MIN (gridded)
  era5  : ERA5 via Open-Meteo archive, daily temperature_2m_min (gridded)
  isd   : NOAA ISD-Lite, hourly -> daily min (station)

HEADLINE RESULT (see reports/india_feasibility_findings.md for the full
write-up): Vidarbha (Warora, Nagpur) has essentially zero cold days in
44 winters at any threshold up to 5C -- it cannot support a frost-type
label. The Indo-Gangetic belt (Punjab/Haryana/Rajasthan) does, at a
Tmin <= 4C threshold. Abohar (Kinnow mandarin citrus belt) is the
proposed primary site.

Threshold convention: uses `<=`, matching IMD's own Cold Wave criterion
(actual Tmin <= 4C) and Attri et al.'s frost-day definition (screen Tmin
<= 4C) -- NOT `<` (an earlier version of this script used `<`, undercounting
every site's true event count against both of those sources).

Run:  pip install requests pandas
      python scripts/india_label_feasibility.py [site1 site2 ...]
      python -m pytest scripts/test_india_label_feasibility.py  # logic-only unit tests
"""
import gzip
import io
import sys

import numpy as np
import pandas as pd
import requests

START_YEAR, END_YEAR = 1981, 2025
THRESHOLDS = [2, 3, 4, 5]

# name: (lat, lon, ISD-Lite id or None). Coordinates approximate - adjust as needed.
# ISD ids are my guesses (WMO id + "0" - 99999); the script reports 404s so you can verify.
# Paper stations (Attri et al. 2020) + Vidarbha + a Kinnow-belt candidate (Abohar).
SITES = {
    "warora":   (20.23, 79.00, None),
    "nagpur":   (21.10, 79.05, "428670-99999"),   # VANP, WMO 42867 -- confirmed usable, 1981-01-01 to 2025-08-24
    "delhi":    (28.61, 77.21, "421820-99999"),   # Safdarjung, WMO 42182 -- VERIFIED (meteostat.net/station/42182, tutiempo.net/climate/ws-421820.html both confirm 42182 = New Delhi/Safdarjung, VIDD, 28.58N 77.20E)
    "jaipur":   (26.82, 75.80, None),
    "hisar":    (29.15, 75.72, None),
    "karnal":   (29.69, 76.99, None),
    "patiala":  (30.34, 76.39, None),
    # FIXED: was "420270-99999" (WMO 42027), which returned a clearly wrong
    # station (mean Tmin 8.9C, min -22.6C -- far too cold for Punjab plains,
    # more consistent with a hill/northern station). Amritsar's actual WMO
    # number is 42071 (see the Rajasansi Wikipedia article's citation of
    # "Amritsar Climate Normals 1971-1990", WMO-Normals/RA-II/IN/42071.TXT).
    # NOT yet re-fetched against the real ISD-Lite file for 420710-99999 --
    # verify the returned data looks like Punjab's climate before trusting it.
    "amritsar": (31.63, 74.87, "420710-99999"),   # WMO 42071 (corrected, not yet re-verified by fetching)
    "abohar":   (30.14, 74.20, None),             # Kinnow belt -- site name not yet confirmed with CCRI/PAU/literature
}


def fetch_nasa(LAT, LON):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    p = dict(parameters="T2M_MIN", community="AG", longitude=LON, latitude=LAT,
             start=f"{START_YEAR}0101", end=f"{END_YEAR}1231", format="JSON")
    r = requests.get(url, params=p, timeout=180)
    r.raise_for_status()
    d = r.json()["properties"]["parameter"]["T2M_MIN"]
    s = pd.Series(d)
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    return s.where(s > -900)


def fetch_era5(LAT, LON):
    url = "https://archive-api.open-meteo.com/v1/archive"
    p = dict(latitude=LAT, longitude=LON, start_date=f"{START_YEAR}-01-01",
             end_date=f"{END_YEAR}-12-31", daily="temperature_2m_min",
             timezone="Asia/Kolkata")
    r = requests.get(url, params=p, timeout=180)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.Series(d["temperature_2m_min"], index=pd.to_datetime(d["time"]))


def fetch_isd(ISD_ID, min_obs_per_day=6):
    """ISD-Lite hourly -> daily min (IST day). Note: synoptic obs miss the true
    minimum, so this biases Tmin HIGH versus an IMD min thermometer."""
    frames, missing = [], []
    for y in range(START_YEAR, END_YEAR + 1):
        url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{ISD_ID}-{y}.gz"
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            missing.append(y)
            continue
        raw = gzip.decompress(r.content).decode()
        df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None,
                         usecols=[0, 1, 2, 3, 4], names=["Y", "M", "D", "H", "T"])
        frames.append(df)
    if missing:
        print(f"[isd] no file for years: {missing}")
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames)
    df = df[df["T"] != -9999]
    ts = pd.to_datetime(dict(year=df.Y, month=df.M, day=df.D, hour=df.H))
    ist = ts + pd.Timedelta(hours=5, minutes=30)
    g = pd.Series((df["T"] / 10.0).values, index=ist.dt.normalize().values).groupby(level=0)
    out = g.min()[g.count() >= min_obs_per_day]
    return out


def count_events(tmin, thresholds=THRESHOLDS):
    """
    Nov-Mar seasons; season id = year in which November falls.

    Uses Tmin <= threshold (inclusive), matching IMD's own Cold Wave
    criterion and Attri et al.'s frost-day definition. An earlier version
    of this function used `<` (strict), which undercounts every site's
    event total against both of those reference conventions -- e.g. a day
    at exactly 4.0C was being excluded from the 4C bucket, when both IMD
    and Attri et al. would include it.
    """
    s = tmin.dropna()
    s = s[s.index.month.isin([11, 12, 1, 2, 3])]
    season = pd.Series(np.where(s.index.month >= 11, s.index.year, s.index.year - 1),
                       index=s.index)
    # drop incomplete seasons (< 120 of 151 days)
    full = season.groupby(season).transform("count") >= 120
    s, season = s[full], season[full]
    rows = []
    for t in thresholds:
        pos = s <= t
        per_w = pos.groupby(season).sum()
        rows.append(dict(
            threshold_C=t,
            seasons=int(per_w.shape[0]),
            pos_days=int(pos.sum()),
            pct_days=round(100 * pos.mean(), 2) if len(pos) else 0,
            winters_ge1=int((per_w >= 1).sum()),
            winters_ge3=int((per_w >= 3).sum()),
            max_days_one_winter=int(per_w.max()) if len(per_w) else 0,
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    chosen = sys.argv[1:] or list(SITES)          # e.g. python script.py nagpur amritsar
    summary = []
    for site in chosen:
        lat, lon, isd_id = SITES[site]
        sources = {"nasa": lambda: fetch_nasa(lat, lon), "era5": lambda: fetch_era5(lat, lon)}
        if isd_id:
            sources["isd"] = lambda: fetch_isd(isd_id)
        got = {}
        for name, fn in sources.items():
            try:
                s = fn()
                if not len(s):
                    print(f"[{site}/{name}] no data returned"); continue
                s.to_csv(f"tmin_{site}_{name}.csv", header=["tmin_c"], index_label="date")
                got[name] = s
                c = count_events(s)
                c.insert(0, "source", name); c.insert(0, "site", site)
                summary.append(c)
                print(f"\n=== {site.upper()} / {name}  ({s.index.min().date()} to {s.index.max().date()}, n={len(s)}) ===")
                print(c.drop(columns=["site", "source"]).to_string(index=False))
            except Exception as e:
                print(f"\n=== {site.upper()} / {name} FAILED: {e}", file=sys.stderr)
            time.sleep(1)
        if "isd" in got:                           # gridded-vs-station bias on cold days
            for other in ("nasa", "era5"):
                if other in got:
                    j = pd.concat([got["isd"], got[other]], axis=1, join="inner").dropna()
                    j.columns = ["isd", other]
                    cold = j[j["isd"] < 10]
                    print(f"[{site}] {other}-isd mean diff: all days {(j[other]-j['isd']).mean():.2f} C; "
                          f"cold days {(cold[other]-cold['isd']).mean():.2f} C")
    if summary:
        out = pd.concat(summary)
        out.to_csv("feasibility_summary.csv", index=False)
        print("\n##### COMPARISON AT 4C and 3C (send me this block) #####")
        print(out[out.threshold_C.isin([3, 4])].to_string(index=False))