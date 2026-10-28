#!/usr/bin/env python3
"""
Build the shippable India (Tier 3) dataset with rule-derived cold-night labels.

    python scripts/build_india_dataset.py

Reads   data/raw/india/winter_data.csv.gz  and  data/raw/india/station_summary.csv
        (outputs of scripts/gsod_download.py summarize)
Writes  data/processed/india_tier3.csv.gz   one row per station-day, features + labels + split
        reports/india_dataset_card.md       documentation generated FROM the data

See src/india_dataset.py for the design decisions and their reasons.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from india_dataset import (  # noqa: E402
    BASE_FEATURES, LABEL_THRESHOLDS_C, PLUS_WIND, PRESSURE_FEATURES, TRAIN_END_SEASON,
    VAL_END_SEASON, WIND_FEATURES, build_india_dataset,
)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def data_card(d):
    p = "label_le4"
    by_split = (d.groupby("split").agg(rows=(p, "size"), stations=("STATION", "nunique"),
                                       seasons=("season", "nunique"), cold_nights_le4=(p, "sum"),
                                       strict_le4=("label_strict_le4", "sum"))
                .reindex(["train", "val", "test"]).reset_index())
    by_split["prevalence_%"] = (100 * by_split["cold_nights_le4"] / by_split["rows"]).round(2)
    by_role = (d.groupby("role").agg(stations=("STATION", "nunique"), rows=(p, "size"),
                                     cold_nights_le4=(p, "sum")).reset_index())
    by_role["prevalence_%"] = (100 * by_role["cold_nights_le4"] / by_role["rows"]).round(2)
    st = (d.groupby(["name", "role"]).agg(rows=(p, "size"), seasons=("season", "nunique"),
                                          le3=("label_le3", "sum"), le4=(p, "sum"), le5=("label_le5", "sum"),
                                          strict_le4=("label_strict_le4", "sum"),
                                          slp_avail_pct=("slp_change_1d", lambda x: round(100 * x.notna().mean())))
          .reset_index().sort_values("le4", ascending=False))
    st["prevalence_le4_%"] = (100 * st["le4"] / st["rows"]).round(2)
    test = d[d["split"] == "test"].groupby("name")[p].agg(["size", "sum"]).rename(
        columns={"size": "test_rows", "sum": "test_le4"}).reset_index()
    top = st.iloc[0]
    share = 100 * top["le4"] / st["le4"].sum()
    return f"""# India Tier 3 dataset card (`data/processed/india_tier3.csv.gz`)

**What it is.** One row per station-day (Nov-Mar, 1990-2024) from NOAA GSOD records of
{d['STATION'].nunique()} Indian stations. Features describe day *D*; the target is the **next day's
minimum temperature**, and the labels apply the project's rule to it.

**Labels are rule-derived, not observed frost or crop damage.** Results on them measure skill at
forecasting the rule's conditions. Nothing here validates the rule against observed damage.

| Column | Meaning |
|---|---|
| `target_tmin_c` | next day's Tmin (C), explicit-minimum days only |
| `label_le3`, `label_le4`, `label_le5` | `target_tmin_c <= 3 / 4 / 5` (inclusive). **Primary: `label_le4`** (IMD plains Cold Wave criterion; Attri et al. 2020 frost day) |
| `label_strict_le4` | `label_le4` AND Td < 0 C, with Td from Tmin and daily-mean RH (Omazic et al. 2024 form). RHmean is itself derived from GSOD mean temperature and mean dew point, so this is an approximation |
| `split` | by winter season: train <= {TRAIN_END_SEASON}, val {TRAIN_END_SEASON + 1}-{VAL_END_SEASON}, test >= {VAL_END_SEASON + 1} |
| `STATION`, `name`, `role` | grouping and bookkeeping only. **Never model features** |
| `today_min_derived` | today's Tmin was derived from synoptic readings (warm-biased); metadata only |

**Features** ({len(PLUS_WIND)} in the widest set; names follow `src/features.py`):
- core, needs a temperature + humidity sensor ({len(BASE_FEATURES)}): `{', '.join(BASE_FEATURES[:8])}`, lags 1-3 and rolling 3/7-day min/mean/max of `tmin_c` and `rh_mean_pct`, `tmin_c_roc1`, `doy_sin`, `doy_cos`
- `+ pressure` ({len(PRESSURE_FEATURES)}): `{', '.join(PRESSURE_FEATURES)}`. Pressure **changes** of sea-level pressure only, never absolute pressure. Optional (may be NaN): sea-level pressure is missing for roughly half or more of the days at Amritsar, Lucknow, Jaipur and Nagpur (see `slp_avail_pct` below). Station pressure (STP) is unusable: in this GSOD extract it is the constant 999.9 missing code
- `+ wind` ({len(WIND_FEATURES)}): `{', '.join(WIND_FEATURES)}`

## Size and prevalence

{md_table(by_split)}

{md_table(by_role)}

Prevalence is the share of station-days whose next-day Tmin is <= 4 C. It is in the rare-event
regime the Europe dry run never reached (Basel was about 30%).

## Per station

{md_table(st)}

`{top['name']}` supplies {share:.0f}% of all cold nights, so pooled results are dominated by it.
Test-period (2018+) cold nights per station:

{md_table(test.sort_values('test_le4', ascending=False))}

## Known limitations (read before using)

1. **Rule-derived labels.** No observed frost or damage. Crop-specific (Kinnow) threshold is still unsourced.
2. **Station series have breaks.** Hisar's cold-night rate is 2.8 per season in the 2000s and 19.2 in the 2010s while NASA POWER and ERA5 do not jump with it; Amritsar's declines 47 -> 33 -> 17 while the grids rise slightly. Treat station labels as an imperfect reference, not truth.
3. **Gridded sources run warm in the cold tail** by about +0.5 to +3.4 C (station-matched, `data/raw/india/grid_threshold_calibration.csv`), so a gridded label at the same threshold undercounts.
4. **Derived minima excluded as targets.** This removes most 1990s data for Amritsar, Hisar, Bikaner and Kota; effective record is roughly 2000-2024.
5. **Sparse where it matters.** Only four usable stations in the north-west belt, none at Abohar itself (nearest: Ganganagar, 37 km, 56% derived minima).
6. **Vidarbha has no cold nights.** Nagpur: 2 nights <= 4 C in 35 winters. It is a negative control, not training signal.
7. **Not validated for Warora/Vidarbha or any single deployment site.** Pooled across stations.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winter", default="data/raw/india/winter_data.csv.gz")
    ap.add_argument("--summary", default="data/raw/india/station_summary.csv")
    ap.add_argument("--out", default="data/processed/india_tier3.csv.gz")
    ap.add_argument("--card", default="reports/india_dataset_card.md")
    a = ap.parse_args()

    winter = pd.read_csv(a.winter, dtype={"STATION": str})
    summary = pd.read_csv(a.summary, dtype={"STATION": str})
    d = build_india_dataset(winter, summary)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.card).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    Path(a.card).write_text(data_card(d), encoding="utf-8")
    print(f"{len(d):,} rows, {d['STATION'].nunique()} stations, {d.shape[1]} columns -> {a.out}")
    print(d.groupby("split")["label_le4"].agg(rows="size", cold_nights="sum").assign(
        prevalence_pct=lambda x: (100 * x["cold_nights"] / x["rows"]).round(2)).to_string())
    print(f"data card -> {a.card}")


if __name__ == "__main__":
    main()
